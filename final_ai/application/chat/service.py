import asyncio
import threading
from collections.abc import AsyncIterator

from fastapi import Request

from final_ai.application.chat.context import ChatContextLoadError, hydrate_chat_request
from final_ai.application.chat.memory import build_memory_payload
from final_ai.api.dependencies import RequestAuthContext, RequestCancelled, bind_request_cancel_event
from final_ai.application.chat.graph_service import invoke_chat_graph
from final_ai.application.chat.dto import build_chat_execution_request
from final_ai.contracts.chat import ChatRequest
from final_ai.contracts.sse import ChatEventType
from final_ai.domain.profile.service import get_pet_name_for_user
from final_ai.infrastructure.observability import get_logger

logger = get_logger(__name__)


def _infer_category(message: str | None) -> str:
    if not message:
        return "상품"
    if "사료" in message:
        return "사료"
    if "간식" in message:
        return "간식"
    if "영양제" in message:
        return "영양제"
    if "용품" in message:
        return "용품"
    return "상품"


async def _stream_response_tokens(
    response_text: str,
    request: Request,
    cancel_event,
) -> AsyncIterator[tuple[ChatEventType, dict]]:
    if not response_text:
        return

    words = response_text.split(" ")
    for index, word in enumerate(words):
        if await request.is_disconnected():
            cancel_event.set()
            return

        chunk = word if index == 0 else " " + word
        yield "token", {"content": chunk}
        await asyncio.sleep(0.01)


async def stream_chat_events(req: ChatRequest, request: Request) -> AsyncIterator[tuple[ChatEventType, dict]]:
    auth_context: RequestAuthContext | None = getattr(request.state, "auth_context", None)
    try:
        req = hydrate_chat_request(req)
    except ChatContextLoadError as exc:
        logger.warning("chat context load failed: %s", exc)
        yield "error", {"message": "대화 문맥을 불러오지 못했습니다. 다시 시도해 주세요."}
        return

    execution_request = build_chat_execution_request(req)
    log_extra = (
        auth_context.log_extra(thread_id=req.thread_id, target_pet_id=req.target_pet_id)
        if auth_context
        else {
            "request_id": getattr(req, "request_id", "") or "",
            "session_id": req.thread_id,
            "user_id": req.user_id or "",
            "target_pet_id": req.target_pet_id or "",
        }
    )

    pet_name = get_pet_name_for_user(req.user_id, req.target_pet_id) or "반려동물"
    category = _infer_category(req.message)
    logger.info("chat stream started", extra=log_extra)
    yield "info", {"content": f"{pet_name}에 어울리는 {category} 조건을 확인하는 중입니다..."}

    cancel_event = threading.Event()
    progress_queue: asyncio.Queue[tuple[ChatEventType, dict]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def enqueue_progress(payload: dict):
        loop.call_soon_threadsafe(progress_queue.put_nowait, ("info", payload))

    try:
        with bind_request_cancel_event(cancel_event):
            graph_task = asyncio.create_task(
                asyncio.to_thread(
                    invoke_chat_graph,
                    execution_request.initial_state,
                    execution_request.config,
                    enqueue_progress,
                ),
            )

            while not graph_task.done():
                if await request.is_disconnected():
                    cancel_event.set()
                    graph_task.cancel()
                    logger.info("chat stream disconnected", extra=log_extra)
                    return
                try:
                    while True:
                        yield progress_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                await asyncio.sleep(0.25)

            try:
                while True:
                    yield progress_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            final_state = await graph_task
    except RequestCancelled:
        logger.info("chat stream cancelled", extra=log_extra)
        return
    except Exception:
        logger.exception("chat stream failed", extra=log_extra)
        yield "error", {"message": "응답 생성 중 오류가 발생했습니다."}
        return

    response_text = final_state.get("response", "")
    product_cards = final_state.get("product_cards", []) or []

    async for event in _stream_response_tokens(response_text, request, cancel_event):
        yield event

    logger.info(
        "chat stream completed",
        extra={**log_extra, "product_count": len(product_cards)},
    )
    memory_payload = build_memory_payload(final_state)
    yield "products", {"cards": product_cards}
    yield "final", {
        "message": response_text,
        "cards": product_cards,
        "meta": {
            "request_id": log_extra.get("request_id", ""),
            "session_id": log_extra.get("session_id", ""),
        },
        "memory": memory_payload,
    }
    yield "done", {}
