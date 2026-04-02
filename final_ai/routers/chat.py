import os
import json
import asyncio
import threading
import uuid
from decimal import Decimal
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from final_ai.observability import traceable
from final_ai.pipeline.chatbot_graph import build_graph
from final_ai.pipeline.utils import (
    RequestCancelled,
    bind_request_cancel_event,
    bind_request_trace,
    execute_traced,
    get_db_connection,
    trace_block,
    trace_log,
    update_request_trace,
)
from final_ai.schemas.chat import ChatRequest

router = APIRouter()
GRAPH_TIMEOUT_SECONDS = float(os.getenv("CHAT_GRAPH_TIMEOUT_SECONDS", "45"))
SSE_HEARTBEAT_INTERVAL_SECONDS = float(os.getenv("CHAT_SSE_HEARTBEAT_INTERVAL_SECONDS", "5"))
GRAPH_TIMEOUT_MESSAGE = os.getenv(
    "CHAT_GRAPH_TIMEOUT_MESSAGE",
    "응답 생성이 지연되고 있습니다. 잠시 후 다시 시도해 주세요.",
)

# 앱 기동 시 한 번만 빌드 (MemorySaver 포함)
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


@traceable(name="tailtalk_fastapi_chat_graph", run_type="chain")
def _invoke_graph(initial_state: dict, config: dict) -> dict:
    with trace_block(
        "graph_invoke",
        thread_id=config.get("configurable", {}).get("thread_id"),
        message_chars=len(initial_state.get("user_input") or ""),
    ):
        result = get_graph().invoke(initial_state, config=config)
    trace_log(
        "graph_invoke_result",
        response_chars=len(result.get("response") or ""),
        product_cards=len(result.get("product_cards") or []),
    )
    return result

def _json_default(value):
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    return str(value)


def _sse(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False, default=_json_default)}\n\n"

async def _stream(req: ChatRequest, request: Request):
    request_id = getattr(request.state, "request_id", None) or request.headers.get("x-request-id") or str(uuid.uuid4())
    with bind_request_trace(
        request_id,
        thread_id=req.thread_id,
        user_id=req.user_id,
        target_pet_id=req.target_pet_id,
        path=str(request.url.path),
    ):
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        initial_state = {
            "user_input": req.message,
            "pet_profile": req.pet_profile,
            "health_concerns": req.health_concerns,
            "allergies": req.allergies,
            "food_preferences": req.food_preferences,
            "user_id": req.user_id,
            "target_pet_id": req.target_pet_id,
        }

        config = {"configurable": {"thread_id": req.thread_id}}
        trace_log(
            "chat_request_received",
            message_chars=len(req.message or ""),
            health_concerns=len(req.health_concerns or []),
            allergies=len(req.allergies or []),
            food_preferences=len(req.food_preferences or []),
        )

        pet_name = "우리 아이"
        if req.user_id:
            conn = None
            try:
                import psycopg2.extras

                conn = get_db_connection()
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    if req.target_pet_id:
                        execute_traced(
                            cur,
                            "chat_router_pet_name_target",
                            "SELECT name FROM pet WHERE user_id = %s AND pet_id = %s LIMIT 1",
                            (req.user_id, req.target_pet_id),
                            user_id=req.user_id,
                            pet_id=req.target_pet_id,
                        )
                    else:
                        execute_traced(
                            cur,
                            "chat_router_pet_name_latest",
                            "SELECT name FROM pet WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
                            (req.user_id,),
                            user_id=req.user_id,
                        )
                    row = cur.fetchone()
                    if row and row["name"]:
                        pet_name = row["name"]
                        update_request_trace(pet_name=pet_name)
            except Exception as exc:
                trace_log(
                    "chat_router_pet_name_error",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            finally:
                if conn is not None:
                    conn.close()

        category = "상품"
        if req.message:
            if "사료" in req.message:
                category = "사료"
            elif "간식" in req.message:
                category = "간식"
            elif "영양제" in req.message:
                category = "영양제"
            elif "용품" in req.message:
                category = "용품"
        update_request_trace(category=category)

        trace_log("chat_stream_info", pet_name=pet_name, category=category)
        yield _sse("info", {"content": f"{pet_name}에 어울리는 {category}를 찾는 중입니다..."})

        cancel_event = threading.Event()
        first_token_logged = False
        next_heartbeat_at = started_at + SSE_HEARTBEAT_INTERVAL_SECONDS
        try:
            with bind_request_cancel_event(cancel_event):
                graph_task = asyncio.create_task(
                    asyncio.to_thread(_invoke_graph, initial_state, config),
                )

                while not graph_task.done():
                    now = loop.time()
                    if await request.is_disconnected():
                        cancel_event.set()
                        graph_task.cancel()
                        trace_log("chat_client_disconnected", stage="graph_wait")
                        return

                    elapsed_ms = round((now - started_at) * 1000, 1)
                    if now - started_at >= GRAPH_TIMEOUT_SECONDS:
                        cancel_event.set()
                        graph_task.cancel()
                        trace_log(
                            "chat_graph_timeout",
                            elapsed_ms=elapsed_ms,
                            timeout_s=GRAPH_TIMEOUT_SECONDS,
                        )
                        yield _sse("error", {"message": GRAPH_TIMEOUT_MESSAGE})
                        return

                    if now >= next_heartbeat_at:
                        trace_log(
                            "chat_stream_keepalive",
                            elapsed_ms=elapsed_ms,
                            interval_s=SSE_HEARTBEAT_INTERVAL_SECONDS,
                        )
                        yield ": keepalive\n\n"
                        next_heartbeat_at = now + SSE_HEARTBEAT_INTERVAL_SECONDS

                    await asyncio.sleep(0.25)

                final_state = await graph_task
        except RequestCancelled:
            trace_log("chat_request_cancelled")
            return
        except Exception as exc:
            trace_log(
                "chat_graph_error",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            yield _sse("error", {"message": str(exc)})
            return

        response_text = final_state.get("response", "")
        product_cards = final_state.get("product_cards", [])
        trace_log(
            "chat_graph_completed",
            elapsed_ms=round((loop.time() - started_at) * 1000, 1),
            response_chars=len(response_text),
            product_cards=len(product_cards),
        )

        words = response_text.split(" ")
        for i, word in enumerate(words):
            if await request.is_disconnected():
                cancel_event.set()
                trace_log("chat_client_disconnected", stage="token_stream", token_index=i)
                return
            chunk = word if i == 0 else " " + word
            if not first_token_logged:
                trace_log(
                    "chat_first_token",
                    elapsed_ms=round((loop.time() - started_at) * 1000, 1),
                    chunk_chars=len(chunk),
                )
                first_token_logged = True
            yield _sse("token", {"content": chunk})
            await asyncio.sleep(0.01)

        trace_log("chat_products_emitted", product_cards=len(product_cards))
        yield _sse("products", {"cards": product_cards})

        trace_log(
            "chat_done",
            elapsed_ms=round((loop.time() - started_at) * 1000, 1),
        )
        yield _sse("done", {})

# 1. POST / (기본 채팅)
@router.post("/")
async def chat(req: ChatRequest, request: Request):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    return StreamingResponse(
        _stream(req, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-Id": request_id,
        },
    )

# 2. POST /sessions/ (Django의 세션 생성 대응)
class SessionCreateRequest(BaseModel):
    title: Optional[str] = None
    target_pet_id: Optional[str] = None

@router.post("/sessions/")
async def create_session(req: SessionCreateRequest):
    # 실제 DB 세션 생성은 Django가 담당하므로, 여기서는 호환성을 위해 ID만 반환
    session_id = str(uuid.uuid4())
    return {
        "session_id": session_id,
        "title": req.title or "새 대화",
        "display_date": "오늘"
    }

# 3. POST /sessions/{session_id}/messages/ (Django의 메시지 전송 대응)
@router.post("/sessions/{session_id}/messages/")
async def session_chat(session_id: str, req: ChatRequest, request: Request):
    # thread_id를 장고의 session_id로 고정하여 상태 유지
    req.thread_id = session_id
    return await chat(req, request)

# 4. GET /sessions/{session_id}/messages/ (Django의 메시지 조회 대응)
@router.get("/sessions/{session_id}/messages/")
async def get_messages(session_id: str):
    # LangGraph의 checkpoint에서 내역을 가져올 수도 있으나, 
    # 현재는 프론트엔드 UI를 위해 빈 배열 또는 기본 인사를 반환
    return {"messages": []}

# 5. DELETE /sessions/{session_id}/ (Django의 세션 삭제 대응)
@router.delete("/sessions/{session_id}/")
async def delete_session(session_id: str):
    return {"status": "success"}
