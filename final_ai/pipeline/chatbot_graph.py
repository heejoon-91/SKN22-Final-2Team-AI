from functools import wraps

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from final_ai.pipeline.nodes.clarify import clarify_node
from final_ai.pipeline.nodes.domain_qa import general_node, rag_node
from final_ai.pipeline.nodes.intent import intent_node
from final_ai.pipeline.nodes.merge import merge_node
from final_ai.pipeline.nodes.recommend import (
    profile_node,
    query_node,
    rerank_node,
    search_node,
)
from final_ai.pipeline.nodes.respond import respond_node
from final_ai.pipeline.state import ChatState
from final_ai.pipeline.utils import trace_block, trace_log


def _summarize_state(state: ChatState) -> dict:
    filters = state.get("filters") or {}
    return {
        "intents": state.get("intents"),
        "category": filters.get("category"),
        "subcategory": filters.get("subcategory"),
        "pet_type": filters.get("pet_type"),
        "user_input_chars": len(state.get("user_input") or ""),
        "search_results": len(state.get("search_results") or []),
        "reranked_results": len(state.get("reranked_results") or []),
        "domain_contexts": len(state.get("domain_contexts") or []),
        "product_cards": len(state.get("product_cards") or []),
        "filter_relaxation_count": state.get("filter_relaxation_count"),
        "clarification_count": state.get("clarification_count"),
        "pet_mismatch": state.get("pet_mismatch"),
    }


def _trace_node(node_name: str, func):
    @wraps(func)
    def wrapper(state: ChatState):
        trace_log("graph_node_enter", node=node_name, **_summarize_state(state))
        with trace_block("graph_node", node=node_name):
            result = func(state)
        if isinstance(result, dict):
            trace_log(
                "graph_node_exit",
                node=node_name,
                response_chars=len(result.get("response") or ""),
                search_results=len(result.get("search_results") or []),
                reranked_results=len(result.get("reranked_results") or []),
                domain_contexts=len(result.get("domain_contexts") or []),
                product_cards=len(result.get("product_cards") or []),
                recommend_retry_pending=result.get("recommend_retry_pending"),
            )
        else:
            trace_log("graph_node_exit", node=node_name, result_type=type(result).__name__)
        return result

    return wrapper


# ── 라우팅 함수 ────────────────────────────────────────────────────────────────

def route_intent(state: ChatState):
    """
    INTENT → 조건부 분기.
    """
    intents = state.get("intents") or ["unclear"]
    filters = state.get("filters") or {}
    pet_profile = state.get("pet_profile") or {}
    relaxation = state.get("filter_relaxation_count", 0)

    trace_log(
        "route_intent_eval",
        intents=intents,
        pet_type=filters.get("pet_type"),
        species=pet_profile.get("species"),
        category=filters.get("category"),
        relaxation=relaxation,
    )

    if "unclear" in intents:
        return "clarify"

    has_domain    = "domain_qa"  in intents
    has_recommend = "recommend"  in intents

    # recommend 의도인데 필수 정보(종류 또는 카테고리)가 없는 경우 재질문
    if has_recommend:
        # pet_type이 AI 추출 결과에도 없고, 펫 프로필에도 없는 경우
        if not filters.get("pet_type") and not pet_profile.get("species"):
            return "clarify"
        # form_hint가 있으면 사료/간식 구분 재질문 필요
        if state.get("form_hint"):
            return "clarify"
        # 카테고리가 없는 경우 (필터 완화 중이 아닐 때만)
        if not filters.get("category") and relaxation == 0:
            return "clarify"

    if has_domain and has_recommend:
        return [Send("general", state), Send("profile", state)]
    elif has_domain:
        return "general"
    elif has_recommend:
        return "profile"
    else:
        return "clarify"


def route_rerank(state: ChatState) -> str:
    """
    RERANK → QUERY (필터 완화 재검색) or MERGE.
    완화 재검색이 필요할 때만 QUERY로 순환한다.
    """
    results = state.get("reranked_results") or []
    relaxation = state.get("filter_relaxation_count", 0)
    retry_pending = bool(state.get("recommend_retry_pending"))

    if retry_pending:
        trace_log(
            "route_rerank_retry",
            result_count=len(results),
            relaxation=relaxation,
            retry_pending=retry_pending,
        )
        return "query"
    trace_log(
        "route_rerank_merge",
        result_count=len(results),
        relaxation=relaxation,
        retry_pending=retry_pending,
    )
    return "merge"


# ── 그래프 빌드 ───────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    g = StateGraph(ChatState)

    # 노드 등록
    g.add_node("intent", _trace_node("intent", intent_node))
    g.add_node("clarify", _trace_node("clarify", clarify_node))
    g.add_node("general", _trace_node("general", general_node))
    g.add_node("rag", _trace_node("rag", rag_node))
    g.add_node("profile", _trace_node("profile", profile_node))
    g.add_node("query", _trace_node("query", query_node))
    g.add_node("search", _trace_node("search", search_node))
    g.add_node("rerank", _trace_node("rerank", rerank_node))
    g.add_node("merge", _trace_node("merge", merge_node))
    g.add_node("respond", _trace_node("respond", respond_node))

    # 엣지
    g.add_edge(START, "intent")

    g.add_conditional_edges(
        "intent",
        route_intent,
        {
            "clarify": "clarify",
            "general": "general",
            "profile": "profile",
        },
    )

    g.add_edge("clarify", END)

    # domain_qa 서브플로우
    g.add_edge("general", "rag")
    g.add_edge("rag",     "merge")

    # recommend 서브플로우
    def route_profile_node(state: ChatState):
        if state.get("pet_mismatch"):
            return "merge"
        return "query"

    g.add_conditional_edges(
        "profile",
        route_profile_node,
        {
            "merge": "merge",
            "query": "query",
        }
    )
    g.add_edge("query",   "search")
    g.add_edge("search",  "rerank")

    g.add_conditional_edges(
        "rerank",
        route_rerank,
        {
            "query": "query",
            "merge": "merge",
        },
    )

    g.add_edge("merge",   "respond")
    g.add_edge("respond", END)

    cp = checkpointer or MemorySaver()
    return g.compile(checkpointer=cp)


# ── 싱글턴 인스턴스 ───────────────────────────────────────────────────────────
graph = build_graph()


# ── 간편 실행 헬퍼 ────────────────────────────────────────────────────────────

def chat(
    user_input: str,
    thread_id: str = "default",
    pet_profile: dict | None = None,
    health_concerns: list[str] | None = None,
    allergies: list[str] | None = None,
    food_preferences: list[str] | None = None,
    user_id: str | None = None,
    target_pet_id: str | None = None,
) -> dict:
    """
    단일 턴 실행 헬퍼.
    같은 thread_id로 반복 호출하면 MemorySaver가 대화 히스토리를 유지한다.
    """
    config = {"configurable": {"thread_id": thread_id}}
    init_state = {
        "user_input":       user_input,
        "messages":         [],
        "pet_profile":      pet_profile,
        "health_concerns":  health_concerns or [],
        "allergies":        allergies       or [],
        "food_preferences": food_preferences or [],
        "user_id":          user_id,
        "target_pet_id":    target_pet_id,
        # 초기화 필드 (매 턴 초기화되지 않도록 필요한 것만 포함)
        "breed_context":           None,
        "search_results":          [],
        "reranked_results":        [],
        "domain_contexts":         [],
        "product_cards":           [],
        "filter_relaxation_count": 0,
        "recommend_retry_pending": False,
        "clarification_count":     0,
        "intents":                 [],
        "is_pet_override":         False,
        "pet_mismatch":            False,
    }
    result = graph.invoke(init_state, config=config)
    return {
        "response":      result.get("response", ""),
        "product_cards": result.get("product_cards", []),
    }
