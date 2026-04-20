import json
import sys
from collections.abc import Callable

from final_ai.application.chat.search_progress import build_search_progress_messages
from final_ai.graph.builder import build_graph
from final_ai.infrastructure.observability import traceable, get_logger

logger = get_logger(__name__)
_graph = None

def get_chat_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

def _pretty_print(msg: str, indent: int = 0):
    """표준 로그 포맷을 우회하여 터미널에 직접 깨끗하게 출력"""
    prefix = " " * indent
    sys.stdout.write(f"{prefix}{msg}\n")
    sys.stdout.flush()

@traceable(name="tailtalk_fastapi_chat_graph", run_type="chain")
def invoke_chat_graph(
    initial_state: dict,
    config: dict,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    graph = get_chat_graph()
    final_state = {}
    WIDTH = 70
    emitted_search_progress = False

    # 시작 알림 박스
    _pretty_print("")
    _pretty_print("┏" + "━" * (WIDTH - 2) + "┓")
    _pretty_print("┃ 🚀 [START] GRAPH EXECUTION")
    user_input = initial_state.get("user_input", "")
    input_display = (user_input[:WIDTH-15] + "..") if len(user_input) > WIDTH-15 else user_input
    _pretty_print(f"┃ 💬 Input: {input_display}")
    _pretty_print("┗" + "━" * (WIDTH - 2) + "┛")

    # stream 모드를 'updates'로 설정하여 각 노드 완료 시마다 이벤트 수신
    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, updates in event.items():
            _pretty_print(f"  ──▶ [NODE: {node_name.upper()}] 완료")
            next_state = {**final_state, **updates}
            
            # 출력할 주요 데이터
            if "intents" in updates:
                _pretty_print(f"      └─ 의도: {updates['intents']}")
            
            if "health_concerns" in updates:
                _pretty_print(f"      └─ 건강고민: {updates['health_concerns']}")
                
            if "filters" in updates:
                f = updates["filters"]
                f_str = f"펫:{f.get('pet_type')}, 카테:{f.get('category')}, 서브:{f.get('subcategory')}"
                _pretty_print(f"      └─ 필터: [{f_str}]")

            if "exclusions" in updates:
                _pretty_print(f"      └─ 제외조건: {json.dumps(updates['exclusions'], ensure_ascii=False)}")
            
            if "search_query" in updates:
                _pretty_print(f"      └─ 검색어: {updates['search_query']}")
            
            if "search_results" in updates:
                _pretty_print(f"      └─ 검색결과: {len(updates['search_results'])}개 상품 찾음")
            
            if "response" in updates:
                res = updates['response'].replace("\n", " ")
                res_display = (res[:WIDTH-20] + "..") if len(res) > WIDTH-20 else res
                _pretty_print(f"      └─ 응답: {res_display}")

            if (
                progress_callback
                and not emitted_search_progress
                and node_name == "query"
                and "search_query" in updates
            ):
                for message in build_search_progress_messages(next_state):
                    progress_callback({"content": message})
                emitted_search_progress = True

            final_state.update(updates)

    # 종료 알림 박스
    _pretty_print("┏" + "━" * (WIDTH - 2) + "┓")
    _pretty_print("┃ ✅ [END] GRAPH EXECUTION")
    _pretty_print("┗" + "━" * (WIDTH - 2) + "┛")
    _pretty_print("")

    return {**initial_state, **final_state}
