from pathlib import Path

from final_ai.observability import traceable
from final_ai.pipeline.state import ChatState
from final_ai.pipeline.utils import (
    DOMAIN_INTENT_TO_CATEGORY,
    LLM_MODEL,
    build_pet_context,
    create_llm_completion,
    ensure_request_active,
    trace_log,
)


@traceable(name="search_domain_context", run_type="retriever")
def _search_domain_pg(query: str, domain_intent: str | None, species: str | None) -> list[str]:
    """
    데이터 CSV 기반 QA를 PostgreSQL을 통해 검색하거나, 
    키워드 기반으로 직접 CSV 파일에서 검색합니다.
    """
    import pandas as pd
    from pathlib import Path

    # CSV 파일 경로 (test/data → pipeline/data 순으로 탐색)
    base = Path(__file__).resolve().parents[3]
    csv_candidates = [
        base / "test" / "data" / "merged_QnA_final.csv",
        Path(__file__).resolve().parents[1] / "data" / "merged_QnA_final.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)

    if csv_path is None:
        print("[RAG] CSV 파일을 찾을 수 없습니다.")
        return []
    trace_log(
        "domain_csv_selected",
        csv_path=str(csv_path),
        domain_intent=domain_intent,
        species=species,
    )

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[RAG] CSV 로드 실패: {e}")
        return []

    # 종 필터 (강아지/고양이)
    if species:
        species_kr = "강아지" if species == "dog" else "고양이"
        if "분류" in df.columns:
            df = df[df["분류"].astype(str).str.contains(species_kr, na=False)]

    # 키워드 기반 매칭
    keywords = [k for k in query.split() if len(k) > 1]
    if keywords and "질문" in df.columns:
        mask = df["질문"].str.contains("|".join(keywords), na=False)
        candidates = df[mask].head(5)
    else:
        candidates = df.head(5)

    contexts = []
    for _, row in candidates.iterrows():
        q = row.get("질문", "")
        a = row.get("답변", "")
        if q or a:
            contexts.append(f"질문: {q}\n답변: {a}")

    trace_log(
        "domain_csv_result",
        csv_name=csv_path.name,
        context_count=len(contexts),
        keyword_count=len(keywords),
    )
    return contexts


@traceable(name="general_node", run_type="chain")
def general_node(state: ChatState) -> dict:
    """쿼리 정제: 모호한 질문을 펫 프로필 기반으로 검색 최적화"""
    pet_ctx = build_pet_context(state)
    prompt  = (
        f"다음 질문을 반려동물 정보를 반영해 검색에 최적화된 한 문장으로 재작성하세요.\n"
        f"펫 정보: {pet_ctx}\n질문: {state['user_input']}"
    )
    ensure_request_active()
    refined = create_llm_completion(
        trace_label="general_node_refine_query",
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    ).choices[0].message.content.strip()

    trace_log("general_node_result", query=refined)
    return {"search_query": refined}


@traceable(name="rag_node", run_type="chain")
def rag_node(state: ChatState) -> dict:
    """CSV 기반 domain_qna 검색"""
    query         = state.get("search_query") or state["user_input"]
    domain_intent = state.get("domain_intent")
    species       = (state.get("pet_profile") or {}).get("species")

    contexts = _search_domain_pg(query, domain_intent, species)
    trace_log(
        "rag_node_result",
        query=query,
        domain_intent=domain_intent,
        species=species,
        context_count=len(contexts),
    )
    return {"domain_contexts": contexts}
