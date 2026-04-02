from langchain_core.messages import AIMessage
from final_ai.observability import traceable
from final_ai.pipeline.state import ChatState
from final_ai.pipeline.utils import (
    LLM_MODEL,
    build_pet_context,
    create_llm_completion,
    ensure_request_active,
    get_user_pets,
    trace_log,
    translate_health_concerns,
)

# RESPOND_SYSTEM = """\
# 당신은 반려동물 쇼핑 서비스의 친절한 AI 어시스턴트입니다.
# 사용자의 펫 정보와 품종별 건강 지식을 결합하여 개인화된 답변을 제공합니다.

# [답변 형식 규칙]
# 1. 추천 의도(recommend)가 포함된 경우 반드시 **섹션 사이마다 빈 줄(Enter 2번)**을 넣어 다음 형식을 엄격히 준수하세요:

#    "{펫이름}에 어울리는 상품을 추천드릴게요.
   
#    등록된 건강 관심사 : {건강관심사} (값이 있는 경우에만 표시)
   
#    종에 해당하는 건강특징: {제공된_건강특징}
   
#    상품 추천 이유:
#    1. {첫 번째 추천 이유: 건강특징과 상품 간의 연관성 설명}
#    2. {두 번째 추천 이유}
   
#    추천 상품을 확인해 주세요!"

# 2. 도메인 지식 답변(domain_qa)이 포함된 경우:
#    - 관련 지식을 친절하게 설명하고 섹션 구분 시 반드시 빈 줄(Enter 2번)을 활용하세요.
#    - 답변 서두에 등록된 건강 관심사가 있다면 "등록된 건강 관심사 : {건강관심사}" 형식을 포함하세요.

# 3. 일반 지점:
#    - 답변의 각 주요 단락 사이에는 **반드시 한 줄의 빈 줄**을 넣어 가독성을 높이세요.
#    - 상품 목록 자체는 언급하지 마세요 (우측 패널에 표시됨).
# """
RESPOND_SYSTEM = """\
당신은 반려동물 쇼핑 서비스의 친절한 AI 어시스턴트입니다.
사용자의 펫 정보와 품종별 건강 지식을 결합하여 개인화된 답변을 제공합니다.

[답변 형식 규칙]
1. 추천 의도(recommend)가 포함된 경우 반드시 **섹션 사이마다 빈 줄(Enter 2번)**을 넣어 다음 형식을 엄격히 준수하세요:

   "{펫이름}에 어울리는 상품을 추천드릴게요.
   
   추천 상품을 확인해 주세요!"

2. 도메인 지식 답변(domain_qa)이 포함된 경우:
   - 관련 지식을 친절하게 설명하고 섹션 구분 시 반드시 빈 줄(Enter 2번)을 활용하세요.
   - 답변 서두에 등록된 건강 관심사가 있다면 "등록된 건강 관심사 : {건강관심사}" 형식을 포함하세요.

3. **펫 프로필 전환 안내**:
   - 만약 '펫 전환 발생' 정보가 있다면, 답변 서두에 "애칭 {전환된_펫이름}의 정보를 바탕으로 다시 추천해 드릴게요!" 라는 문구를 반드시 포함하세요.

4. **대기 중인 추천(Next Queue) 안내**:
   - **대기 중인 카테고리**가 있다면, 답변 마지막에 "혹시 {다음_카테고리} 추천 상품도 바로 보여드릴까요?" 라는 질문을 던지세요.
   - 대기 중인 카테고리가 없고 **대기 중인 펫**이 있다면, 답변 마지막에 "혹시 {다음_펫이름}의 추천 상품도 바로 보여드릴까요?" 라는 질문을 던지세요.

5. 일반 지점:
   - 답변의 각 주요 단락 사이에는 **반드시 한 줄의 빈 줄**을 넣어 가독성을 높이세요.
   - 상품 목록 자체는 언급하지 마세요 (우측 패널에 표시됨).
"""


@traceable(name="respond_node", run_type="chain")
def respond_node(state: ChatState) -> dict:
    """최종 응답 생성 (LLM)"""
    if state.get("pet_mismatch"):
        return {
            "response": "펫 프로필과 다른 반려동물입니다. 펫 프로필 등록 먼저 해주세요.",
            "product_cards": []
        }
    domain_contexts  = state.get("domain_contexts")  or []
    reranked_results = state.get("reranked_results") or []
    pet_ctx          = build_pet_context(state)
    user_input       = state["user_input"]
    health_concerns  = state.get("health_concerns") or []
    
    # 펫 이름 및 카테고리 추출
    pet_profile = state.get("pet_profile") or {}
    pet_name = pet_profile.get("name")
    if not pet_name:
        breed = pet_profile.get("breed")
        pet_name = f"{breed} 아이" if breed else "우리 아이"
        
    category = (state.get("filters") or {}).get("category") or "상품"
    health_traits = state.get("health_traits") or "특별한 데이터가 없습니다."
    
    # ── 컨텍스트 조합 ───────────────────────────────────────────────────────────
    context_parts = []

    if domain_contexts:
        joined = "\n\n".join(domain_contexts[:2])
        context_parts.append(f"[도메인 지식]\n{joined}")

    if reranked_results:
        # 추천 상품들의 특징(브랜드 등)을 LLM이 알 수 있도록 전달
        products_info = "\n".join([f"- {p.get('brand_name')} {p.get('goods_name')}" for p in reranked_results[:3]])
        context_parts.append(f"[추천 상품 후보]\n{products_info}")

    context_block = "\n\n".join(context_parts) if context_parts else "검색된 정보가 없습니다."

    translated_concerns = translate_health_concerns(health_concerns)
    
    # 대기 중인 펫 이름 및 카테고리 조회
    pending_ids = state.get("pending_pet_ids") or []
    pending_names = []
    if pending_ids and state.get("user_id"):
        all_pets = get_user_pets(state["user_id"])
        pending_names = [p["name"] for p in all_pets if p["pet_id"] in pending_ids]
    
    pending_cats = state.get("pending_categories") or []

    user_msg = (
        f"현재 상황 정보:\n"
        f"- 펫 이름: {pet_name}\n"
        f"- 펫 전환 발생: {'YES' if state.get('is_pet_switched') else 'NO'}\n"
        f"- 전환된 펫 이름: {state.get('switched_pet_name') or 'N/A'}\n"
        f"- 대기 중인 펫 목록: {', '.join(pending_names) if pending_names else '없음'}\n"
        f"- 대기 중인 카테고리: {', '.join(pending_cats) if pending_cats else '없음'}\n"
        f"- 다음_카테고리: {pending_cats[0] if pending_cats else 'N/A'}\n"
        f"- 카테고리: {category}\n"
        f"- 등록된 건강 관심사: {', '.join(translated_concerns) if translated_concerns else '없음'}\n"
        f"- 건강 특징: {health_traits}\n"
        f"- 전체 펫 정보: {pet_ctx}\n\n"
        f"사용자 질문: {user_input}\n\n"
        f"참고 데이터:\n{context_block}"
    )


    try:
        ensure_request_active()
        response = create_llm_completion(
            trace_label="respond_node_response",
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": RESPOND_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
        ).choices[0].message.content.strip()
    except Exception as e:
        if reranked_results:
            response = f"{pet_name}에 어울리는 {category} 후보를 찾았어요.\n\n추천 상품을 확인해 주세요!"
        elif domain_contexts:
            response = "관련 정보를 찾았지만 답변 생성 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
        else:
            response = "지금은 추천 정보를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."
        trace_log("respond_node_fallback", error_type=type(e).__name__, error=str(e))

    trace_log(
        "respond_node_result",
        response_chars=len(response),
        product_cards=len(reranked_results),
        domain_contexts=len(domain_contexts),
    )
    return {
        "messages": [AIMessage(content=response)],
        "response": response,
    }
