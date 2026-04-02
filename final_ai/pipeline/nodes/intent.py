import json
from pathlib import Path

from final_ai.observability import traceable
from final_ai.pipeline.state import ChatState
from final_ai.pipeline.utils import (
    LLM_MODEL,
    create_llm_completion,
    ensure_request_active,
    get_pet_full_profile,
    get_user_pets,
    trace_log,
)

CATEGORY_FILE = Path(__file__).resolve().parents[1] / "data" / "category.json"
with open(CATEGORY_FILE, encoding="utf-8") as f:
    _categories = json.load(f)

INTENT_SYSTEM = f"""
당신은 반려동물 쇼핑 서비스의 의도 분류기입니다. 사용자 입력을 분석해 JSON으로만 반환하세요.

### intents 규칙
- recommend : 상품 추천/검색. "A 중에서 B" 형태 포함.
- popularity : "인기 있는", "잘나가는", "베스트셀러", "많이 팔린" 등의 요청이 있을 때 반드시 포함. (일반적으로 recommend와 함께 사용)
- domain_qa : 반려동물 건강·사료·행동 전문 지식 질문
- unclear   : 잡담·인사·무관·의도불명 (small_talk 없음, 모두 unclear)
두 의도 동시 감지 시 복수 반환: ["recommend", "popularity"]

### 펫 정보 추출 (pet_profile)
질문에서 다음 정보를 찾아내세요 (JSON의 루트 레벨에 포함):
- pet_type: 강아지 / 고양이 / null
- breed: 품종명 / null
- age: 나이 / null
- mentioned_pet_names: 질문에 언급된 반려동물의 이름 리스트 / []
- is_next_request: 사용자가 "다음 것도 보여줘", "응 보여줘", "다른 카테고리는?" 등 대기 중인 다른 펫이나 다음 카테고리의 추천을 요청하는 긍정 답변인 경우 true / false

### domain_intent (domain_qa 포함 시)
health_disease / care_management / nutrition_diet / behavior_psychology / travel

### 카테고리 추출 ### Few-shot (문맥 활용 예시)
1. 신규: "7살 말티즈 사료 추천해줘"
   -> {{"intents":["recommend"],"pet_type":"강아지","breed":"말티즈","target_categories":["사료"]}}
2. 다중: "고양이 사료랑 간식 보여줘"
   -> {{"intents":["recommend"],"pet_type":"고양이","target_categories":["사료", "간식"]}}
3. 펫 전환: "바나나 사료 추천해줘" (바나나가 유저의 다른 펫 이름일 경우)
   -> {{"intents":["recommend"],"mentioned_pet_names":["바나나"],"target_categories":["사료"]}}
4. 고양이 캔 사료 명확: "고양이 주식캔 추천해줘"
   -> {{"intents":["recommend"],"pet_type":"고양이","target_categories":["사료"],"subcategory":"주식캔"}}
5. 후속(긍정): "응 다음 것도 보여줘"
   -> {{"intents":["recommend"],"is_next_request":true}}

### 카테고리
{json.dumps(_categories, ensure_ascii=False)}

출력: JSON only (target_categories 리스트 필수)
"""

@traceable(name="intent_node", run_type="chain")
def intent_node(state: ChatState) -> dict:
    user_input = state["user_input"]
    user_id = state.get("user_id")
    prev_intents = state.get("intents") or []
    prev_filters = state.get("filters") or {}
    prev_pet = state.get("pet_profile") or {}
    target_pet_id = state.get("target_pet_id")
    pending_pet_ids = state.get("pending_pet_ids") or []
    pending_categories = state.get("pending_categories") or []

    # 1. 사용자 펫 목록 및 컨텍스트 준비
    user_pets = get_user_pets(user_id) if user_id else []
    context = ""
    if (state.get("clarification_count", 0) > 0 or prev_intents) and user_input:
        prev_data = {
            "intents": prev_intents,
            "filters": prev_filters,
            "pet_profile": prev_pet,
            "current_pet_id": target_pet_id,
            "user_registered_pets": [p["name"] for p in user_pets]
        }
        context = f"\n이전 대화 정보 및 등록된 펫 정보: {json.dumps(prev_data, ensure_ascii=False)}"

    # 2. LLM 호출
    ensure_request_active()
    res = create_llm_completion(
        trace_label="intent_node_classification",
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": INTENT_SYSTEM + context},
            {"role": "user",   "content": user_input},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    r = json.loads(res.choices[0].message.content)
    
    new_intents = r.get("intents") or []
    mentioned_names = r.get("mentioned_pet_names") or []
    is_next_request = r.get("is_next_request", False)
    target_categories = r.get("target_categories") or []
    is_explicit_pet_info = bool(r.get("pet_type") or r.get("breed"))
    trace_log(
        "intent_node_result",
        intents=new_intents,
        target_categories=target_categories,
        mentioned_pet_names=mentioned_names,
        is_next_request=is_next_request,
        explicit_pet_info=is_explicit_pet_info,
    )

    is_pet_switched = False
    switched_pet_name = None
    overridden_metadata = {}

    # 3. 펫 전환 및 대기열 로직
    # (A) "다음" 요청 처리
    if is_next_request:
        if pending_categories:
            target_categories = [pending_categories.pop(0)]
            new_intents = ["recommend"]
        elif pending_pet_ids:
            next_id = pending_pet_ids.pop(0)
            full_p = get_pet_full_profile(next_id)
            if full_p:
                target_pet_id = next_id
                prev_pet = full_p["pet_profile"]
                overridden_metadata = {
                    "health_concerns": full_p["health_concerns"],
                    "allergies": full_p["allergies"],
                    "food_preferences": full_p["food_preferences"],
                }
                is_pet_switched, switched_pet_name = True, prev_pet.get("name")
                new_intents = ["recommend"]

    # (B) 이름 언급에 의한 프로필 전환 (바나나 -> 초코)
    elif mentioned_names:
        matched = [p for p in user_pets if p["name"] in mentioned_names]
        if matched:
            first_pet = matched[0]
            if str(first_pet["pet_id"]) != str(target_pet_id):
                full_p = get_pet_full_profile(str(first_pet["pet_id"]))
                if full_p:
                    target_pet_id = str(first_pet["pet_id"])
                    prev_pet = full_p["pet_profile"]
                    overridden_metadata = {
                        "health_concerns": full_p["health_concerns"],
                        "allergies": full_p["allergies"],
                        "food_preferences": full_p["food_preferences"],
                        "breed_context": "", "health_traits": ""
                    }
                    is_pet_switched, switched_pet_name = True, prev_pet.get("name")
                    if "recommend" not in new_intents: new_intents.append("recommend")
            # 나머지는 대기열로
            for p in matched[1:]:
                pid = str(p["pet_id"])
                if pid != target_pet_id and pid not in pending_pet_ids:
                    pending_pet_ids.append(pid)

    # (C) 품종/종 언급에 의한 일반 문맥 전환 (포메라니안 등)
    elif is_explicit_pet_info:
        # 추출된 정보가 현재 펫의 정보와 일치하는지 확인
        new_species = "dog" if r.get("pet_type") == "강아지" else "cat" if r.get("pet_type") == "고양이" else None
        new_breed = r.get("breed")
        
        current_species = prev_pet.get("species")
        current_breed = prev_pet.get("breed")
        
        # [FIX] 품종이 명시되었는데 현재 프로필과 다르면 '무조건' 프로필 해제 (일반 모드로 전환)
        is_contradictory = False
        if new_species and current_species and new_species != current_species:
            is_contradictory = True
        if new_breed and current_breed and new_breed != current_breed:
            # 품종 이름이 명확히 다르면 프로필 Context를 버림
            is_contradictory = True
            
        if is_contradictory or (new_breed and not target_pet_id):
            target_pet_id = None
            prev_pet = {} # 기존 이름 기반 프로필 초기화
            overridden_metadata = {
                "health_concerns": [], "allergies": [], "food_preferences": [],
                "breed_context": "", "health_traits": ""
            }
            print(f"[PET_CONTEXT] Switching to general breed context: {new_breed or new_species}")
            is_pet_switched = True
        else:
            print(f"[PET_CONTEXT] Matches current profile or extension: {prev_pet.get('name')}")

    # 4. 의도 및 펫 프로필 확정
    if not new_intents:
        if "recommend" in prev_intents: new_intents = ["recommend"]
        else: new_intents = ["unclear"]
    
    new_pet = dict(prev_pet)
    if is_explicit_pet_info and not target_pet_id:
        new_pet = {}
        if r.get("pet_type"): new_pet["species"] = "dog" if r["pet_type"] == "강아지" else "cat"
        if r.get("breed"): new_pet["breed"] = r["breed"]
        if r.get("age"): new_pet["age"] = r["age"]
    elif r.get("age"):
        new_pet["age"] = r["age"]

    # 5. 필터 추출 및 유지
    new_filters = {}
    
    # [FIX] 펫이 바뀌거나 새로운 카테고리가 들어오면 이전 턴의 소분류 필터를 청소함
    is_major_switch = is_pet_switched or target_categories

    # pet_type 결정 (프로필 정보가 있다면 최우선 적용)
    species = new_pet.get("species")
    if species:
        new_filters["pet_type"] = "강아지" if species == "dog" else "고양이"
    elif r.get("pet_type"):
        new_filters["pet_type"] = r["pet_type"]
    elif prev_filters.get("pet_type"):
        new_filters["pet_type"] = prev_filters["pet_type"]

    current_pet_kr = new_filters.get("pet_type") or "강아지"
    pet_cat_map = _categories.get(current_pet_kr, {})

    # 카테고리 결정
    detected_cat = None
    if target_categories:
        detected_cat = target_categories[0]
        if len(target_categories) > 1:
            for c in target_categories[1:]:
                if c not in pending_categories: pending_categories.append(c)
    elif "recommend" in new_intents and not is_major_switch:
        detected_cat = prev_filters.get("category")

    # 소분류 결정
    detected_sub = r.get("subcategory")
    if not detected_sub and "recommend" in new_intents and not is_major_switch:
        if detected_cat == prev_filters.get("category"):
            detected_sub = prev_filters.get("subcategory")

    # 키워드 기반 보정 (방금 입력에 "사료" 등이 있으면 detected_cat 업데이트)
    if not detected_sub and "recommend" in new_intents:
        found_sub, found_cat = None, None
        targets = [detected_cat] if detected_cat else pet_cat_map.keys()
        for cname in targets:
            subs = pet_cat_map.get(cname, {}).get("subcategories", [])
            for s in subs:
                kws = s.split("/") if "/" in s else [s]
                if any(k in user_input and len(k) > 1 for k in kws):
                    found_sub, found_cat = s, cname; break
            if found_sub: break
        if found_sub:
            detected_sub, detected_cat = found_sub, found_cat
            # 만약 키워드로 새로운 카테고리를 찾았다면 이전 소분류 잔재는 지움
            if detected_cat != prev_filters.get("category"):
                 detected_sub = found_sub

    if detected_cat: new_filters["category"] = detected_cat
    if detected_sub: new_filters["subcategory"] = detected_sub

    # 캔/파우치 특수 매핑
    FORM_TO_SUB = {
        ("고양이", "캔", "사료"): "주식캔", ("고양이", "캔", "간식"): "간식캔",
        ("고양이", "파우치", "사료"): "주식파우치", ("고양이", "파우치", "간식"): "간식파우치",
        ("강아지", "캔", "간식"): "캔/파우치",
    }
    prev_form_hint = None if is_major_switch else state.get("form_hint")
    llm_form_hint = r.get("form_hint")
    detected_form = next((kw for kw in ("캔", "파우치") if kw in user_input), None)
    
    # 이번 턴에 없으면 이전 턴 정보 유지
    final_form_hint = detected_form or llm_form_hint or prev_form_hint
    
    form_hint = None
    if final_form_hint and current_pet_kr:
        cat = new_filters.get("category")
        sub = FORM_TO_SUB.get((current_pet_kr, final_form_hint, cat))
        if sub: 
            new_filters["subcategory"] = sub
            form_hint = None # 매핑 완료 시 힌트 제거
        else:
            form_hint = final_form_hint
    else:
        form_hint = None

    # 인기 추천 시 소분류 제거
    if "popularity" in new_intents and "subcategory" in new_filters:
        del new_filters["subcategory"]

    print(f"[INTENT] input='{user_input}' -> intents={new_intents}, pet={new_pet.get('name', new_pet.get('breed', 'Unknown'))}, filters={new_filters}")

    return {
        "intents":         new_intents,
        "target_pet_id":   target_pet_id,
        "pending_pet_ids": pending_pet_ids,
        "pending_categories": pending_categories,
        "is_pet_switched": is_pet_switched,
        "switched_pet_name": switched_pet_name,
        "domain_intent":   r.get("domain_intent") or state.get("domain_intent"),
        "detected_aspect": r.get("detected_aspect") or state.get("detected_aspect"),
        "budget":          int(r["budget"]) if r.get("budget") else state.get("budget"),
        "filters":         new_filters,
        "pet_profile":     new_pet,
        "is_pet_override": is_explicit_pet_info or is_pet_switched,
        "pet_mismatch":    False if is_pet_switched else state.get("pet_mismatch", False),
        "form_hint":       form_hint,
        "filter_relaxation_count": 0 if target_categories else state.get("filter_relaxation_count", 0),
        **overridden_metadata
    }
