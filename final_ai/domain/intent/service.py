import json
import re

from final_ai.api.dependencies.request_context import ensure_request_active
from final_ai.application.chat.memory import format_conversation_history
from final_ai.contracts.filters import (
    SearchExclusions,
    SearchFilters,
    build_search_exclusions,
    build_search_filters,
    normalize_filter_list,
    normalize_filter_value,
    normalize_search_exclusions,
    normalize_search_filters,
)
from final_ai.domain.intent.prompts import CATEGORIES, build_intent_prompt
from final_ai.domain.profile.service import get_pet_full_profile, get_user_pets
from final_ai.domain.recommendation.constants import ALLERGY_TERM_ALIASES
from final_ai.infrastructure.llm.openai_client import LLM_MODEL, llm
from final_ai.infrastructure.observability import get_logger
from final_ai.graph.state import ChatState

logger = get_logger(__name__)

# 건강 고민 표준 태그 매핑 사전
HEALTH_CONCERN_MAP = {
    "체중": ["다이어트", "살", "비만", "체중조절", "저칼로리", "슬림"],
    "눈물": ["눈물자국", "눈건강", "눈세정", "아이케어"],
    "피부": ["아토피", "가려움", "알러지", "피부염", "피부건강", "피부/모질"],
    "관절": ["슬개골", "뼈", "관절건강", "다리", "튼튼"],
    "소화": ["장", "변비", "설사", "소화불량", "위건강", "소화/장"],
    "치아": ["양치", "치석", "구강", "입냄새", "덴탈"],
    "요로": ["신장", "방광", "결석", "신장건강"],
    "헤어볼": ["그루밍", "헤어볼제거"],
    "면역": ["체력", "활력", "항산화", "면역력"],
}

_RESULT_REFINEMENT_TOKENS = (
    "이중에서",
    "이중",
    "그중에서",
    "그중",
    "추천한것중",
    "추천해준것중",
    "방금추천한것중",
    "위에나온것중",
)

_EXCLUSION_TOKENS = (
    "제외",
    "빼고",
    "빼줘",
    "말고",
    "삭제",
    "제거",
    "없는",
)

_ALTERNATIVE_RECOMMENDATION_TOKENS = (
    "다른거",
    "다른걸로",
    "다른상품",
    "딴거",
    "말고다른",
    "빼고다른",
)

_RECOMMENDATION_SIGNAL_TOKENS = (
    "추천",
    "추천해줘",
    "추천해주세요",
    "보여줘",
    "보여주세요",
    "찾아줘",
    "찾아주세요",
    "골라줘",
    "골라주세요",
)

_BUDGET_UNDER_PATTERNS = (
    r"(\d+(?:\.\d+)?)\s*만\s*원?\s*(?:이하|미만|까지|안쪽|선)",
    r"(\d+(?:\.\d+)?)\s*원\s*(?:이하|미만|까지|안쪽|선)",
)

_BUDGET_OVER_PATTERNS = (
    r"(\d+(?:\.\d+)?)\s*만\s*원?\s*(?:이상|초과|넘는|부터)",
    r"(\d+(?:\.\d+)?)\s*원\s*(?:이상|초과|넘는|부터)",
)

_EXCLUSION_FALLBACK_STOPWORDS = {
    "강아지",
    "고양이",
    "사료",
    "간식",
    "상품",
    "추천",
    "추천해줘",
    "추천해주세요",
    "보여줘",
    "알려줘",
    "다른",
    "거",
}


def _normalize_compact_text(text: object) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _merge_unique_terms(*values: object) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in normalize_filter_list(value):
            normalized = _normalize_compact_text(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(item)
    return merged


def _has_exclusion_signal(text: str | None) -> bool:
    normalized = _normalize_compact_text(text)
    if not normalized:
        return False
    return any(token in normalized for token in _EXCLUSION_TOKENS)


def _has_recommendation_signal(text: str | None) -> bool:
    normalized = _normalize_compact_text(text)
    if not normalized:
        return False
    return any(token in normalized for token in _RECOMMENDATION_SIGNAL_TOKENS)


def _is_alternative_recommendation_request(
    text: str | None,
    *,
    previous_goods_ids: list[str] | None,
) -> bool:
    if not previous_goods_ids:
        return False
    normalized = _normalize_compact_text(text)
    if not normalized or "다른카테고리" in normalized:
        return False
    return any(token in normalized for token in _ALTERNATIVE_RECOMMENDATION_TOKENS)


def _extract_exclusion_keywords_from_text(text: str | None) -> list[str]:
    raw = str(text or "")
    if not raw or not _has_exclusion_signal(raw):
        return []

    found: list[str] = []
    seen: set[str] = set()

    def add_candidate(candidate: str):
        normalized = normalize_filter_value(candidate)
        if not normalized or normalized in _EXCLUSION_FALLBACK_STOPWORDS or len(normalized) < 2:
            return
        compact = _normalize_compact_text(normalized)
        if compact in seen:
            return
        seen.add(compact)
        found.append(normalized)

    for match in re.finditer(r"[\"']([^\"']{2,30})[\"']\s*(?:제외|빼고|말고|삭제|제거|없는)", raw):
        add_candidate(match.group(1))

    if found:
        return found

    for match in re.finditer(r"([0-9A-Za-z가-힣/]+)\s*(?:은|는|이|가|을|를)?\s*(?:제외|빼고|말고|삭제|제거|없는)", raw):
        add_candidate(match.group(1))

    return found


def _extract_exclusion_ingredients_from_text(text: str | None) -> list[str]:
    candidates = _extract_exclusion_keywords_from_text(text)
    if not candidates:
        return []

    known_ingredients = {
        _normalize_compact_text(alias)
        for aliases in ALLERGY_TERM_ALIASES.values()
        for alias in aliases
    }
    known_ingredients.update(_normalize_compact_text(key) for key in ALLERGY_TERM_ALIASES)

    ingredients = []
    seen = set()
    for candidate in candidates:
        normalized = _normalize_compact_text(candidate)
        if normalized not in known_ingredients or normalized in seen:
            continue
        seen.add(normalized)
        ingredients.append(candidate)
    return ingredients


def _merge_search_exclusions(*values: SearchExclusions | None) -> SearchExclusions:
    return build_search_exclusions(
        brands=_merge_unique_terms(*(value.get("brands") if value else [] for value in values)),
        categories=_merge_unique_terms(*(value.get("categories") if value else [] for value in values)),
        subcategories=_merge_unique_terms(*(value.get("subcategories") if value else [] for value in values)),
        health_concerns=_merge_unique_terms(*(value.get("health_concerns") if value else [] for value in values)),
        ingredients=_merge_unique_terms(*(value.get("ingredients") if value else [] for value in values)),
        keywords=_merge_unique_terms(*(value.get("keywords") if value else [] for value in values)),
        goods_ids=_merge_unique_terms(*(value.get("goods_ids") if value else [] for value in values)),
    )


def _prune_conflicting_exclusions(filters: SearchFilters, exclusions: SearchExclusions) -> SearchExclusions:
    pruned = normalize_search_exclusions(exclusions)
    include_brand = normalize_filter_value(filters.get("brand"))
    include_category = normalize_filter_value(filters.get("category"))
    include_subcategory = normalize_filter_value(filters.get("subcategory"))

    def remove_match(values: list[str] | None, target: str | None) -> list[str]:
        if not values or not target:
            return list(values or [])
        target_normalized = _normalize_compact_text(target)
        return [value for value in values if _normalize_compact_text(value) != target_normalized]

    if "brands" in pruned:
        next_values = remove_match(pruned.get("brands"), include_brand)
        if next_values:
            pruned["brands"] = next_values
        else:
            pruned.pop("brands", None)
    if "categories" in pruned:
        next_values = remove_match(pruned.get("categories"), include_category)
        if next_values:
            pruned["categories"] = next_values
        else:
            pruned.pop("categories", None)
    if "subcategories" in pruned:
        next_values = remove_match(pruned.get("subcategories"), include_subcategory)
        if next_values:
            pruned["subcategories"] = next_values
        else:
            pruned.pop("subcategories", None)

    return pruned


def _infer_pet_type_from_text(text: str | None) -> str | None:
    raw = str(text or "")
    lowered = raw.lower()

    has_dog = "강아지" in raw or bool(re.search(r"\b(?:dog|puppy)\b", lowered))
    has_cat = "고양이" in raw or bool(re.search(r"\b(?:cat|kitten)\b", lowered))

    if has_dog == has_cat:
        return None
    return "강아지" if has_dog else "고양이"


def _is_result_refinement_request(
    text: str | None,
    *,
    previous_goods_ids: list[str] | None,
) -> bool:
    if not previous_goods_ids:
        return False
    normalized = re.sub(r"\s+", "", str(text or "")).lower()
    if not normalized:
        return False
    return any(token in normalized for token in _RESULT_REFINEMENT_TOKENS)


def _parse_budget_amount(text: str, *, is_manwon: bool) -> int | None:
    try:
        amount = float(text)
    except (TypeError, ValueError):
        return None
    return int(amount * (10000 if is_manwon else 1))


def _extract_budget_constraints(text: str | None) -> tuple[int | None, int | None]:
    raw = str(text or "")
    min_budget = None
    max_budget = None

    for pattern in _BUDGET_UNDER_PATTERNS:
        match = re.search(pattern, raw)
        if match:
            max_budget = _parse_budget_amount(match.group(1), is_manwon="만" in pattern)
            break

    for pattern in _BUDGET_OVER_PATTERNS:
        match = re.search(pattern, raw)
        if match:
            min_budget = _parse_budget_amount(match.group(1), is_manwon="만" in pattern)
            break

    return min_budget, max_budget


def _has_budget_constraint_signal(text: str | None) -> bool:
    min_budget, max_budget = _extract_budget_constraints(text)
    return min_budget is not None or max_budget is not None



def _resolve_category_subcategory(
    *,
    text_to_search: str,
    category: str | None,
    subcategory: str | None,
    pet_type_kr: str,
) -> tuple[str | None, str | None]:
    """
    category.json의 별칭(aliases) 데이터를 기반으로 카테고리와 서브카테고리를 보정합니다.
    모든 매칭 후보 중 가장 길게 일치하는 항목을 선택하는 Longest Match 전략을 사용합니다.
    """
    pet_category_map = CATEGORIES.get(pet_type_kr or "강아지", {})
    # LLM이 제안한 값과 원문 텍스트를 모두 합쳐서 검색 대상으로 함
    combined_text = f"{category or ''} {subcategory or ''} {text_to_search}".strip()
    
    best_match_len = 0
    found_sub, found_cat = None, None
    
    # 1. 전수 조사를 통한 최적의 서브카테고리 매칭
    for cat_name, cat_info in pet_category_map.items():
        sub_dict = cat_info.get("subcategories", {})
        for canonical, aliases in sub_dict.items():
            targets = [canonical] + (aliases or [])
            for target in targets:
                # 2글자 이상 일치하는지 확인
                if target in combined_text and len(target) > 1:
                    # 더 긴 단어가 매칭되면 업데이트 (예: '사료'보다는 '습식사료'가 더 정확함)
                    if len(target) > best_match_len:
                        best_match_len = len(target)
                        found_sub = canonical
                        found_cat = cat_name
                    # 길이가 같을 경우, LLM이 원래 제안했던 subcategory와 일치하는 것이 있다면 그것을 유지
                    elif len(target) == best_match_len and canonical == subcategory:
                        found_sub = canonical
                        found_cat = cat_name

    # 2. 카테고리 직접 매칭 (서브카테고리보다 후순위)
    if not found_cat:
        for cat_name, cat_info in pet_category_map.items():
            targets = [cat_name] + (cat_info.get("aliases") or [])
            for target in targets:
                if target in combined_text and len(target) > best_match_len:
                    best_match_len = len(target)
                    found_cat = cat_name

    # 매칭된 결과가 있다면 업데이트, 없으면 LLM 제안값 유지
    final_sub = found_sub if found_sub else subcategory
    final_cat = found_cat if found_cat else category
    
    # 2. 서브카테고리 기반 부모 카테고리 확정 (서브카테고리는 결정됐으나 카테고리가 없거나 오매칭된 경우 보정)
    if final_sub:
        for cat_name, cat_info in pet_category_map.items():
            if final_sub in cat_info.get("subcategories", {}):
                final_cat = cat_name
                break
                
    return final_cat, final_sub


def _resolve_category_subcategory_any_pet(
    *,
    text_to_search: str,
    category: str | None,
    subcategory: str | None,
    pet_type_kr: str | None,
) -> tuple[str | None, str | None]:
    candidate_pet_types = [pet_type_kr] if pet_type_kr in CATEGORIES else list(CATEGORIES.keys())

    best_cat = category
    best_sub = subcategory
    best_score = max(len(str(subcategory or "")), len(str(category or "")))

    for candidate_pet_type in candidate_pet_types:
        resolved_cat, resolved_sub = _resolve_category_subcategory(
            text_to_search=text_to_search,
            category=category,
            subcategory=subcategory,
            pet_type_kr=candidate_pet_type,
        )
        score = max(len(str(resolved_sub or "")), len(str(resolved_cat or "")))
        if score > best_score:
            best_score = score
            best_cat = resolved_cat
            best_sub = resolved_sub

    return best_cat, best_sub


def _build_context(
    *,
    state: ChatState,
    user_input: str,
    user_pets: list[dict],
    prev_intents: list[str],
    prev_filters: SearchFilters,
    prev_pet: dict,
    target_pet_id: str | None,
) -> str:
    context_parts = []
    memory_summary = (state.get("memory_summary") or "").strip()
    if memory_summary:
        context_parts.append(f"누적 대화 요약:\n{memory_summary}")

    summary_candidates_text = format_conversation_history(state.get("summary_candidates"), limit=8)
    if summary_candidates_text != "없음":
        context_parts.append(f"이번 턴에 메모리로 편입할 이전 대화:\n{summary_candidates_text}")

    history_text = format_conversation_history(state.get("conversation_history"), limit=10)
    if history_text != "없음":
        context_parts.append(f"최근 대화 기록:\n{history_text}")

    if not ((state.get("clarification_count", 0) > 0 or prev_intents) and user_input):
        return "\n\n".join(context_parts)

    prev_data = {
        "intents": prev_intents,
        "filters": prev_filters,
        "pet_profile": prev_pet,
        "current_pet_id": target_pet_id,
        "has_previous_recommendations": bool(state.get("last_recommended_goods_ids")),
        "previous_recommendation_count": len(state.get("last_recommended_goods_ids") or []),
        "user_registered_pets": [pet["name"] for pet in user_pets],
    }
    context_parts.append(f"이전 대화 상태 및 등록된 펫 정보: {json.dumps(prev_data, ensure_ascii=False)}")
    return "\n\n".join(context_parts)


def _classify_user_input(user_input: str, context: str) -> dict:
    ensure_request_active()
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": build_intent_prompt(context)},
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def classify_intent(state: ChatState) -> dict:
    original_user_input = state["user_input"]
    current_user_input = original_user_input
    user_id = state.get("user_id")
    prev_intents = state.get("intents") or []
    prev_filters = normalize_search_filters(state.get("filters"))
    prev_exclusions = normalize_search_exclusions(state.get("exclusions"))
    prev_pet = state.get("pet_profile") or {}
    target_pet_id = state.get("target_pet_id")
    prev_last_recommended_goods_ids = list(state.get("last_recommended_goods_ids") or [])
    prev_last_search_goods_ids = list(state.get("last_search_goods_ids") or [])
    pending_requests = list(state.get("pending_requests") or [])
    decomposed_tasks = list(state.get("decomposed_tasks") or [])

    user_pets = get_user_pets(user_id) if user_id else []
    
    # 1. 초기 분류
    context = _build_context(
        state=state,
        user_input=current_user_input,
        user_pets=user_pets,
        prev_intents=prev_intents,
        prev_filters=prev_filters,
        prev_pet=prev_pet,
        target_pet_id=target_pet_id,
    )
    result = _classify_user_input(current_user_input, context)
    
    is_next_request = result.get("is_next_request", False)

    # 2. 후속 요청 처리 (Sequential Processing)
    if is_next_request and decomposed_tasks:
        next_task = decomposed_tasks.pop(0)
        pet_name = next_task.get("pet_name") or ""
        category = next_task.get("category") or ""
        subcategory = next_task.get("subcategory") or ""
        health = next_task.get("health_concern") or ""
        age = next_task.get("age") or ""
        
        # [개선] LLM 재호출 대신 큐의 데이터를 직접 결과에 매핑하여 정확도 보장
        logger.info("Processing queued task: %s", next_task)
        result = {
            "intents": ["recommend"],
            "mentioned_pet_names": [pet_name] if pet_name else [],
            "target_categories": [category] if category else [],
            "subcategory": subcategory,
            "health_concerns": [health] if health else [],
            "age": age,
            "is_next_request": False,
            "is_synthetic": True # 합성된 데이터임을 표시
        }
        current_user_input = f"{pet_name} {health} {subcategory} {category} 추천".strip()

    new_intents = result.get("intents") or []
    mentioned_names = result.get("mentioned_pet_names") or []
    
    # [추가] AI 질문(강아지/고양이?) 내의 단어가 이름으로 오인되는 것 방지
    mentioned_names = [name for name in mentioned_names if name.strip() not in ["강아지", "고양이", "dog", "cat"]]
    
    exclude_brands = normalize_filter_list(result.get("exclude_brands"))
    exclude_categories = normalize_filter_list(result.get("exclude_categories"))
    exclude_subcategories = normalize_filter_list(result.get("exclude_subcategories"))
    exclude_health_concerns_raw = normalize_filter_list(result.get("exclude_health_concerns"))
    exclude_ingredients = normalize_filter_list(result.get("exclude_ingredients"))
    exclude_keywords = normalize_filter_list(result.get("exclude_keywords"))
    raw_health_concerns = result.get("health_concerns") or []
    target_categories = result.get("target_categories") or []
    detected_brand = normalize_filter_value(result.get("brand"))
    explicit_pet_type = result.get("pet_type") or _infer_pet_type_from_text(original_user_input)
    is_explicit_pet_info = bool(explicit_pet_type or result.get("breed"))
    new_decomposed_tasks = result.get("decomposed_tasks") or []
    refinement_sort = normalize_filter_value(result.get("refinement_sort"))
    is_alternative_request = _is_alternative_recommendation_request(
        original_user_input,
        previous_goods_ids=prev_last_recommended_goods_ids,
    )
    llm_result_refinement = result.get("is_result_refinement")
    if isinstance(llm_result_refinement, bool):
        is_result_refinement = llm_result_refinement and not is_alternative_request
    else:
        is_result_refinement = _is_result_refinement_request(
            original_user_input,
            previous_goods_ids=prev_last_recommended_goods_ids,
        ) and not is_alternative_request

    if detected_brand and _has_exclusion_signal(original_user_input):
        exclude_brands = _merge_unique_terms(exclude_brands, [detected_brand])
        detected_brand = None
    exclude_ingredients = _merge_unique_terms(
        exclude_ingredients,
        _extract_exclusion_ingredients_from_text(original_user_input),
    )
    if not (
        exclude_brands
        or exclude_categories
        or exclude_subcategories
        or exclude_health_concerns_raw
        or exclude_ingredients
        or exclude_keywords
    ):
        exclude_keywords = _extract_exclusion_keywords_from_text(original_user_input)

    parsed_min_budget, parsed_max_budget = _extract_budget_constraints(original_user_input)
    if (
        not is_result_refinement
        and prev_last_search_goods_ids
        and _has_budget_constraint_signal(original_user_input)
        and _has_recommendation_signal(original_user_input)
    ):
        is_result_refinement = True

    if new_decomposed_tasks:
        logger.info("─── Query Decomposition Detected ───")
        for i, task in enumerate(new_decomposed_tasks):
            logger.info(
                "Task [%d]: pet_name=%s, category=%s, subcategory=%s, health_concern=%s",
                i + 1,
                task.get("pet_name", "N/A"),
                task.get("category", "N/A"),
                task.get("subcategory", "N/A"),
                task.get("health_concern", "N/A"),
            )
        logger.info("───────────────────────────────────")

    # ── [중요] Decomposition 맥락 방어 로직 ──
    # 질문 원문(original_user_input)에 펫 이름이 2개 이상 직접 언급되거나, 
    # 혹은 카테고리가 2개 이상 직접 언급되지 않았다면 과거 이력에 의한 과잉 분해로 간주하고 차단합니다.
    # 단, 합성된 입력(is_synthetic)에 대해서는 이 로직을 건너뜁니다.
    if not result.get("is_synthetic"):
        actual_mentions_in_input = [pet["name"] for pet in user_pets if pet["name"] in original_user_input]
        
        # 카테고리 키워드 추출 (사료, 간식, 모래 등)
        all_cat_keywords = []
        for p_type in CATEGORIES.values():
            all_cat_keywords.extend(p_type.keys())
        actual_categories_in_input = [cat for cat in set(all_cat_keywords) if cat in original_user_input]
        
        # 펫 이름도 1개고 카테고리 언급도 1개 이하라면 (즉, 질문에 복합 요소가 없다면) 분해 취소
        if len(new_decomposed_tasks) > 1 and len(actual_mentions_in_input) < 2 and len(actual_categories_in_input) < 2:
            logger.info("Preventing excessive decomposition triggered by history context. (mentions: pets=%d, cats=%d)", 
                        len(actual_mentions_in_input), len(actual_categories_in_input))
            new_decomposed_tasks = [] # 쪼개지 않고 현재 입력 전체를 하나로 처리

    def map_health_concerns(concerns):
        detected = []
        for raw in concerns:
            mapped_tag = raw
            for tag, keywords in HEALTH_CONCERN_MAP.items():
                if any(keyword in raw for keyword in keywords) or raw == tag:
                    mapped_tag = tag
                    break
            detected.append(mapped_tag)
        return detected

    detected_health_concerns = map_health_concerns(raw_health_concerns)
    detected_excluded_health_concerns = map_health_concerns(exclude_health_concerns_raw)

    has_structured_followup_signal = bool(
        mentioned_names
        or new_decomposed_tasks
        or explicit_pet_type
        or result.get("breed")
        or result.get("age")
        or target_categories
        or detected_health_concerns
        or exclude_brands
        or exclude_categories
        or exclude_subcategories
        or detected_excluded_health_concerns
        or exclude_ingredients
        or exclude_keywords
        or result.get("subcategory")
        or result.get("budget")
        or result.get("min_budget")
        or parsed_min_budget is not None
        or parsed_max_budget is not None
        or is_result_refinement
        or is_alternative_request
    )

    if new_intents == ["unclear"] and "recommend" in prev_intents and has_structured_followup_signal:
        new_intents = ["recommend"]

    if not new_intents and "recommend" in prev_intents and is_result_refinement:
        new_intents = ["recommend"]

    if "recommend" in new_intents and "domain_qa" not in new_intents:
        if any(keyword in original_user_input for keyword in ("왜", "설명", "이유")):
            new_intents = [*new_intents, "domain_qa"]

    is_pet_switched = False
    switched_pet_name = None
    overridden_metadata = {}

    category_lookup_pet_type = explicit_pet_type
    if not category_lookup_pet_type:
        if prev_pet.get("species") == "dog":
            category_lookup_pet_type = "강아지"
        elif prev_pet.get("species") == "cat":
            category_lookup_pet_type = "고양이"
        else:
            category_lookup_pet_type = prev_filters.get("pet_type")
    
    # 펫 타입 결정
    temp_pet = dict(prev_pet)
    if is_explicit_pet_info and not target_pet_id:
        if explicit_pet_type:
            temp_pet["species"] = "dog" if explicit_pet_type == "강아지" else "cat"
    pet_type_kr = "고양이" if (temp_pet.get("species") == "cat" or explicit_pet_type == "고양이") else "강아지"

    fallback_category, fallback_subcategory = _resolve_category_subcategory_any_pet(
        text_to_search=original_user_input,
        category=target_categories[0] if target_categories else None,
        subcategory=normalize_filter_value(result.get("subcategory")),
        pet_type_kr=category_lookup_pet_type,
    )
    if not target_categories and fallback_category:
        target_categories = [fallback_category]
    if not result.get("subcategory") and fallback_subcategory:
        result["subcategory"] = fallback_subcategory

    if target_categories and _has_recommendation_signal(original_user_input) and "recommend" not in new_intents:
        new_intents = [intent for intent in new_intents if intent != "unclear"]
        new_intents.append("recommend")

    resolved_exclude_categories = []
    for raw_category in exclude_categories:
        resolved_category, _ = _resolve_category_subcategory(
            text_to_search=raw_category,
            category=raw_category,
            subcategory=None,
            pet_type_kr=pet_type_kr,
        )
        resolved_exclude_categories.append(resolved_category or raw_category)

    resolved_exclude_subcategories = []
    for raw_subcategory in exclude_subcategories:
        _, resolved_subcategory = _resolve_category_subcategory(
            text_to_search=raw_subcategory,
            category=None,
            subcategory=raw_subcategory,
            pet_type_kr=pet_type_kr,
        )
        resolved_exclude_subcategories.append(resolved_subcategory or raw_subcategory)

    # ── [중요] decomposed_tasks 개별 보정 로직 ──
    if new_decomposed_tasks and "recommend" in new_intents:
        for task in new_decomposed_tasks:
            task_text = f"{task.get('category', '')} {task.get('subcategory', '')}".strip()
            t_cat, t_sub = _resolve_category_subcategory(
                text_to_search=task_text, 
                category=task.get("category"),
                subcategory=task.get("subcategory"),
                pet_type_kr=pet_type_kr
            )
            task["category"] = t_cat
            task["subcategory"] = t_sub

    # 4. Query Decomposition 처리
    if new_decomposed_tasks and not is_next_request:
        # [수정] 여러 작업이 감지되면, 현재 턴에서는 첫 번째 작업만 수행하고
        # 나머지는 모두 decomposed_tasks 큐에 쌓습니다.
        first_task = new_decomposed_tasks[0]
        decomposed_tasks.extend(new_decomposed_tasks[1:]) 
        
        logger.info("Multi-task detected. Processing first task and queuing %d tasks.", len(new_decomposed_tasks) - 1)

        # [중요] 복합 질문 시 LLM이 문장 전체에서 추출한 Root 레벨의 필터 정보(잔상)를 초기화합니다.
        # 이렇게 해야 마지막에 언급된 '모래' 등의 정보가 첫 번째 작업인 '사료'에 섞이지 않습니다.
        mentioned_names = [first_task["pet_name"]] if first_task.get("pet_name") else []
        target_categories = [first_task["category"]] if first_task.get("category") else []
        result["subcategory"] = first_task.get("subcategory") # null이면 null로 명시적 덮어쓰기
        
        # [핵심 추가] 하단 Step 6 보정 로직이 문장 전체를 보지 못하도록 현재 태스크의 텍스트로 국소화합니다.
        current_user_input = f"{first_task.get('pet_name','') or ''} {first_task.get('category','') or ''} {first_task.get('subcategory','') or ''}".strip()

        if first_task.get("health_concern"):
            detected_health_concerns = map_health_concerns([first_task["health_concern"]])
        else:
            detected_health_concerns = []

        if first_task.get("age"):
            result["age"] = first_task["age"]
        
        new_intents = ["recommend"]

    if "recommend" in new_intents and "domain_qa" not in new_intents:
        if any(keyword in original_user_input for keyword in ("왜", "설명", "이유")):
            new_intents = [*new_intents, "domain_qa"]

    # 5. 펫 매칭 및 전환 로직
    if mentioned_names:
        raw_matched = [pet for pet in user_pets if pet["name"] in mentioned_names]
        matched_pets = sorted(
            raw_matched,
            key=lambda p: mentioned_names.index(p["name"]) if p["name"] in mentioned_names else 999,
        )
        if matched_pets:
            first_pet = matched_pets[0]
            if str(first_pet["pet_id"]) != str(target_pet_id):
                full_profile = get_pet_full_profile(str(first_pet["pet_id"]))
                if full_profile:
                    target_pet_id = str(first_pet["pet_id"])
                    prev_pet = full_profile["pet_profile"]
                    overridden_metadata = {
                        "health_concerns": full_profile["health_concerns"],
                        "allergies": full_profile["allergies"],
                        "food_preferences": full_profile["food_preferences"],
                        "breed_context": "",
                        "health_traits": "",
                    }
                    is_pet_switched = True
                    switched_pet_name = prev_pet.get("name")
                    logger.info("Pet switched to '%s' by name mention.", switched_pet_name)

            if not new_decomposed_tasks:
                remaining_categories = list(target_categories[1:]) if len(target_categories) > 1 else []
                for i, pet in enumerate(matched_pets[1:]):
                    pet_id = str(pet["pet_id"])
                    pending_cat = remaining_categories[i] if i < len(remaining_categories) else None
                    pending_entry = {"pet_id": pet_id, "category": pending_cat}
                    if not any(r.get("pet_id") == pet_id for r in pending_requests):
                        pending_requests.append(pending_entry)
                multi_pet_categories_registered = len(matched_pets) > 1 and bool(remaining_categories)
            else:
                multi_pet_categories_registered = True
    elif is_explicit_pet_info:
        # 이름 언급은 없지만 종/품종 정보가 명시된 경우
        multi_pet_categories_registered = False
        new_breed = result.get("breed")
        current_breed = prev_pet.get("breed")
        
        # [수정] 펫 미선택 시에는 품종이 들어와도 '전환' 플래그를 남발하지 않도록 조건 보강
        if target_pet_id and new_breed and current_breed and new_breed != current_breed:
            target_pet_id = None
            prev_pet = {}
            overridden_metadata = {
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "breed_context": "",
                "health_traits": "",
            }
            is_pet_switched = True
            logger.info("Breed switch detected: %s", new_breed)

    if not new_intents:
        if "recommend" in prev_intents:
            new_intents = ["recommend"]
        else:
            new_intents = ["unclear"]
    else:
        multi_pet_categories_registered = False

    new_pet = dict(prev_pet)
    # [수정] 펫 미선택 상태라면 기존 프로필(고양이 등)을 초기화하고 사용자의 새로운 입력(강아지 등)을 우선 적용
    if not target_pet_id:
        if is_explicit_pet_info or result.get("age"):
            new_pet = {} # 펫 선택 안함 상태에서 새로운 정보가 들어오면 기존 프로필 오염 제거
            if explicit_pet_type:
                new_pet["species"] = "dog" if explicit_pet_type == "강아지" else "cat"
            if result.get("breed"):
                new_pet["breed"] = result["breed"]
            if result.get("age"):
                new_pet["age"] = result["age"]
    elif result.get("age"):
        new_pet["age"] = result["age"]

    new_filters: SearchFilters = {}
    is_major_switch = bool(target_categories)

    # [중요] 펫 타입 필터 결정 우선순위 보정
    # 1. 사용자의 현재 입력에서 감지된 펫 타입이 있으면 최우선 (이게 "고양이"로 튀는 것을 막아줍니다)
    if explicit_pet_type:
        new_filters["pet_type"] = explicit_pet_type
    # 2. 입력에 없으면 프로필 정보를 따름
    elif new_pet.get("species"):
        new_filters["pet_type"] = "강아지" if new_pet["species"] == "dog" else "고양이"
    # 3. 그것도 없으면 이전 필터 유지
    elif prev_filters.get("pet_type"):
        new_filters["pet_type"] = prev_filters["pet_type"]

    # 6. 메인 분석 결과 보정
    detected_cat = target_categories[0] if target_categories else None
    detected_sub = normalize_filter_value(result.get("subcategory"))

    if "recommend" in new_intents:
        detected_cat, detected_sub = _resolve_category_subcategory_any_pet(
            text_to_search=current_user_input,
            category=detected_cat,
            subcategory=detected_sub,
            pet_type_kr=category_lookup_pet_type or pet_type_kr,
        )

    # 필터 적용
    if detected_cat:
        new_filters["category"] = detected_cat
        if target_categories and len(target_categories) > 1 and not multi_pet_categories_registered:
            for category in target_categories[1:]:
                pending_requests.append({"pet_id": None, "category": category})
    elif "recommend" in new_intents and not is_major_switch:
        new_filters["category"] = prev_filters.get("category")

    if detected_sub:
        new_filters["subcategory"] = detected_sub
    elif "recommend" in new_intents and not is_major_switch:
        if new_filters.get("category") == prev_filters.get("category"):
            new_filters["subcategory"] = prev_filters.get("subcategory")

    if detected_brand:
        new_filters["brand"] = detected_brand

    if "popularity" in new_intents and "subcategory" in new_filters:
        del new_filters["subcategory"]

    logger.info(
        "intent classified original_input=%s normalized_input=%s intents=%s pet=%s filters=%s exclusions=%s decomposed_count=%s refinement=%s allowed_ids=%s",
        original_user_input,
        current_user_input,
        new_intents,
        new_pet.get("name", new_pet.get("breed", "Unknown")),
        build_search_filters(
            pet_type=new_filters.get("pet_type"),
            category=new_filters.get("category"),
            subcategory=new_filters.get("subcategory"),
            brand=new_filters.get("brand"),
        ),
        build_search_exclusions(
            brands=exclude_brands,
            categories=resolved_exclude_categories,
            subcategories=resolved_exclude_subcategories,
            health_concerns=detected_excluded_health_concerns,
            ingredients=exclude_ingredients,
            keywords=exclude_keywords,
            goods_ids=prev_last_recommended_goods_ids if is_alternative_request else [],
        ),
        len(decomposed_tasks),
        is_result_refinement,
        len(prev_last_recommended_goods_ids),
    )

    combined_allergies = _merge_unique_terms(overridden_metadata.get("allergies") or state.get("allergies") or [])

    current_health_concerns = overridden_metadata.get("health_concerns") or state.get("health_concerns") or []
    combined_health_concerns = _merge_unique_terms(current_health_concerns, detected_health_concerns)
    base_exclusions = prev_exclusions if (is_result_refinement or is_alternative_request) else {}
    detected_exclusions = build_search_exclusions(
        brands=exclude_brands,
        categories=resolved_exclude_categories,
        subcategories=resolved_exclude_subcategories,
        health_concerns=detected_excluded_health_concerns,
        ingredients=exclude_ingredients,
        keywords=exclude_keywords,
        goods_ids=prev_last_recommended_goods_ids if is_alternative_request else [],
    )
    combined_exclusions = _prune_conflicting_exclusions(
        new_filters,
        _merge_search_exclusions(base_exclusions, detected_exclusions),
    )

    return {
        "original_user_input": original_user_input,
        "normalized_user_input": current_user_input,
        "intents": new_intents,
        "target_pet_id": target_pet_id,
        "last_recommended_goods_ids": prev_last_recommended_goods_ids,
        "last_search_goods_ids": prev_last_search_goods_ids,
        "allowed_goods_ids": prev_last_search_goods_ids if is_result_refinement else [],
        "pending_requests": pending_requests,
        "decomposed_tasks": decomposed_tasks,
        "new_decomposed_tasks": new_decomposed_tasks,
        "is_pet_switched": is_pet_switched,
        "is_result_refinement": is_result_refinement,
        "refinement_sort": refinement_sort if is_result_refinement else None,
        "switched_pet_name": switched_pet_name,
        "domain_intent": result.get("domain_intent") or state.get("domain_intent"),
        "detected_aspect": result.get("detected_aspect") or state.get("detected_aspect"),
        "budget": (
            int(result["budget"])
            if result.get("budget")
            else parsed_max_budget
            if parsed_max_budget is not None
            else state.get("budget")
            if is_result_refinement
            else None
        ),
        "min_budget": (
            int(result["min_budget"])
            if result.get("min_budget")
            else parsed_min_budget
            if parsed_min_budget is not None
            else state.get("min_budget")
            if is_result_refinement
            else None
        ),
        "filters": build_search_filters(
            pet_type=new_filters.get("pet_type"),
            category=new_filters.get("category"),
            subcategory=new_filters.get("subcategory"),
            brand=new_filters.get("brand"),
        ),
        "exclusions": combined_exclusions,
        "pet_profile": new_pet,
        "is_pet_override": is_explicit_pet_info or is_pet_switched,
        "pet_mismatch": False if is_pet_switched else state.get("pet_mismatch", False),
        "filter_relaxation_count": 0 if target_categories else state.get("filter_relaxation_count", 0),
        **overridden_metadata,
        "allergies": combined_allergies,
        "health_concerns": combined_health_concerns,
    }
