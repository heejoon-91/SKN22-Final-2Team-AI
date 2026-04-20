from final_ai.contracts.filters import normalize_search_exclusions, normalize_search_filters


def _unique(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _species_label(value: str | None) -> str:
    normalized = str(value or "").lower()
    if normalized in {"dog", "강아지"}:
        return "강아지"
    if normalized in {"cat", "고양이"}:
        return "고양이"
    return str(value or "").strip()


def _format_age(value) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.endswith("살") or text.endswith("세") or "개월" in text:
        return text
    try:
        return f"{int(float(text))}살"
    except (TypeError, ValueError):
        return text


def _format_price(value) -> str:
    try:
        amount = int(value)
    except (TypeError, ValueError):
        return ""
    if amount >= 10000 and amount % 10000 == 0:
        return f"{amount // 10000}만원"
    return f"{amount:,}원"


def _profile_phrase(state: dict) -> str:
    pet_profile = state.get("pet_profile") or {}
    filters = normalize_search_filters(state.get("filters") or state.get("effective_filters"))

    species = _species_label(filters.get("pet_type") or pet_profile.get("species"))
    breed = str(pet_profile.get("breed") or "").strip()
    age = _format_age(pet_profile.get("age"))

    parts = [part for part in (age, breed, species) if part]
    return " ".join(parts) or "반려동물"


def _target_phrase(state: dict) -> str:
    filters = normalize_search_filters(state.get("filters") or state.get("effective_filters"))
    category = str(filters.get("category") or "").strip()
    subcategory = str(filters.get("subcategory") or "").strip()

    if category == "사료":
        return "사료"
    return subcategory or category or "상품"


def build_search_progress_messages(state: dict) -> list[str]:
    intents = state.get("intents") or []
    if "recommend" not in intents:
        return []

    exclusions = normalize_search_exclusions(state.get("exclusions"))
    excluded_ingredients = _unique(
        [
            *list(exclusions.get("ingredients") or []),
            *list(state.get("allergies") or []),
        ]
    )
    health_concerns = _unique(list(state.get("health_concerns") or []))
    requested_product_terms = _unique(list(state.get("requested_product_terms") or []))

    profile = _profile_phrase(state)
    target = _target_phrase(state)
    messages = []

    if excluded_ingredients:
        ingredient_label = ", ".join(excluded_ingredients)
        messages.append(f"{profile}의 {ingredient_label} 없는 {target}를 찾아보는 중...")
    else:
        messages.append(f"{profile}에게 맞는 {target}를 찾아보는 중...")

    budget_label = _format_price(state.get("budget"))
    min_budget_label = _format_price(state.get("min_budget"))
    if min_budget_label and budget_label:
        messages.append(f"{min_budget_label} 이상 {budget_label} 이하의 제품을 찾아보는 중...")
    elif budget_label:
        messages.append(f"{budget_label} 이하의 제품을 찾아보는 중...")
    elif min_budget_label:
        messages.append(f"{min_budget_label} 이상의 제품을 찾아보는 중...")

    if health_concerns:
        messages.append(f"{', '.join(health_concerns)} 고민을 고려하는 중...")

    if requested_product_terms:
        messages.append(f"{', '.join(requested_product_terms)} 상품을 우선 확인하는 중...")

    return messages[:4]
