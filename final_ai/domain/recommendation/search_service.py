import ast
import json
import unicodedata

from final_ai.contracts.filters import normalize_search_exclusions, normalize_search_filters
from final_ai.domain.profile.health_concerns import normalize_health_concerns
from final_ai.domain.recommendation.constants import (
    AGE_EXCLUDE_KEYWORDS,
    AGE_MANDATORY_KEYWORDS,
    ALLERGY_SAFE_WORDS,
    ALLERGY_STOP_NOUNS,
    ALLERGY_TERM_ALIASES,
    CORE_ANIMAL_PLANTS,
    FEED_CATEGORIES,
    MIXED_SUBS,
    PURE_CAN_SUBS,
    PURE_POUCH_SUBS,
    SAMPLE_BLACKLIST_WORDS,
)
from final_ai.domain.recommendation.filter_relaxation import (
    build_effective_search_filters,
    build_relaxed_filter_names,
    clamp_relaxation_count,
    should_include_health_concerns,
    should_include_profile_hints,
)
from final_ai.domain.recommendation.product_intent import candidate_matches_requested_terms
from final_ai.graph.state import ChatState
from final_ai.infrastructure.observability import get_logger
from final_ai.infrastructure.repositories.product_repository import list_products
from final_ai.infrastructure.search.hybrid_search import hybrid_search_pg, normalize_pet_species

logger = get_logger(__name__)


def _parse_collection(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return [raw]
        return [raw]
    return list(raw)

def _build_allergy_roots(allergies: list[str]) -> set[str]:
    if not allergies:
        return set()

    from kiwipiepy import Kiwi

    kiwi = Kiwi()
    roots = set()
    for text in allergies:
        text_lower = str(text).lower()
        for token in kiwi.tokenize(text_lower):
            if token.tag.startswith("NN") and token.form not in ALLERGY_STOP_NOUNS:
                roots.add(token.form)

        cleaned_text = text_lower
        for safe_word in ALLERGY_SAFE_WORDS:
            cleaned_text = cleaned_text.replace(safe_word, " ")
        for animal in CORE_ANIMAL_PLANTS:
            if animal in cleaned_text:
                roots.add(animal)
        roots.add(text_lower)

    expanded_roots = set(roots)
    for root in list(roots):
        normalized_root = _normalize_text(root)
        for aliases in ALLERGY_TERM_ALIASES.values():
            normalized_aliases = {_normalize_text(alias) for alias in aliases}
            if normalized_root in normalized_aliases:
                expanded_roots.update(aliases)

    logger.debug("allergy roots=%s expanded=%s", roots, expanded_roots)
    return expanded_roots


def _normalize_text(text) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFC", str(text)).lower().replace(" ", "")


def _effective_price(candidate: dict) -> float | None:
    for key in ("discount_price", "price"):
        value = candidate.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _matches_budget(candidate: dict, *, min_budget: int | None, budget: int | None) -> bool:
    effective_price = _effective_price(candidate)
    if effective_price is None:
        return False
    if min_budget is not None and effective_price < float(min_budget):
        return False
    if budget is not None and effective_price > float(budget):
        return False
    return True


def _merge_candidates(*candidate_groups: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    for candidates in candidate_groups:
        for candidate in candidates:
            goods_id = candidate.get("goods_id")
            dedupe_key = str(goods_id) if goods_id is not None else id(candidate)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            merged.append(dict(candidate))
    return merged


def _load_requested_product_candidates(
    *,
    requested_product_terms: list[str],
    pet_type: str | None,
    category: str | None,
    subcategory: str | None,
    brand: str | None,
    budget: int | None,
) -> list[dict]:
    if not requested_product_terms:
        return []

    strict_candidates = []
    for term in requested_product_terms:
        offset = 0
        page_size = 100
        while True:
            page = list_products(
                pet_type=pet_type,
                category=category,
                subcategory=subcategory,
                brand=brand,
                budget=budget,
                query=term,
                include_soldout=False,
                limit=page_size,
                offset=offset,
            )
            if not page:
                break
            strict_candidates.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
    return strict_candidates


def _prioritize_requested_product_candidates(
    candidates: list[dict],
    requested_product_terms: list[str],
    *,
    target_count: int,
) -> tuple[list[dict], int, int]:
    if not requested_product_terms:
        return candidates, 0, len(candidates)

    strict_candidates = []
    relaxed_candidates = []
    for candidate in candidates:
        candidate_with_flag = dict(candidate)
        is_match = candidate_matches_requested_terms(candidate_with_flag, requested_product_terms)
        candidate_with_flag["_requested_product_match"] = is_match
        candidate_with_flag["_requested_product_terms"] = requested_product_terms
        if is_match:
            strict_candidates.append(candidate_with_flag)
        else:
            relaxed_candidates.append(candidate_with_flag)

    if len(strict_candidates) >= target_count:
        return strict_candidates, len(strict_candidates), len(relaxed_candidates)
    return strict_candidates + relaxed_candidates, len(strict_candidates), len(relaxed_candidates)


def _to_normalized_list(raw) -> list[str]:
    if isinstance(raw, str):
        return [_normalize_text(value) for value in raw.replace("{", "").replace("}", "").split(",")]
    return [_normalize_text(str(value)) for value in raw]


def _is_excluded_candidate(candidate: dict, *, exclusions: dict[str, list[str]]) -> bool:
    if not exclusions:
        return False

    goods_id = str(candidate.get("goods_id") or "")
    if goods_id and goods_id in set(str(value) for value in exclusions.get("goods_ids") or []):
        return True

    goods_name = _normalize_text(candidate.get("goods_name"))
    brand_name = _normalize_text(candidate.get("brand_name"))
    ingredient_ocr = _normalize_text(candidate.get("ingredient_text_ocr"))
    sub_list = _to_normalized_list(candidate.get("subcategory") or [])
    cat_list = _to_normalized_list(candidate.get("category") or [])
    product_tags = _to_normalized_list(candidate.get("health_concern_tags") or [])

    main_ingredients = candidate.get("main_ingredients") or []
    if isinstance(main_ingredients, str):
        try:
            main_ingredients = json.loads(main_ingredients)
        except Exception:
            main_ingredients = [main_ingredients]
    normalized_main_ingredients = [_normalize_text(ingredient) for ingredient in main_ingredients]

    searchable_fields = [
        goods_name,
        brand_name,
        ingredient_ocr,
        *cat_list,
        *sub_list,
        *product_tags,
        *normalized_main_ingredients,
    ]

    for brand in exclusions.get("brands") or []:
        brand_term = _normalize_text(brand)
        if brand_term and (brand_term in brand_name or brand_term in goods_name):
            return True

    for category in exclusions.get("categories") or []:
        category_term = _normalize_text(category)
        if category_term and any(category_term in value for value in [*cat_list, *sub_list]):
            return True

    for subcategory in exclusions.get("subcategories") or []:
        subcategory_term = _normalize_text(subcategory)
        if subcategory_term and any(subcategory_term in value for value in sub_list):
            return True

    for concern in exclusions.get("health_concerns") or []:
        concern_term = _normalize_text(concern)
        if concern_term and any(concern_term in value for value in product_tags):
            return True

    expanded_ingredient_terms = []
    for ingredient in exclusions.get("ingredients") or []:
        ingredient_term = _normalize_text(ingredient)
        if not ingredient_term:
            continue
        expanded_ingredient_terms.append(ingredient_term)
        for aliases in ALLERGY_TERM_ALIASES.values():
            normalized_aliases = {_normalize_text(alias) for alias in aliases}
            if ingredient_term in normalized_aliases:
                expanded_ingredient_terms.extend(normalized_aliases)

    for ingredient_term in set(expanded_ingredient_terms):
        ingredient_fields = [ingredient_ocr, *normalized_main_ingredients]
        searchable_ingredient_fields = ingredient_fields if len(ingredient_term) <= 1 else [goods_name, *ingredient_fields]
        if ingredient_term and any(ingredient_term in value for value in searchable_ingredient_fields):
            return True

    for keyword in exclusions.get("keywords") or []:
        keyword_term = _normalize_text(keyword)
        if keyword_term and any(keyword_term in value for value in searchable_fields):
            return True

    return False


def _is_safe_candidate(
    candidate: dict,
    *,
    allergy_roots: set[str],
    target_age_group: str,
    mandatory_keywords: list[str],
    forbidden_age_keywords: list[str],
) -> bool:
    goods_name = _normalize_text(candidate.get("goods_name"))
    ingredient_ocr = _normalize_text(candidate.get("ingredient_text_ocr"))
    sub_list = _to_normalized_list(candidate.get("subcategory") or [])
    cat_list = _to_normalized_list(candidate.get("category") or [])

    is_feed = any(feed_category in value for feed_category in FEED_CATEGORIES for value in cat_list)
    
    # ── 연령별 필터링 (사료 카테고리에만 적용) ──────────────────────────────────
    if is_feed:
        # 1. 필수 키워드 검사 (키튼/퍼피/전연령 중 하나는 있어야 함)
        if mandatory_keywords:
            has_mandatory = any(_normalize_text(keyword) in value for keyword in mandatory_keywords for value in sub_list)
            has_mandatory = has_mandatory or any(_normalize_text(keyword) in goods_name for keyword in mandatory_keywords)
            if not has_mandatory:
                return False

        # 2. 금지 키워드 검사 (키튼인데 어덜트가 있으면 제외)
        for forbidden in forbidden_age_keywords:
            forbidden_normalized = _normalize_text(forbidden)
            if any(forbidden_normalized in value for value in sub_list) or forbidden_normalized in goods_name:
                logger.debug(
                    "age mismatch filtered product=%s forbidden=%s",
                    candidate["goods_name"],
                    forbidden_normalized,
                )
                return False

    # ── 알러지 필터링 (모든 카테고리에 적용) ────────────────────────────────────
    if allergy_roots:
        for allergy_root in allergy_roots:
            allergy_normalized = _normalize_text(allergy_root)
            if allergy_normalized in goods_name or allergy_normalized in ingredient_ocr:
                return False

        main_ingredients = candidate.get("main_ingredients") or []
        if isinstance(main_ingredients, str):
            try:
                main_ingredients = json.loads(main_ingredients)
            except Exception:
                main_ingredients = [main_ingredients]

        for ingredient in main_ingredients:
            ingredient_normalized = _normalize_text(ingredient)
            if any(_normalize_text(allergy_root) in ingredient_normalized for allergy_root in allergy_roots):
                return False

    return True


def execute_search_state(state: ChatState) -> dict:
    query = state.get("search_query") or state["user_input"]
    original_filters = normalize_search_filters(state.get("original_filters") or state.get("filters"))
    exclusions = normalize_search_exclusions(state.get("exclusions"))
    relaxation = clamp_relaxation_count(state.get("filter_relaxation_count", 0))
    filters = build_effective_search_filters(original_filters, relaxation=relaxation)
    pet_type = filters.get("pet_type")
    category = filters.get("category")
    subcategory = filters.get("subcategory")
    brand = filters.get("brand")
    min_budget = state.get("min_budget")
    budget = state.get("budget")
    health_concerns = normalize_health_concerns(state.get("health_concerns") or [])
    search_health_concerns = health_concerns if should_include_health_concerns(relaxation) else []
    requested_product_terms = list(state.get("requested_product_terms") or [])
    allowed_goods_ids = list(state.get("allowed_goods_ids") or [])
    if not allowed_goods_ids and state.get("is_result_refinement"):
        allowed_goods_ids = list(state.get("last_search_goods_ids") or [])

    pet_type_kr = normalize_pet_species(pet_type)
    if not pet_type_kr:
        pet_type_kr = normalize_pet_species((state.get("pet_profile") or {}).get("species"))

    candidates = hybrid_search_pg(
        query=query,
        top_k=50,
        pet_type=pet_type_kr,
        category=category,
        subcategory=subcategory,
        health_concerns=search_health_concerns,
        brand=brand,
        exclude_brands=exclusions.get("brands"),
        exclude_categories=exclusions.get("categories"),
        exclude_subcategories=exclusions.get("subcategories"),
        exclude_health_concerns=exclusions.get("health_concerns"),
        exclude_goods_ids=exclusions.get("goods_ids"),
        min_budget=min_budget,
        budget=budget,
        allowed_goods_ids=allowed_goods_ids,
    )
    requested_product_candidates = _load_requested_product_candidates(
        requested_product_terms=requested_product_terms,
        pet_type=pet_type_kr,
        category=category,
        subcategory=subcategory,
        brand=brand,
        budget=budget,
    )
    candidates = _merge_candidates(requested_product_candidates, candidates)
    if min_budget is not None or budget is not None:
        candidates = [
            candidate
            for candidate in candidates
            if _matches_budget(candidate, min_budget=min_budget, budget=budget)
        ]
    logger.info(
        "search hybrid returned=%s relaxation=%s relaxed=%s subcategory=%s category=%s pet=%s health=%s exclusions=%s allowed_ids=%s refinement=%s",
        len(candidates),
        relaxation,
        build_relaxed_filter_names(
            relaxation=relaxation,
            filters=original_filters,
            health_concerns=health_concerns,
            age_group=state.get("age_group"),
            breed=(state.get("pet_profile") or {}).get("breed"),
        ),
        subcategory,
        category,
        pet_type_kr,
        search_health_concerns,
        exclusions,
        len(allowed_goods_ids),
        bool(state.get("is_result_refinement")),
    )
    candidates = [
        candidate
        for candidate in candidates
        if not any(word in candidate.get("goods_name", "") for word in SAMPLE_BLACKLIST_WORDS)
    ]
    candidates = [
        candidate
        for candidate in candidates
        if not _is_excluded_candidate(candidate, exclusions=exclusions)
    ]
    logger.debug("blacklist filter count=%s", len(candidates))

    target_age_group = state.get("age_group", "어덜트")
    if not should_include_profile_hints(relaxation):
        target_age_group = None
    forbidden_age_keywords = AGE_EXCLUDE_KEYWORDS.get(target_age_group, []) if target_age_group else []
    mandatory_keywords = AGE_MANDATORY_KEYWORDS.get(target_age_group, []) if target_age_group else []
    logger.info(
        "search age_group=%s forbidden=%s mandatory=%s",
        target_age_group,
        forbidden_age_keywords,
        mandatory_keywords,
    )

    allergy_roots = _build_allergy_roots(state.get("allergies") or [])
    candidates = [
        candidate
        for candidate in candidates
        if _is_safe_candidate(
            candidate,
            allergy_roots=allergy_roots,
            target_age_group=target_age_group,
            mandatory_keywords=mandatory_keywords,
            forbidden_age_keywords=forbidden_age_keywords,
        )
    ]
    target_count = int(state.get("recommendation_limit") or 5)
    candidates, requested_product_strict_count, requested_product_relaxed_count = (
        _prioritize_requested_product_candidates(
            candidates,
            requested_product_terms,
            target_count=target_count,
        )
    )

    candidate_count_by_stage = dict(state.get("candidate_count_by_stage") or {})
    candidate_count_by_stage[str(relaxation)] = len(candidates)
    if requested_product_terms:
        candidate_count_by_stage[f"{relaxation}:requested_product_strict"] = requested_product_strict_count
        candidate_count_by_stage[f"{relaxation}:requested_product_relaxed"] = requested_product_relaxed_count
    relaxed_filters = build_relaxed_filter_names(
        relaxation=relaxation,
        filters=original_filters,
        health_concerns=health_concerns,
        age_group=state.get("age_group"),
        breed=(state.get("pet_profile") or {}).get("breed"),
    )

    logger.info(
        "search candidates=%s relaxation=%s target_age_group=%s relaxed=%s requested_terms=%s requested_strict=%s requested_relaxed=%s",
        len(candidates),
        relaxation,
        target_age_group,
        relaxed_filters,
        requested_product_terms,
        requested_product_strict_count,
        requested_product_relaxed_count,
    )
    return {
        "search_results": candidates,
        "last_search_goods_ids": [
            str(candidate.get("goods_id"))
            for candidate in candidates
            if candidate.get("goods_id") is not None
        ],
        "original_filters": original_filters,
        "effective_filters": filters,
        "exclusions": exclusions,
        "relaxed_filters": relaxed_filters,
        "candidate_count_by_stage": candidate_count_by_stage,
        "requested_product_terms": requested_product_terms,
        "requested_product_strict_count": requested_product_strict_count,
        "requested_product_relaxed_count": requested_product_relaxed_count,
    }
