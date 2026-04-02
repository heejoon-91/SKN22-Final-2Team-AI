import re
import psycopg2.extras
from final_ai.observability import traceable
from final_ai.pipeline.state import ChatState
from final_ai.pipeline.utils import (
    LLM_MODEL,
    build_pet_context,
    create_llm_completion,
    ensure_request_active,
    execute_traced,
    get_db_connection,
    hybrid_search_pg,
    normalize_pet_species,
    trace_log,
)


# ── profile_node ──────────────────────────────────────────────────────────────

# 핵심 동물성 명사 (1글자라도 상품명 등에서 무조건 차단해야 하는 것들)
CORE_ANIMAL_PLANTS = {"닭", "소", "양", "말", "굴", "게", "꿀", "오리", "연어", "참치", "돼지"}


@traceable(name="profile_node", run_type="chain")
def profile_node(state: ChatState) -> dict:
    """
    user_id가 있으면 DB에서 pet 정보를 조회하고, 해당 품종의 breed_meta 정보를 결합합니다.
    """
    user_id = state.get("user_id")
    target_pet_id = state.get("target_pet_id")
    pet_profile = dict(state.get("pet_profile") or {})
    health_concerns = list(state.get("health_concerns") or [])
    allergies = list(state.get("allergies") or [])
    food_prefs = list(state.get("food_preferences") or [])
    breed_context = ""
    health_traits = ""
    budget_val = None

    # 1. DB 또는 현재 상태에서 기초 정보 확보
    target_breed = pet_profile.get("breed")
    target_species = pet_profile.get("species")
    target_age = 0
    is_pet_override = state.get("is_pet_override", False)
    
    # 나이(숫자) 추출 시도 (예: "7살" -> 7)
    try:
        age_val = pet_profile.get("age") or ""
        if isinstance(age_val, int): target_age = age_val
        else:
            nums = re.findall(r'\d+', str(age_val))
            if nums: target_age = int(nums[0])
    except: pass

    pet_mismatch = False

    if user_id:
        conn = None
        cur = None
        try:
            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # [FIX] 사용자가 명시적으로 품종/종을 언급한 경우(is_pet_override), 
            # target_pet_id가 없다면 DB에서 최근 펫을 가져오지 않고 해당 입력 정보만 사용함.
            if target_pet_id:
                execute_traced(
                    cur,
                    "profile_node_pet_lookup_by_id",
                    """
                    SELECT pet_id, name, species, breed, age_years, age_months, weight_kg, gender, budget_range
                    FROM pet
                    WHERE user_id = %s AND pet_id = %s
                    LIMIT 1
                    """,
                    (user_id, target_pet_id),
                    user_id=user_id,
                    pet_id=target_pet_id,
                )
                pet_row = cur.fetchone()
            elif is_pet_override:
                # 명시적 override 상황에서는 DB 자동 조회를 건너뜀
                pet_row = None
                trace_log("profile_node_pet_override_skip_db")
            else:
                execute_traced(
                    cur,
                    "profile_node_pet_lookup_latest",
                    """
                    SELECT pet_id, name, species, breed, age_years, age_months, weight_kg, gender, budget_range
                    FROM pet WHERE user_id = %s ORDER BY created_at DESC LIMIT 1
                    """,
                    (user_id,),
                    user_id=user_id,
                )
                pet_row = cur.fetchone()

            if pet_row:
                # [불일치 검증 로직 추가]
                db_species = normalize_pet_species(pet_row["species"])
                chat_species = normalize_pet_species(target_species)
                db_breed = pet_row["breed"]
                
                # 펫 전환이 이미 처리된 경우(is_pet_switched), DB에서 가져온 정보이므로 불일치 체크를 건너뜀
                is_pet_switched = state.get("is_pet_switched", False)
                if not is_pet_switched:
                    # 1. 종 불일치 체크
                    if chat_species and db_species != chat_species:
                        pet_mismatch = True
                    # 2. 품종 불일치 체크 (채팅에서 품종을 언급 도중 등록 품종과 다를 때)
                    if target_breed and db_breed != target_breed:
                        pet_mismatch = True
                trace_log(
                    "profile_node_pet_compare",
                    db_species=db_species,
                    chat_species=chat_species,
                    db_breed=db_breed,
                    requested_breed=target_breed,
                    pet_mismatch=pet_mismatch,
                    is_pet_switched=is_pet_switched,
                )
                
                # 불일치가 없거나 펫이 전환된 상황일 때만 프로필 업데이트 수행
                if not pet_mismatch:
                    pet_id = pet_row["pet_id"]
                    target_breed = pet_row["breed"] or target_breed
                    target_species = pet_row["species"] or target_species
                    target_age = pet_row["age_years"] or target_age
                    
                    pet_profile.update({
                        "name":    pet_row["name"],
                        "species": pet_row["species"],
                        "breed":   pet_row["breed"],
                        "age":     f"{pet_row['age_years']}세 {pet_row['age_months']}개월",
                        "weight":  f"{pet_row['weight_kg']}kg",
                        "gender":  pet_row["gender"],
                    })
                    
                    # 예산 범위 매핑
                    budget_raw = pet_row.get("budget_range")
                    if budget_raw == "under_5":    budget_val = 50000
                    elif budget_raw == "5_10":     budget_val = 100000
                    elif budget_raw == "10_20":    budget_val = 200000
                    
                    execute_traced(
                        cur,
                        "profile_node_health_concerns",
                        "SELECT concern FROM pet_health_concern WHERE pet_id = %s",
                        (pet_id,),
                        pet_id=pet_id,
                    )
                    health_concerns = [r[0] for r in cur.fetchall()] or health_concerns
                    execute_traced(
                        cur,
                        "profile_node_allergies",
                        "SELECT ingredient FROM pet_allergy WHERE pet_id = %s",
                        (pet_id,),
                        pet_id=pet_id,
                    )
                    allergies = [r[0] for r in cur.fetchall()] or allergies
                    execute_traced(
                        cur,
                        "profile_node_food_preferences",
                        "SELECT food_type FROM pet_food_preference WHERE pet_id = %s",
                        (pet_id,),
                        pet_id=pet_id,
                    )
                    food_prefs = [r[0] for r in cur.fetchall()] or food_prefs

        except Exception as e:
            print(f"[PROFILE] DB 조회 중 오류: {e}")
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()

    # 2. 품종 메타 정보 가져오기 (DB에 펫 정보가 없어도 target_breed가 있으면 수행)
    if target_breed:
        conn = None
        cur = None
        try:
            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            # 연령대 매칭 로직 (퍼피: 1세 미만, 시니어: 7세 이상, 나머지 어덜트)
            age_group = "어덜트"
            if target_age < 1:  age_group = "퍼피"
            elif target_age >= 7: age_group = "시니어"

            # 품종명 + 연령대 우선 매칭
            execute_traced(
                cur,
                "profile_node_breed_meta",
                """
                SELECT preferred_food, health_products, chunk_text
                FROM breed_meta 
                WHERE (breed_name = %s OR breed_name_en ILIKE %s)
                ORDER BY CASE WHEN age_group = %s THEN 0 ELSE 1 END, id ASC
                LIMIT 1
                """,
                (target_breed, f"%{target_breed}%", age_group),
                breed=target_breed,
                age_group=age_group,
            )
            
            bm = cur.fetchone()
            if bm:
                parts = []
                if bm["preferred_food"]:  parts.append(f"선호 사료: {bm['preferred_food']}")
                if bm["health_products"]: parts.append(f"추천 건강제품: {bm['health_products']}")
                if bm["chunk_text"]:      parts.append(f"품종 특성: {bm['chunk_text']}")
                breed_context = "\n".join(parts)
                
                # 정규표현식으로 건강 특징 추출 (더 유연하게 파싱)
                chunk = bm["chunk_text"] or ""
                match = re.search(r"\[건강 특징\](.*?)(\[|$)", chunk, re.DOTALL)
                if match:
                    health_traits = match.group(1).strip()
                    # 혹시 다른 태그가 포함되어 있다면 그 전까지 자름
                    if "[" in health_traits: 
                        health_traits = health_traits.split("[")[0].strip()

        except Exception as e:
            print(f"[PROFILE] BreedMeta 조회 중 오류: {e}")
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()

    trace_log(
        "profile_node_result",
        user_id=user_id,
        pet_name=pet_profile.get("name"),
        breed=pet_profile.get("breed"),
        health_concerns=len(health_concerns),
        allergies=len(allergies),
        food_preferences=len(food_prefs),
        pet_mismatch=pet_mismatch,
    )
    return {
        "pet_profile":      pet_profile,
        "health_concerns":  health_concerns,
        "allergies":        allergies,
        "food_preferences": food_prefs,
        "breed_context":    breed_context,
        "health_traits":    health_traits,
        "budget":           budget_val,
        "pet_mismatch":     pet_mismatch
    }


# ── query_node ────────────────────────────────────────────────────────────────

@traceable(name="query_node", run_type="chain")
def query_node(state: ChatState) -> dict:
    """
    LLM을 통해 검색 쿼리를 생성합니다.
    filters에서 pet_type, category, subcategory를 추출합니다.
    """
    pet_ctx  = build_pet_context(state)
    filters  = state.get("filters") or {}
    relaxation = state.get("filter_relaxation_count", 0)

    category_hint = filters.get("category") or ""
    raw_sub = filters.get("subcategory") or ""
    # 재검색(relaxation>0) 중에도 STRICT_SUBCATEGORIES(캔/파우치 등 제형)는 절대 해제하지 않음
    subcategory_hint = raw_sub if (relaxation == 0 or raw_sub in STRICT_SUBCATEGORIES) else ""

    prominent_concerns = ", ".join(state.get("health_concerns") or [])
    concern_clause = (
        f"특히 다음 건강 고민사항을 반드시 해결할 수 있는 상품 위주로 검색어를 구성하세요: {prominent_concerns}\n"
        f"- **사료(주식)와 간식(보상용)을 엄격히 구분하세요.**\n"
        f"- **캔(Can)과 파우치(Pouch)는 서로 다른 제형입니다. 사용자가 \"캔\"을 언급하면 반드시 \"캔\"이 포함된 소분류를, \"파우치\"를 언급하면 \"파우치\"가 포함된 소분류를 선택하세요.**"
    ) if prominent_concerns else (
        f"- **사료(주식)와 간식(보상용)을 엄격히 구분하세요.**\n"
        f"- **캔(Can)과 파우치(Pouch)는 서로 다른 제형입니다. 사용자가 \"캔\"을 언급하면 반드시 \"캔\"이 포함된 소분류를, \"파우치\"를 언급하면 \"파우치\"가 포함된 소분류를 선택하세요.**"
    )

    prompt = (
        f"반려동물 상품 검색을 위한 최적화된 한국어 검색어를 한 문장으로만 반환하세요.\n"
        f"펫 정보: {pet_ctx}\n"
        f"품종 특성 지식:\n{state.get('breed_context') or '없음'}\n"
        f"카테고리: {category_hint} / 세부: {subcategory_hint}\n"
        f"{concern_clause}\n"
        f"원래 질문: {state['user_input']}"
    )
    try:
        ensure_request_active()
        search_query = create_llm_completion(
            trace_label="query_node_search_query",
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        ).choices[0].message.content.strip()
    except Exception as e:
        fallback_parts = [category_hint, subcategory_hint, state["user_input"]]
        search_query = " ".join(part for part in fallback_parts if part).strip() or state["user_input"]
        trace_log("query_node_fallback", error_type=type(e).__name__, error=str(e))

    trace_log(
        "query_node_result",
        query=search_query,
        relaxation=relaxation,
        category=category_hint,
        subcategory=subcategory_hint,
    )
    return {
        "search_query": search_query,
        "filters": {
            "pet_type":    filters.get("pet_type"),
            "category":    category_hint,
            "subcategory": subcategory_hint,
        },
    }


# ── search_node ───────────────────────────────────────────────────────────────

# 완화(Relaxation) 시에도 절대로 해제하지 않을 제형 관련 핵심 소분류
STRICT_SUBCATEGORIES = {
    "주식캔", "주식파우치", "간식캔", "간식파우치", "습식사료", "캔/파우치", 
    "동결건조/에어드라이", "동결/건조간식", "화식", "소프트사료"
}


@traceable(name="search_node", run_type="chain")
def search_node(state: ChatState) -> dict:
    """
    PostgreSQL hybrid_search_pg 실행:
    pgvector(코사인 유사도) + tsvector(FTS) + RRF 결합
    """
    query      = state.get("search_query") or state["user_input"]
    filters    = state.get("filters") or {}
    relaxation = state.get("filter_relaxation_count", 0)

    pet_type    = filters.get("pet_type")
    category    = filters.get("category")
    
    # [수정] 소분류 필터 적용 로직:
    # 1. 초기 검색(relaxation=0) 시에는 소분류를 항상 적용.
    # 2. 완화 검색(relaxation>0) 시에는 일반 소분류는 해제하지만, STRICT_SUBCATEGORIES에 해당하면 유지.
    subcategory = filters.get("subcategory")
    if relaxation > 0 and subcategory not in STRICT_SUBCATEGORIES:
        subcategory = None
    budget      = state.get("budget")

    pt_kr = normalize_pet_species(pet_type)
    if not pt_kr:
        pt_kr = normalize_pet_species((state.get("pet_profile") or {}).get("species"))

    candidates = hybrid_search_pg(
        query=query,
        top_k=50,
        pet_type=pt_kr,
        category=category,
        subcategory=subcategory,
        budget=budget,
    )
    print(f"[SEARCH-DEBUG] hybrid_search_pg 반환: {len(candidates)}개 | subcategory={subcategory} | category={category} | pet={pt_kr}")
    # [추가] 샘플/체험팩 상품 강제 제외 (SQL 필터 우회 대비 2차 방어)
    blacklist_words = ["샘플", "맛보기", "체험팩"]
    candidates = [
        c for c in candidates 
        if not any(bw in c.get("goods_name", "") for bw in blacklist_words)
    ]
    print(f"[SEARCH-DEBUG] 블랙리스트 필터 후: {len(candidates)}개")

    # [수정v2] 제형(캔/파우치) 적극 포함 필터링 (Positive Filter)
    # - 명확한 캔 전용 subcategory (주식캔, 간식캔): goods_name 조건 없이 통과
    # - 혼합 subcategory (캔/파우치): goods_name에 해당 제형 키워드가 있어야 통과
    #   예) 강아지 "캔/파우치" subcategory에서 파우치 제품이 섞여 나오는 것 방지

    PURE_CAN_SUBS   = {"주식캔", "간식캔"}          # 캔 전용 subcategory
    PURE_POUCH_SUBS = {"주식파우치", "간식파우치"}   # 파우치 전용 subcategory
    MIXED_SUBS      = {"캔/파우치"}                 # 혼합 subcategory → goods_name으로 추가 판단

    def _get_subs(c: dict) -> list:
        import ast
        subs = c.get("subcategory") or []
        if isinstance(subs, str):
            try: subs = ast.literal_eval(subs)
            except: subs = [subs]
        return subs

    user_input_lower = state["user_input"].lower()
    print(f"[SEARCH-DEBUG] user_input_lower={user_input_lower!r} | 캔포함={'캔' in user_input_lower} | 파우치포함={'파우치' in user_input_lower}")
    if "캔" in user_input_lower and "파우치" not in user_input_lower:
        # 캔 요청 → 캔 전용 subcategory는 무조건 통과,
        #            혼합(캔/파우치) subcategory는 goods_name에 "캔" 있어야 통과
        def is_can_product(c):
            subs = _get_subs(c)
            name = c.get("goods_name", "")
            if any(s in PURE_CAN_SUBS for s in subs):
                return True   # 캔 전용 subcategory → 무조건 통과
            if any(s in MIXED_SUBS for s in subs) and "캔" in name:
                return True   # 혼합 subcategory → goods_name에 "캔" 있어야 통과
            if "캔" in name:
                return True   # subcategory 없어도 goods_name에 "캔" 있으면 통과
            return False
        candidates = [c for c in candidates if is_can_product(c)]
        print(f"[SEARCH-DEBUG] 캔 필터 후: {len(candidates)}개 | 상품명: {[c.get('goods_name','?')[:30] for c in candidates[:5]]}")

    elif "파우치" in user_input_lower and "캔" not in user_input_lower:
        # 파우치 요청 → 파우치 전용 subcategory는 무조건 통과,
        #               혼합(캔/파우치) subcategory는 goods_name에 "캔"이 없어야 통과
        #               (파우치 상품은 이름에 "캔"이 안 들어감)
        def is_pouch_product(c):
            subs = _get_subs(c)
            name = c.get("goods_name", "")
            if any(s in PURE_POUCH_SUBS for s in subs):
                return True   # 파우치 전용 subcategory → 무조건 통과
            if any(s in MIXED_SUBS for s in subs) and "캔" not in name:
                return True   # 혼합 subcategory → goods_name에 "캔" 없어야 통과
            if "파우치" in name:
                return True   # subcategory 없어도 goods_name에 "파우치" 있으면 통과
            return False
        candidates = [c for c in candidates if is_pouch_product(c)]
        print(f"[SEARCH-DEBUG] 파우치 필터 후: {len(candidates)}개 | 상품명: {[c.get('goods_name','?')[:30] for c in candidates[:5]]}")
    else:
        print(f"[SEARCH-DEBUG] 제형 필터 미적용")

    # [GP 보충] 5개 미만이면 모아보기(GP) 상품으로 보충
    _MIN_CANDIDATES = 5
    if len(candidates) < _MIN_CANDIDATES:
        existing_ids = {c["goods_id"] for c in candidates}
        # 알레르기 여분 확보를 위해 더 넉넉하게 가져옴
        needed = max((_MIN_CANDIDATES - len(candidates)) * 4, 12)

        try:
            _conn = get_db_connection()
            _cur = _conn.cursor()

            gp_filter_parts = [
                "AND goods_id LIKE 'GP%%'",            # GP 상품만
                "AND goods_name NOT ILIKE '%%샘플%%'",  # 샘플 제외
            ]
            gp_params = []

            # [핵심] 제형 조건을 SQL 단계에서 직접 적용 (LIMIT 낭비 없이)
            if "캔" in user_input_lower and "파우치" not in user_input_lower:
                gp_filter_parts.append("AND goods_name ILIKE '%%캔%%'")
            elif "파우치" in user_input_lower and "캔" not in user_input_lower:
                gp_filter_parts.append("AND goods_name NOT ILIKE '%%캔%%'")

            if pt_kr:
                gp_filter_parts.append("AND %s = ANY(pet_type)")
                gp_params.append(pt_kr)
            if category:
                gp_filter_parts.append("AND (%s = ANY(category) OR %s = ANY(subcategory))")
                gp_params.extend([category, category])
            if subcategory:
                gp_filter_parts.append("AND %s = ANY(subcategory)")
                gp_params.append(subcategory)

            gp_sql = (
                "SELECT goods_id, goods_name, pet_type, category, subcategory,"
                "       price, thumbnail_url, product_url, brand_name, discount_price,"
                "       popularity_score, sentiment_avg, repeat_rate, health_concern_tags,"
                "       rating, review_count, main_ingredients"
                " FROM product"
                " WHERE 1=1 "
                + " ".join(gp_filter_parts) +
                " ORDER BY popularity_score DESC NULLS LAST, review_count DESC NULLS LAST"
                " LIMIT %s"
            )

            execute_traced(
                _cur,
                "search_node_gp_fill",
                gp_sql,
                gp_params + [needed],
                pet_type=pt_kr,
                category=category,
                subcategory=subcategory,
                needed=needed,
            )
            gp_cols = [d[0] for d in _cur.description]
            gp_rows = [dict(zip(gp_cols, row)) for row in _cur.fetchall()
                       if row[0] not in existing_ids]

            _cur.close()
            _conn.close()

            print(f"[SEARCH-DEBUG] GP 쿼리 결과: {len(gp_rows)}개 (SQL 자체 제형 필터 적용됨)")
            if gp_rows:
                print(f"[SEARCH] GP 보충: {len(gp_rows)}개 추가 (현재 {len(candidates)}개 → 목표 {_MIN_CANDIDATES}개+)")
                print(f"[SEARCH-DEBUG] GP 상품명: {[c.get('goods_name','?')[:30] for c in gp_rows[:5]]}")
                candidates.extend(gp_rows)
            else:
                print(f"[SEARCH-DEBUG] GP 보충 없음 (해당 제형 GP 상품 자체 없음)")

        except Exception as _e:
            print(f"[SEARCH] GP 보충 실패: {_e}")

    # 알레르기 post-filter
    allergies = state.get("allergies") or []
    if allergies:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        
        # 1. 알레르기 키워드에서 핵심 명사 추출
        stop_nouns = {"고기", "가루", "분말", "생물", "제품", "성분", "첨가물", "함유", "용", "포함"}
        # 1글자 동물 키워드가 포함되어도 무시해야 할 단어들 (오탐 방지)
        safe_words = {"말티즈", "소프트", "소화", "소형", "소형견", "소프", "말티", "소중형", "말랑"}
        
        def get_roots(text_list):
            roots = set()
            for text in text_list:
                text_lower = str(text).lower()
                # A. 형태소 분석 기반 추출
                for token in kiwi.tokenize(text_lower):
                    if token.tag.startswith("NN"):
                        if token.form not in stop_nouns and len(token.form) >= 1:
                            roots.add(token.form)
                # B. 핵심 동물성 키워드 강제 추출 (복합어/미분절 대응)
                # 오탐 방지를 위해 safe_words가 포함된 경우 해당 단어를 제외하고 검사하거나 전처리
                cleaned_text = text_lower
                for sw in safe_words:
                    cleaned_text = cleaned_text.replace(sw, " ")
                
                for animal in CORE_ANIMAL_PLANTS:
                    if animal in cleaned_text:
                        roots.add(animal)
                # C. 원문 자체 추가
                roots.add(text_lower)
            return roots

        allergy_roots = get_roots(allergies)
        print(f"[SEARCH] Allergy Roots: {allergy_roots}")

        def is_safe(c):
            ocr_text = str(c.get("ingredient_text_ocr") or "").lower()
            goods_name = str(c.get("goods_name") or "").lower()

            for r in allergy_roots:
                if len(r) > 1:
                    if r in ocr_text or r in goods_name: return False
                else:
                    if r in CORE_ANIMAL_PLANTS:
                        # [오탐 방지] 1글자 코어 동물(말, 소 등)이 말티즈, 소프트 등의 일부인 경우 제외
                        c_ocr = ocr_text
                        c_name = goods_name
                        for sw in safe_words:
                            c_ocr = c_ocr.replace(sw, " ")
                            c_name = c_name.replace(sw, " ")
                        
                        if r in c_ocr or r in c_name: return False
                    else:
                        # 일반 1글자 어근은 오탐 방지를 위해 공백 포함 체크
                        if f" {r} " in f" {ocr_text} " or f" {r} " in f" {goods_name} ":
                            return False

            # B. 메인 성분 리스트 검사 (형태소 단위 체크)
            m_ingredients = c.get("main_ingredients") or []
            if isinstance(m_ingredients, str):
                import json
                try: m_ingredients = json.loads(m_ingredients)
                except: m_ingredients = [m_ingredients]
            
            for ing in m_ingredients:
                ing_lower = str(ing).lower()
                # 1. 단순 전체 포함 여부 (이미 구현됨)
                if any(a.lower() in ing_lower for a in allergies):
                    return False
                
                # 2. 형태소 어근 교집합 및 상호 포함 체크
                ing_roots = get_roots([ing_lower])
                
                for r_all in allergy_roots:
                    for r_ing in ing_roots:
                        # 완벽 일치
                        if r_all == r_ing: return False
                        # 상호 포함 관계 (예: 닭고기-닭, 닭가슴살-닭)
                        if (r_all in r_ing) or (r_ing in r_all):
                            # 오탐 방지: 둘 중 하나가 코어 동물이거나, 겹치는 어근이 충분히 길 때
                            if (r_all in CORE_ANIMAL_PLANTS) or (r_ing in CORE_ANIMAL_PLANTS) or (len(set(r_all) & set(r_ing)) >= 2):
                                return False
            return True
            
        candidates = [c for c in candidates if is_safe(c)]

    trace_log(
        "search_node_result",
        candidate_count=len(candidates),
        relaxation=relaxation,
        pet_type=pt_kr,
        category=category,
        subcategory=subcategory,
        allergies=len(allergies),
    )
    return {"search_results": candidates}


# ── rerank_node ───────────────────────────────────────────────────────────────

_ALPHA   = 0.50
_BETA    = 0.25
_GAMMA   = 0.15
_DELTA   = 0.10
_EPSILON = 0.10
_TOP_K   = 5


def _normalize(values: list[float]) -> list[float]:
    mn, mx = min(values), max(values)
    if mx == mn:
        return [1.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


@traceable(name="rerank_node", run_type="chain")
def rerank_node(state: ChatState) -> dict:
    """재랭킹: RRF 점수 + 인기도·감성·재구매율 가중치"""
    candidates      = state.get("search_results") or []
    detected_aspect = state.get("detected_aspect")
    intents         = state.get("intents") or []
    relaxation      = state.get("filter_relaxation_count", 0)

    if not candidates:
        should_retry = relaxation < 1
        next_relaxation = relaxation + 1 if should_retry else relaxation
        print(f"[RERANK] 후보 없음 (relaxation={relaxation}, retry={should_retry})")
        return {
            "reranked_results": [],
            "filter_relaxation_count": next_relaxation,
            "recommend_retry_pending": should_retry,
        }

    # "popularity" 의도 여부 확인
    is_popularity_mode = "popularity" in intents

    rrf_scores  = [float(c.get("_score", 0.0)) for c in candidates]
    pop_scores  = [float(c.get("popularity_score") or 0.0) for c in candidates]
    sent_scores = [float(c.get("sentiment_avg") or 0.0)    for c in candidates]
    rep_scores  = [float(c.get("repeat_rate") or 0.0)      for c in candidates]

    # 각 지표 정규화
    norm_rrf  = _normalize(rrf_scores)
    norm_pop  = _normalize(pop_scores)
    norm_sent = _normalize(sent_scores)
    norm_rep  = _normalize(rep_scores)

    scored = []
    for i, c in enumerate(candidates):
        if is_popularity_mode:
            # [인기 상품 모드] 판매량(50%) + 평점(25%) + 재구매율(25%)
            # 검색 엔진 점수(RRF)는 필터링 용도로만 쓰고 점수 계산에서는 제외하거나 아주 작게 유지
            score = (
                0.50 * norm_pop[i] +
                0.25 * norm_sent[i] +
                0.25 * norm_rep[i] +
                0.01 * norm_rrf[i]  # 최소한의 유사도 유지
            )
        else:
            # [일반 추천 모드] 기존 가중치 유지
            has_pop       = c.get("popularity_score") is not None
            has_sentiment = c.get("sentiment_avg")    is not None
            has_repeat    = c.get("repeat_rate")      is not None

            if not has_pop and not has_sentiment and not has_repeat:
                score = norm_rrf[i]
            elif not has_sentiment and not has_repeat:
                score = _ALPHA * norm_rrf[i] + 0.35 * norm_pop[i]
            else:
                score = (
                    _ALPHA * norm_rrf[i]
                    + _BETA  * norm_pop[i]
                    + _GAMMA * norm_sent[i]
                    + _DELTA * norm_rep[i]
                )

        # 관점 기반 추가 가점 (일반 모드 전용)
        if not is_popularity_mode and detected_aspect and c.get("sentiment_avg") is not None:
            score += _EPSILON * float(c["sentiment_avg"])

        # 건강관심사 및 품종 특성 매칭 가산점
        user_concerns = state.get("health_concerns") or []
        health_traits = state.get("health_traits") or ""
        product_tags  = c.get("health_concern_tags") or []
        
        if isinstance(product_tags, str):
            import ast
            try: product_tags = ast.literal_eval(product_tags)
            except: product_tags = [product_tags]

        # 1. 사용자 직접 등록 건강관심사 매칭 (+0.2)
        if any(h in product_tags for h in user_concerns):
            score += 0.20
            
        # 2. 품종별 취약 건강 정보(health_traits) 내 키워드 매칭 (+0.1)
        if health_traits:
            trait_keywords = [k for k in ["슬개골", "기관허탈", "눈물", "피부", "관절", "체중", "소화", "신장", "심장"] if k in health_traits]
            if any(k in product_tags for k in trait_keywords):
                score += 0.10

        scored.append((score, c))

    # 점수 높은 순으로 정렬
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[:_TOP_K]]

    should_retry = len(top) < 3 and relaxation < 1
    new_relaxation = relaxation + 1 if should_retry else relaxation
    
    mode_str = "POPULARITY" if is_popularity_mode else "NORMAL"
    print(f"[RERANK] mode={mode_str}, final {len(top)}개 (relaxation={relaxation})")

    return {
        "reranked_results": top,
        "filter_relaxation_count": new_relaxation,
        "recommend_retry_pending": should_retry,
    }
