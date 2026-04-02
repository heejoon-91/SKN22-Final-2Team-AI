import os
import re
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar

import psycopg2
from dotenv import load_dotenv, find_dotenv

# .env 파일 로드 (find_dotenv로 상위 디렉토리까지 탐색)
load_dotenv(find_dotenv())

from fastembed import TextEmbedding
from final_ai.observability import wrap_openai
from final_ai.pipeline.state import ChatState

# ── 클라이언트 (lazy init) ──────────────────────────────────────────────────────
_llm = None
LLM_MODEL = "gpt-4o-mini"
LLM_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))
POSTGRES_CONNECT_TIMEOUT_SECONDS = int(os.getenv("POSTGRES_CONNECT_TIMEOUT_SECONDS", "5"))
POSTGRES_STATEMENT_TIMEOUT_MS = int(os.getenv("POSTGRES_STATEMENT_TIMEOUT_MS", "20000"))
EMBED_MODEL_NAME = os.getenv("FASTEMBED_MODEL", "intfloat/multilingual-e5-large")
_SUPPORTED_EMBED_MODELS = {
    model["model"]: model for model in TextEmbedding.list_supported_models()
}
_embed_model_unavailable_reason = None
_request_cancel_event: ContextVar[object | None] = ContextVar("request_cancel_event", default=None)
_request_trace: ContextVar[dict | None] = ContextVar("request_trace", default=None)


class RequestCancelled(RuntimeError):
    pass


def _trim_text(value: str, limit: int = 160) -> str:
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _normalize_trace_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _normalize_trace_value(v) for k, v in list(value.items())[:10]}
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if len(items) > 10:
            items = items[:10] + ["..."]
        return [_normalize_trace_value(item) for item in items]
    return _trim_text(repr(value), limit=120)


def _compact_sql(sql: str | None) -> str | None:
    if not sql:
        return None
    return _trim_text(sql, limit=180)


def get_request_trace() -> dict:
    return dict(_request_trace.get() or {})


def update_request_trace(**fields) -> None:
    current = dict(_request_trace.get() or {})
    current.update({key: _normalize_trace_value(value) for key, value in fields.items() if value is not None})
    _request_trace.set(current)


def trace_log(event: str, **fields) -> None:
    payload = {**get_request_trace(), **{key: _normalize_trace_value(value) for key, value in fields.items() if value is not None}}
    serialized = " ".join(
        f"{key}={json.dumps(value, ensure_ascii=False)}"
        for key, value in sorted(payload.items())
    )
    print(f"[FASTAPI_TRACE] event={event} {serialized}".rstrip())


@contextmanager
def bind_request_trace(request_id: str, **fields):
    trace = {"request_id": request_id}
    trace.update({key: _normalize_trace_value(value) for key, value in fields.items() if value is not None})
    token = _request_trace.set(trace)
    try:
        yield
    finally:
        _request_trace.reset(token)


@contextmanager
def trace_block(event: str, **fields):
    started_at = time.perf_counter()
    trace_log(f"{event}_start", **fields)
    try:
        yield
    except RequestCancelled:
        trace_log(f"{event}_cancelled", elapsed_ms=round((time.perf_counter() - started_at) * 1000, 1), **fields)
        raise
    except Exception as exc:
        trace_log(
            f"{event}_error",
            elapsed_ms=round((time.perf_counter() - started_at) * 1000, 1),
            error_type=type(exc).__name__,
            error=str(exc),
            **fields,
        )
        raise
    else:
        trace_log(f"{event}_done", elapsed_ms=round((time.perf_counter() - started_at) * 1000, 1), **fields)


def execute_traced(cur, label: str, sql: str, params=None, **fields) -> None:
    params = params or ()
    with trace_block(
        "db_query",
        label=label,
        sql=_compact_sql(sql),
        param_count=len(params) if hasattr(params, "__len__") else None,
        **fields,
    ):
        cur.execute(sql, params)


@contextmanager
def bind_request_cancel_event(cancel_event):
    token = _request_cancel_event.set(cancel_event)
    try:
        yield
    finally:
        _request_cancel_event.reset(token)


def ensure_request_active() -> None:
    cancel_event = _request_cancel_event.get()
    if cancel_event is not None and cancel_event.is_set():
        raise RequestCancelled("Client disconnected")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_memory_limit_bytes() -> int | None:
    paths = (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            limit = int(raw)
        except ValueError:
            continue
        if limit > 0 and limit < (1 << 60):
            return limit
    return None


def _read_host_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0 or total_pages <= 0:
        return None
    return page_size * total_pages


def _total_memory_gb() -> float | None:
    limits = [value for value in (_read_memory_limit_bytes(), _read_host_memory_bytes()) if value]
    if not limits:
        return None
    return min(limits) / (1024 ** 3)


def _embed_model_size_gb() -> float | None:
    model = _SUPPORTED_EMBED_MODELS.get(EMBED_MODEL_NAME)
    if not model:
        return None
    size = model.get("size_in_GB")
    try:
        return float(size)
    except (TypeError, ValueError):
        return None


def _dense_model_allowed() -> tuple[bool, str | None]:
    if not _env_flag("FASTEMBED_ENABLED", default=True):
        return False, "FASTEMBED_ENABLED=false"

    if not _env_flag("FASTEMBED_AUTO_DISABLE_LOW_MEMORY", default=True):
        return True, None

    total_memory_gb = _total_memory_gb()
    model_size_gb = _embed_model_size_gb()
    headroom_gb = float(os.getenv("FASTEMBED_MEMORY_HEADROOM_GB", "0.75"))

    if total_memory_gb is None or model_size_gb is None:
        return True, None

    required_memory_gb = model_size_gb + headroom_gb
    if total_memory_gb < required_memory_gb:
        return (
            False,
            "insufficient memory for dense embedding "
            f"(available={total_memory_gb:.2f}GB, required~={required_memory_gb:.2f}GB, model={EMBED_MODEL_NAME})",
        )

    return True, None

def get_llm():
    global _llm
    if _llm is None:
        from openai import OpenAI
        _llm = wrap_openai(OpenAI(timeout=LLM_TIMEOUT_SECONDS, max_retries=0))
    return _llm

# 기존 코드 호환을 위한 별칭 (노드들이 llm.chat.completions.create 형식으로 호출)
class _LazyLLM:
    def __getattr__(self, name):
        return getattr(get_llm(), name)

llm = _LazyLLM()


def create_llm_completion(*, trace_label: str, **kwargs):
    ensure_request_active()
    model = kwargs.get("model")
    with trace_block("llm_call", label=trace_label, model=model):
        response = get_llm().chat.completions.create(**kwargs)
    choice = None
    if getattr(response, "choices", None):
        choice = response.choices[0]
    message = getattr(getattr(choice, "message", None), "content", "") or ""
    trace_log(
        "llm_call_result",
        label=trace_label,
        model=model,
        output_chars=len(message),
        finish_reason=getattr(choice, "finish_reason", None),
    )
    return response

# ── DB 연결 설정 ────────────────────────────────────────────────────────────────
_PET_SPECIES_KR = {
    "dog": "강아지",
    "cat": "고양이",
    "강아지": "강아지",
    "고양이": "고양이",
}

_SEARCH_STOPWORDS = {
    "추천",
    "추천해줘",
    "추천해주세요",
    "추천해",
    "알려줘",
    "알려주세요",
    "좋은",
    "좋아요",
    "우리",
    "아이",
    "반려동물",
    "반려견",
    "반려묘",
}

_TOKEN_SUFFIXES = (
    "추천해주세요",
    "추천해줘",
    "알려주세요",
    "알려줘",
    "입니다",
    "이에요",
    "예요",
    "에요",
    "으로",
    "에서",
    "한테",
    "용",
    "에",
    "의",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "과",
    "와",
    "도",
    "로",
)


def normalize_pet_species(species: str | None) -> str | None:
    if not species:
        return None
    return _PET_SPECIES_KR.get(str(species).strip())


def _extract_search_terms(query: str, *extra_terms: str | None) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def add_token(raw: str):
        token = raw.strip()
        if not token:
            return
        if token in _SEARCH_STOPWORDS:
            return
        if token not in seen:
            seen.add(token)
            terms.append(token)

    for source in (query, *extra_terms):
        if not source:
            continue
        for raw in re.findall(r"[0-9A-Za-z가-힣/]+", str(source)):
            pieces = [piece for piece in raw.split("/") if piece]
            for piece in pieces:
                add_token(piece)
                for suffix in _TOKEN_SUFFIXES:
                    if len(piece) <= len(suffix) + 1:
                        continue
                    if piece.endswith(suffix):
                        trimmed = piece[: -len(suffix)].strip()
                        if len(trimmed) >= 2:
                            add_token(trimmed)
    return terms


def get_db_connection():
    """PostgreSQL 연결 반환.

    배포/로컬 실행 모두 `POSTGRES_HOST`에 지정된 값을 그대로 사용한다.
    - Docker compose 내부: `postgres`
    - 호스트 로컬 실행: `localhost`
    - 배포 환경: RDS endpoint
    """
    host = (os.getenv("POSTGRES_HOST") or "localhost").strip() or "localhost"
    with trace_block(
        "db_connect",
        host=host,
        db=os.getenv("POSTGRES_DB", "tailtalk_db"),
        statement_timeout_ms=POSTGRES_STATEMENT_TIMEOUT_MS,
    ):
        return psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", "tailtalk_db"),
            user=os.getenv("POSTGRES_USER", "mungnyang"),
            password=os.getenv("POSTGRES_PASSWORD", "finalprojectljs1908"),
            host=host,
            port=os.getenv("POSTGRES_PORT", "5432"),
            connect_timeout=POSTGRES_CONNECT_TIMEOUT_SECONDS,
            options=f"-c statement_timeout={POSTGRES_STATEMENT_TIMEOUT_MS}",
            application_name="tailtalk-fastapi",
        )

# ── 임베딩 모델 (lazy loading) ──────────────────────────────────────────────────
_embed_model = None

def get_embed_model() -> TextEmbedding | None:
    global _embed_model, _embed_model_unavailable_reason
    if _embed_model is not None:
        return _embed_model

    if _embed_model_unavailable_reason is not None:
        return None

    allowed, reason = _dense_model_allowed()
    if not allowed:
        _embed_model_unavailable_reason = reason
        print(f"[get_embed_model] dense embedding disabled: {reason}")
        return None

    try:
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH")
        local_files_only = _env_flag("FASTEMBED_LOCAL_FILES_ONLY", default=False)
        kwargs = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            kwargs["cache_dir"] = cache_dir
        if local_files_only:
            kwargs["local_files_only"] = True

        print(
            f"임베딩 모델 로드 중 ({EMBED_MODEL_NAME}, "
            f"local_only={local_files_only}, cache_dir={cache_dir or 'default'})..."
        )
        _embed_model = TextEmbedding(EMBED_MODEL_NAME, **kwargs)
        print("임베딩 모델 로드 완료")
    except Exception as e:
        _embed_model_unavailable_reason = str(e)
        raise
    return _embed_model


def embed_query(query: str) -> list[float] | None:
    """E5 모델로 쿼리 임베딩 생성.

    배포 환경에서 모델 로드에 실패하더라도 FTS 검색으로 폴백할 수 있게
    예외를 삼키고 `None`을 반환한다.
    """
    try:
        model = get_embed_model()
        if model is None:
            if _embed_model_unavailable_reason:
                print(f"[embed_query] dense embedding unavailable: {_embed_model_unavailable_reason}")
            return None
        return list(model.embed([f"query: {query}"]))[0].tolist()
    except Exception as e:
        print(f"[embed_query] dense embedding unavailable: {e}")
        return None


# ── Hybrid Search (Vector + Keyword + RRF) ──────────────────────────────────────

def hybrid_search_pg(query: str, top_k: int = 20,
                     pet_type: str | None = None,
                     category: str | None = None,
                     subcategory: str | None = None,
                     budget: int | None = None) -> list[dict]:
    """
    PostgreSQL의 pgvector(embedding 컬럼)와 tsvector(search_vector 컬럼)를
    이용한 Hybrid Search 후 RRF(Reciprocal Rank Fusion)로 상위 결과 반환.
    """
    with trace_block(
        "hybrid_search",
        query=query,
        top_k=top_k,
        pet_type=pet_type,
        category=category,
        subcategory=subcategory,
        budget=budget,
    ):
        query_vec = embed_query(query)
        pet_type_kr = normalize_pet_species(pet_type) or pet_type
        k = 60  # RRF 상수

        conn = None
        cur = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            vec_sql = """
                SELECT goods_id, goods_name, pet_type, category, subcategory,
                       price, thumbnail_url, product_url, brand_name, discount_price,
                       popularity_score, sentiment_avg, repeat_rate, health_concern_tags,
                       rating, review_count, main_ingredients
                FROM product
                WHERE 1=1 {filters}
                ORDER BY embedding <=> %s::vector
                LIMIT 100
            """
            keyword_sql = """
                SELECT goods_id, goods_name, pet_type, category, subcategory,
                       price, thumbnail_url, product_url, brand_name, discount_price,
                       popularity_score, sentiment_avg, repeat_rate, health_concern_tags,
                       rating, review_count, main_ingredients
                FROM product
                WHERE search_vector @@ plainto_tsquery('simple', %s) {filters}
                ORDER BY ts_rank(search_vector, plainto_tsquery('simple', %s)) DESC
                LIMIT 100
            """
            pop_sql = """
                SELECT goods_id, goods_name, pet_type, category, subcategory,
                       price, thumbnail_url, product_url, brand_name, discount_price,
                       popularity_score, sentiment_avg, repeat_rate, health_concern_tags,
                       rating, review_count, main_ingredients
                FROM product
                WHERE 1=1 {filters}
                ORDER BY popularity_score DESC NULLS LAST, review_count DESC NULLS LAST
                LIMIT 100
            """

            filter_parts = [
                "AND goods_name NOT ILIKE '%%샘플%%'",
                "AND goods_id NOT LIKE 'GP%%'",
            ]
            filter_params_shared = []

            if pet_type_kr:
                filter_parts.append("AND %s = ANY(pet_type)")
                filter_params_shared.append(pet_type_kr)
            if category:
                filter_parts.append("AND (%s = ANY(category) OR %s = ANY(subcategory))")
                filter_params_shared.extend([category, category])
            if subcategory:
                filter_parts.append("AND %s = ANY(subcategory)")
                filter_params_shared.append(subcategory)
            if budget:
                filter_parts.append("AND price <= %s")
                filter_params_shared.append(budget)

            filter_str = " ".join(filter_parts)

            cols = []
            vec_rows = []
            if query_vec is not None:
                execute_traced(
                    cur,
                    "hybrid_search_vector",
                    vec_sql.format(filters=filter_str),
                    filter_params_shared + [query_vec],
                    category=category,
                    subcategory=subcategory,
                )
                cols = [d[0] for d in cur.description]
                vec_rows = [dict(zip(cols, row)) for row in cur.fetchall()]
                trace_log("db_query_rows", label="hybrid_search_vector", row_count=len(vec_rows))

            execute_traced(
                cur,
                "hybrid_search_keyword",
                keyword_sql.format(filters=filter_str),
                [query] + filter_params_shared + [query],
                category=category,
                subcategory=subcategory,
            )
            kw_cols = [d[0] for d in cur.description]
            if not cols:
                cols = kw_cols
            kw_rows = [dict(zip(kw_cols, row)) for row in cur.fetchall()]
            trace_log("db_query_rows", label="hybrid_search_keyword", row_count=len(kw_rows))

            execute_traced(
                cur,
                "hybrid_search_popularity",
                pop_sql.format(filters=filter_str),
                filter_params_shared,
                category=category,
                subcategory=subcategory,
            )
            pop_cols = [d[0] for d in cur.description]
            pop_rows = [dict(zip(pop_cols, row)) for row in cur.fetchall()]
            trace_log("db_query_rows", label="hybrid_search_popularity", row_count=len(pop_rows))

            if not vec_rows and not kw_rows:
                loose_terms = _extract_search_terms(query, category, subcategory)
                if loose_terms:
                    score_parts = []
                    score_params = []
                    where_parts = []
                    where_params = []

                    for term in loose_terms:
                        like = f"%{term}%"
                        score_parts.append(
                            "("
                            "CASE WHEN goods_name ILIKE %s THEN 5 ELSE 0 END + "
                            "CASE WHEN brand_name ILIKE %s THEN 2 ELSE 0 END + "
                            "CASE WHEN COALESCE(array_to_string(category, ' '), '') ILIKE %s THEN 3 ELSE 0 END + "
                            "CASE WHEN COALESCE(array_to_string(subcategory, ' '), '') ILIKE %s THEN 4 ELSE 0 END + "
                            "CASE WHEN COALESCE(array_to_string(health_concern_tags, ' '), '') ILIKE %s THEN 4 ELSE 0 END"
                            ")"
                        )
                        score_params.extend([like, like, like, like, like])
                        where_parts.append(
                            "("
                            "goods_name ILIKE %s OR "
                            "brand_name ILIKE %s OR "
                            "COALESCE(array_to_string(category, ' '), '') ILIKE %s OR "
                            "COALESCE(array_to_string(subcategory, ' '), '') ILIKE %s OR "
                            "COALESCE(array_to_string(health_concern_tags, ' '), '') ILIKE %s"
                            ")"
                        )
                        where_params.extend([like, like, like, like, like])

                    loose_sql = f"""
                        SELECT goods_id, goods_name, pet_type, category, subcategory,
                               price, thumbnail_url, product_url, brand_name, discount_price,
                               popularity_score, sentiment_avg, repeat_rate, health_concern_tags,
                               rating, review_count, main_ingredients,
                               ({' + '.join(score_parts)}) AS loose_score
                        FROM product
                        WHERE 1=1 {filter_str}
                          AND ({' OR '.join(where_parts)})
                        ORDER BY loose_score DESC,
                                 popularity_score DESC NULLS LAST,
                                 review_count DESC NULLS LAST,
                                 price ASC
                        LIMIT 100
                    """
                    execute_traced(
                        cur,
                        "hybrid_search_loose_fallback",
                        loose_sql,
                        score_params + filter_params_shared + where_params,
                        terms=loose_terms,
                    )
                    loose_cols = [d[0] for d in cur.description]
                    kw_rows = [dict(zip(loose_cols, row)) for row in cur.fetchall()]
                    trace_log(
                        "db_query_rows",
                        label="hybrid_search_loose_fallback",
                        row_count=len(kw_rows),
                    )
                    if kw_rows:
                        print(
                            f"[hybrid_search_pg] loose fallback search matched "
                            f"{len(kw_rows)} rows for query={query!r}, terms={loose_terms}"
                        )

            scores: dict[str, float] = {}
            rows_by_id: dict[str, dict] = {}

            for rank, row in enumerate(vec_rows):
                gid = row["goods_id"]
                scores[gid] = scores.get(gid, 0) + 1 / (k + rank + 1)
                rows_by_id[gid] = row

            for rank, row in enumerate(kw_rows):
                gid = row["goods_id"]
                scores[gid] = scores.get(gid, 0) + 1 / (k + rank + 1)
                rows_by_id[gid] = row

            for rank, row in enumerate(pop_rows):
                gid = row["goods_id"]
                scores[gid] = scores.get(gid, 0) + 1 / (k + rank + 1)
                rows_by_id[gid] = row

            sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
            results = []
            for gid in sorted_ids[:top_k]:
                row = rows_by_id[gid]
                row["_score"] = scores[gid]
                results.append(row)

            trace_log(
                "hybrid_search_result",
                vector_rows=len(vec_rows),
                keyword_rows=len(kw_rows),
                popularity_rows=len(pop_rows),
                result_count=len(results),
            )
            return results

        except Exception as e:
            print(f"[hybrid_search_pg] 오류: {e}")
            return []
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()


# ── 공통 헬퍼 ───────────────────────────────────────────────────────────────────

DOMAIN_INTENT_TO_CATEGORY = {
    "health_disease":      "건강 및 질병",
    "care_management":     "사육 및 관리",
    "nutrition_diet":      "영양 및 식단",
    "behavior_psychology": "행동 및 심리",
    "travel":              "여행 및 이동",
}


HEALTH_CONCERN_MAP = {
    "skin": "피부",
    "joint": "관절",
    "digestion": "소화",
    "weight": "체중",
    "urinary": "요로",
    "eye": "눈물",
    "hairball": "헤어볼",
    "dental": "치아",
    "immunity": "면역",
}


def translate_health_concerns(concerns: list[str] | None) -> list[str]:
    if not concerns:
        return []
    return [HEALTH_CONCERN_MAP.get(c, c) for c in concerns]


def build_pet_context(state: ChatState) -> str:
    p = state.get("pet_profile") or {}
    parts = []
    if p.get("species"):
        parts.append(f"종: {normalize_pet_species(p['species']) or p['species']}")
    if p.get("breed"):   parts.append(f"품종: {p['breed']}")
    if p.get("age"):     parts.append(f"나이: {p['age']}")
    
    concerns = translate_health_concerns(state.get("health_concerns"))
    if concerns:
        parts.append(f"건강관심사: {', '.join(concerns)}")

    if state.get("allergies"):
        parts.append(f"알레르기: {', '.join(state['allergies'])}")
    if state.get("food_preferences"):
        parts.append(f"선호사료타입: {', '.join(state['food_preferences'])}")
    return " / ".join(parts) if parts else "펫 프로필 없음"


def get_user_pets(user_id: str) -> list[dict]:
    """사용자의 전체 펫 목록(ID, 이름) 조회"""
    if not user_id: return []
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        execute_traced(
            cur,
            "get_user_pets",
            "SELECT pet_id, name, species, breed, age_years FROM pet WHERE user_id = %s",
            (user_id,),
            user_id=user_id,
        )
        rows = cur.fetchall()
        trace_log("db_query_rows", label="get_user_pets", row_count=len(rows), user_id=user_id)
        return [{"pet_id": str(r[0]), "name": r[1], "species": r[2], "breed": r[3], "age": f"{r[4]}살"} for r in rows]
    except Exception as e:
        print(f"[get_user_pets] 오류: {e}")
        return []
    finally:
        if cur: cur.close()
        if conn: conn.close()


def get_pet_full_profile(pet_id: str) -> dict:
    """특정 펫의 상세 프로필 및 메타데이터(알러지, 관심사 등) 전체 조회"""
    if not pet_id: return {}
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 1. 기본 정보
        execute_traced(
            cur,
            "get_pet_full_profile_base",
            "SELECT name, species, breed, age_years, gender, weight_kg FROM pet WHERE pet_id = %s",
            (pet_id,),
            pet_id=pet_id,
        )
        base = cur.fetchone()
        if not base: return {}
        
        res = {
            "pet_id": pet_id,
            "pet_profile": {
                "name": base[0],
                "species": base[1],
                "breed": base[2],
                "age": f"{base[3]}살",
                "gender": base[4],
                "weight": float(base[5]) if base[5] else None
            },
            "health_concerns": [],
            "allergies": [],
            "food_preferences": []
        }
        
        # 2. 건강 관심사
        execute_traced(
            cur,
            "get_pet_full_profile_concerns",
            "SELECT concern FROM pet_health_concern WHERE pet_id = %s",
            (pet_id,),
            pet_id=pet_id,
        )
        res["health_concerns"] = [r[0] for r in cur.fetchall()]
        trace_log(
            "db_query_rows",
            label="get_pet_full_profile_concerns",
            row_count=len(res["health_concerns"]),
            pet_id=pet_id,
        )

        # 3. 알러지
        execute_traced(
            cur,
            "get_pet_full_profile_allergies",
            "SELECT ingredient FROM pet_allergy WHERE pet_id = %s",
            (pet_id,),
            pet_id=pet_id,
        )
        res["allergies"] = [r[0] for r in cur.fetchall()]
        trace_log(
            "db_query_rows",
            label="get_pet_full_profile_allergies",
            row_count=len(res["allergies"]),
            pet_id=pet_id,
        )

        # 4. 사료 선호도
        execute_traced(
            cur,
            "get_pet_full_profile_food_preferences",
            "SELECT food_type FROM pet_food_preference WHERE pet_id = %s",
            (pet_id,),
            pet_id=pet_id,
        )
        res["food_preferences"] = [r[0] for r in cur.fetchall()]
        trace_log(
            "db_query_rows",
            label="get_pet_full_profile_food_preferences",
            row_count=len(res["food_preferences"]),
            pet_id=pet_id,
        )
        
        return res
    except Exception as e:
        print(f"[get_pet_full_profile] 오류: {e}")
        return {}
    finally:
        if cur: cur.close()
        if conn: conn.close()
