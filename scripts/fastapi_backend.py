# fastapi_backend.py
import os
import requests
import json
from typing import List, Dict, Optional, Any, Tuple
from fastapi import FastAPI
from pydantic import BaseModel

from scripts.utils import read_config, _hit_key, load_embedder, resolve_project_path
from scripts.translate_prod import get_translator, translate_text
from scripts.rag_core import (
    load_index,
    load_subindex,
    load_id_maps,
    load_rewrite_prompts,
    load_router_prompt,
    search,
    search_multi_stage,
    detect_identifiers,
    structured_lookup_first,
    build_messages_llm_only,
    build_messages_hybrid,
    build_messages_with_history,
    call_router_llm,
    rewrite_query_llm_from_yaml,
)

from time import time
from fastapi import Request
from fastapi.staticfiles import StaticFiles


USE_STUB_LLM = False  # running on Windows without vLLM -> True; set False on GPU server


# ============================================================
# HELPERS
# ============================================================
def resolve_full_chunk_for_api(hit: Dict[str, Any], docstore: Any) -> Tuple[str, Any]:
    """
    Returns (kind, content) where kind in {"text","json"}.
    Uses metadata keys to fetch full section/record from docstore if possible.
    Falls back to hit["text"].
    """
    md = (hit.get("metadata") or {})

    # 0) if already present
    if md.get("section_text"):
        return "text", str(md["section_text"])
    if md.get("json_entry"):
        return "json", md["json_entry"]
    if md.get("raw_json"):
        try:
            return "json", json.loads(md["raw_json"])
        except Exception:
            return "text", str(md["raw_json"])

    def _ds_get(key, default=None):
        if isinstance(docstore, dict):
            return docstore.get(key, default)
        return default

    doc_id = md.get("doc_id") or md.get("document_id") or md.get("source_id")
    if doc_id is not None:
        rec_id = md.get("record_id") or md.get("entry_id") or md.get("key")
        if rec_id is not None:
            val = _ds_get(("json", doc_id, rec_id)) or _ds_get(f"{doc_id}:{rec_id}")
            if val is not None:
                if isinstance(val, (dict, list)):
                    return "json", val
                try:
                    return "json", json.loads(val)
                except Exception:
                    return "text", str(val)

        section_id = md.get("section_id") or md.get("sec_id")
        if section_id is not None:
            val = _ds_get(("pdf_section", doc_id, section_id)) or _ds_get(f"{doc_id}:{section_id}")
            if val is not None:
                if isinstance(val, dict) and "text" in val:
                    return "text", str(val["text"])
                return "text", str(val)

        page = md.get("page") or md.get("page_index")
        if page is not None:
            try:
                p = int(page)
            except Exception:
                p = page
            val = _ds_get(("pdf_page", doc_id, p)) or _ds_get(f"{doc_id}:p{p}")
            if val is not None:
                if isinstance(val, dict) and "text" in val:
                    return "text", str(val["text"])
                return "text", str(val)

    return "text", str(hit.get("text", ""))


# ============================================================
# 1. CONFIG + INITIALIZATION
# ============================================================

cfg = read_config()
gen_cfg = cfg.get("generation", {}) or {}
retr_cfg = cfg.get("retrieval", {}) or {}
trans_cfg = cfg.get("translation", {}) or {}
paths = cfg.get("paths", {}) or {}

# index_dir = paths["index_dir"]          # e.g. "data/index"
index_dir = resolve_project_path(cfg["paths"]["index_dir"])
os.makedirs(index_dir, exist_ok=True)
prompts_dir = paths.get("prompts_dir", "prompts")
embedding_model_name = cfg["embedding"]["model_name"]

# --- Translation (FR <-> EN) ---
translator = get_translator(
    device=trans_cfg.get("device"),
    max_new_tokens=int(trans_cfg.get("max_new_tokens", 512)),
    cache_bust="v1",
)

# --- FAISS indices and docstore ---
index, docstore, meta = load_index(index_dir)

gr_ix, gr_ds = load_subindex(index_dir, "genereviews")
cl_ix, cl_ds = load_subindex(index_dir, "clinvar")
mt_ix, mt_ds = load_subindex(index_dir, "mitocarta")
per_source_indices = {
    "genereviews": (gr_ix, gr_ds),
    "clinvar": (cl_ix, cl_ds),
    "mitocarta": (mt_ix, mt_ds),
}

id_maps = load_id_maps(index_dir)

# --- Embedder ---
# embedder = SentenceTransformer(embedding_model_name)
embedder = load_embedder(embedding_model_name)

# --- Prompts ---
rewrite_prompts_en = load_rewrite_prompts(lang="en", base_dir=prompts_dir)
router_prompt_en = load_router_prompt(lang="en", base_dir=prompts_dir)

# ============================================================
# 2. DEFINE THE vLLM CALL FUNCTION
# ============================================================
# For target run (does not work in Windows)
def generate_with_vllm(
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Optional[str]:
    """
    Call vLLM OpenAI-compatible server.
    """
    base_url = gen_cfg.get("vllm_base_url") or os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    model = gen_cfg.get("vllm_model") or os.getenv("VLLM_MODEL", "qwen2.5-7b-instruct")
    api_key = gen_cfg.get("vllm_api_key") or os.getenv("VLLM_API_KEY", "EMPTY")

    temperature = float(temperature if temperature is not None else gen_cfg.get("temperature", 0.2))
    max_tokens = int(max_tokens if max_tokens is not None else gen_cfg.get("max_tokens", 400))

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        json=payload,
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


# Stub version for debug
def generate_with_vllm_stub(
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Optional[str]:
    # TEMP STUB: avoid calling real vLLM
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    return f"[LLM STUB] Last user message (EN): {last_user[:200]}"



# ============================================================
# 3. DEFINE FASTAPI APP AND ENDPOINTS
# ============================================================

app = FastAPI(title="RAG Backend API", version="1.0")

app.mount("/assets", StaticFiles(directory="assets"), name="assets")

class ChatTurn(BaseModel):
    role: str
    content: str

class RagRequest(BaseModel):
    chat_history: list[ChatTurn]
    query_fr: str

class RagResponse(BaseModel):
    answer_fr: str
    answer_en: str
    mode: str
    hits: list[dict]
    router: dict | None = None
    rewrite_en: str

@app.post("/rag", response_model=RagResponse)
def rag_endpoint(req: RagRequest):
    print(f"[/rag] ← query_fr='{req.query_fr[:120]}' | hist={len(req.chat_history)}")

    # ---------- 1) Prepare inputs ----------
    # FR → EN
    try:
        q_en = translate_text(req.query_fr, "fr", "en", translator)
    except Exception:
        q_en = req.query_fr
    print(q_en)

    history = [t.dict() for t in req.chat_history]  # [{"role": "...", "content": "..."}]

    # FR -> EN already done: q_en

    if USE_STUB_LLM:
        # 1) BYPASS REWRITE: use the translated user question directly
        standalone_q = q_en
    else:
        # real rewrite with vLLM (or your actual llm_generate)
        standalone_q, rw_debug = rewrite_query_llm_from_yaml(
            chat_history=history,
            user_input=q_en,
            llm_generate=generate_with_vllm,
            prompts=rewrite_prompts_en,
        )
        # safety: if something still smells like the stub output, fall back
        if isinstance(standalone_q, str) and standalone_q.startswith("[LLM STUB]"):
            standalone_q = q_en

    # # Rewrite EN query (using vLLM through rag_core)
    # standalone_q, rw_debug = rewrite_query_llm_from_yaml(
    #     chat_history=history,
    #     user_input=q_en,
    #     llm_generate=generate_with_vllm,
    #     prompts=rewrite_prompts_en,
    # )

    # ---------- 2) Retrieval (structured + dense) ----------
    multi_stage = bool(retr_cfg.get("multi_stage", False))
    enable_reranker = bool(retr_cfg.get("enable_reranker", True))
    per_source_k = retr_cfg.get("per_source_k", {"genereviews": 40, "clinvar": 6, "mitocarta": 6})
    top_k = int(retr_cfg.get("top_k", 20))
    rerank_top_k = int(retr_cfg.get("rerank_top_k", 8))
    reranker_model = retr_cfg.get("reranker_model", "BAAI/bge-reranker-base")

    # Structured lookup first (ClinVar IDs etc.)
    struct_hits = structured_lookup_first(
        standalone_q,
        id_maps=id_maps,
        docstore=docstore,
        limit_per_key=3,
    )

    # Dense retrieval
    have_sub = any(ix and ds for ix, ds in per_source_indices.values())
    if multi_stage and have_sub:
        dense_hits = search_multi_stage(
            standalone_q,
            embedder=embedder,
            per_source_indices=per_source_indices,
            per_source_k=per_source_k,
            final_k=rerank_top_k,
            enable_reranker=enable_reranker,
            reranker_model=reranker_model,
            retr_cfg={**retr_cfg, "final_k": rerank_top_k, "rerank_top_k": rerank_top_k},
        )
    else:
        dense_hits = search(
            standalone_q,
            index=index,
            docstore=docstore,
            embedder=embedder,
            top_k=top_k,
            enable_reranker=enable_reranker,
            rerank_top_k=rerank_top_k,
            reranker_model=reranker_model,
            retr_cfg={**retr_cfg, "rerank_top_k": rerank_top_k},
            id_maps=id_maps,
        )

    # Merge + dedupe (same as your Streamlit code)
    hits, seen = [], set()
    for h in struct_hits + dense_hits:
        k = _hit_key(h)   # this helper should also be in rag_core
        if k in seen:
            continue
        hits.append(h)
        seen.add(k)

    if not hits:
        # no context → let’s answer something graceful
        answer_en = "I couldn't find relevant passages. Try to rephrase your question."
        try:
            answer_fr = translate_text(answer_en, "en", "fr", translator)
        except Exception:
            answer_fr = answer_en
        return RagResponse(
            answer_fr=answer_fr,
            answer_en=answer_en,
            mode="NO_HITS",
            hits=[],
            router=None,
            rewrite_en=standalone_q,
        )

    for h in hits:
        kind, content = resolve_full_chunk_for_api(h, docstore)
        md = h.setdefault("metadata", {})
        if kind == "text" and content:
            md["section_text"] = content
        elif kind == "json":
            md["json_entry"] = content


    # ---------- 3) Router decision ----------
    router = call_router_llm(
        router_prompt=router_prompt_en,
        query=standalone_q,
        hits=hits,
        llm_generate=generate_with_vllm,  # vLLM callback
    )

    mode = router.get("mode", "RAG" if hits else "LLM")
    use_ids = router.get("use_ids", [])

    # Optional guardrails: force RAG when we have identifiers / structured hits
    has_identifier = bool(detect_identifiers(standalone_q)) or bool(detect_identifiers(req.query_fr))
    has_struct = bool(struct_hits)
    if has_identifier or has_struct:
        mode = "RAG"

    # ---------- 4) Build messages based on mode ----------
    system_prompt = gen_cfg.get("system_prompt", "")
    num_ctx = int(gen_cfg.get("num_ctx", 4096))

    if mode == "LLM":
        messages = build_messages_llm_only(
            query=standalone_q,
            system_prompt=system_prompt,
            chat_history=history,
        )
    elif mode == "HYBRID":
        messages = build_messages_hybrid(
            query=standalone_q,
            hits=hits,
            system_prompt=system_prompt,
            num_ctx=num_ctx,
            chat_history=history,
            use_ids=use_ids,
        )
    else:  # "RAG"
        # Optionally restrict to selected docs
        if use_ids:
            selected = [h for i, h in enumerate(hits, 1) if i in use_ids]
            if selected:
                messages = build_messages_with_history(
                    query=standalone_q,
                    hits=selected,
                    system_prompt=system_prompt,
                    num_ctx=num_ctx,
                    chat_history=history,
                )
            else:
                messages = build_messages_with_history(
                    query=standalone_q,
                    hits=hits,
                    system_prompt=system_prompt,
                    num_ctx=num_ctx,
                    chat_history=history,
                )
        else:
            messages = build_messages_with_history(
                query=standalone_q,
                hits=hits,
                system_prompt=system_prompt,
                num_ctx=num_ctx,
                chat_history=history,
            )

    # --- 6. Generate EN answer ---
    answer_en = generate_with_vllm(messages)

    # --- 7. Translate EN→FR ---
    answer_fr = translate_text(answer_en, "en", "fr", translator)

    print(f"[/rag] → mode={router.get('mode')} | hits={len(hits)} | rewrite='{standalone_q[:120]}'")

    return RagResponse(
        answer_fr=answer_fr,
        answer_en=answer_en,
        mode=router.get("mode", "RAG"),
        hits=hits,
        router=router,
        rewrite_en=standalone_q,
    )


# TEST
@app.get("/health")
def health():
    return {"status": "ok"}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time()
    print(f"[REQ] {request.method} {request.url.path}")
    resp = await call_next(request)
    dur = (time() - start) * 1000
    print(f"[RES] {request.method} {request.url.path} -> {resp.status_code} ({dur:.1f} ms)")
    return resp

@app.get("/whoami")
def whoami():
    return {
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "instance": os.getenv("BACKEND_INSTANCE_ID", "default"),
    }


@app.post("/rag_debug")
def rag_debug(req: RagRequest):
    q_fr = req.query_fr
    try:
        q_en = translate_text(q_fr, "fr", "en", translator)
    except Exception:
        q_en = q_fr

    # choose query fed to retrieval (match your testing mode)
    standalone_q = q_en  # or run rewrite and show both q_en and standalone_q

    # collect config knobs actually used
    cfg_used = {
        "multi_stage": bool(retr_cfg.get("multi_stage", False)),
        "enable_reranker": bool(retr_cfg.get("enable_reranker", True)),
        "per_source_k": retr_cfg.get("per_source_k", {}),
        "top_k": int(retr_cfg.get("top_k", 20)),
        "rerank_top_k": int(retr_cfg.get("rerank_top_k", 8)),
        "final_k": int(retr_cfg.get("final_k", retr_cfg.get("rerank_top_k", 8))),
        "reranker_model": retr_cfg.get("reranker_model"),
    }

    # structured
    struct_hits = structured_lookup_first(standalone_q, id_maps=id_maps, docstore=docstore, limit_per_key=3)

    # dense (multi-stage)
    have_sub = any(ix and ds for ix, ds in per_source_indices.values())
    if cfg_used["multi_stage"] and have_sub:
        dense_hits = search_multi_stage(
            standalone_q, embedder=embedder,
            per_source_indices=per_source_indices,
            per_source_k=cfg_used["per_source_k"],
            final_k=cfg_used["final_k"],
            enable_reranker=cfg_used["enable_reranker"],
            reranker_model=cfg_used["reranker_model"],
            retr_cfg={**retr_cfg, "final_k": cfg_used["final_k"], "rerank_top_k": cfg_used["rerank_top_k"]},
        )
    else:
        dense_hits = search(
            standalone_q, index=index, docstore=docstore, embedder=embedder,
            top_k=cfg_used["top_k"],
            enable_reranker=cfg_used["enable_reranker"],
            rerank_top_k=cfg_used["rerank_top_k"],
            reranker_model=cfg_used["reranker_model"],
            retr_cfg={**retr_cfg, "rerank_top_k": cfg_used["rerank_top_k"]},
            id_maps=id_maps,
        )

    # merge/dedupe like app
    hits, seen = [], set()
    for h in struct_hits + dense_hits:
        k = _hit_key(h)
        if k in seen: continue
        hits.append(h); seen.add(k)

    # return rich debug
    # show top 12 with (score, rerank_score, _adj_score, source, ids)
    def _brief(h):
        md = h.get("metadata") or {}
        return {
            "score": h.get("score"),
            "rerank_score": h.get("rerank_score"),
            "_adj_score": h.get("_adj_score"),
            "source": md.get("source"),
            "doc_id": md.get("doc_id") or md.get("json_key") or md.get("symbol") or md.get("variant_id"),
            "page": md.get("page"),
        }

    return {
        "q_fr": q_fr,
        "q_en": q_en,
        "standalone_q": standalone_q,
        "_RERANK_OK": True,  # flip to an actual global flag if you expose it
        "cfg_used": cfg_used,
        "struct_hits_count": len(struct_hits),
        "dense_hits_count": len(dense_hits),
        "final_hits_count": len(hits),
        "final_hits_preview": [_brief(h) for h in hits[:12]],
    }
