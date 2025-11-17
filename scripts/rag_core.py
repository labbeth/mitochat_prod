"""Core functions for the MitoChat app"""

import os
import json
import re
import yaml
import faiss
import torch

from typing import Any, Callable, Dict, List, Tuple, Optional
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder

from scripts.utils import _hit_key, load_embedder


# Optional tiktoken encoder for token-budget trimming
try:
    import tiktoken
    _enc = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    _enc = None


# Optional reranker
try:
    # from sentence_transformers import CrossEncoder
    _RERANK_OK = True
except Exception:
    CrossEncoder = None  # type: ignore
    _RERANK_OK = False


# ----------------------------
# Data loading
# ----------------------------
def load_index(index_dir: str):
    index_path = os.path.join(index_dir, "index.faiss")
    ds_path = os.path.join(index_dir, "docstore.jsonl")
    meta_path = os.path.join(index_dir, "meta.json")

    if not os.path.exists(index_path) or not os.path.exists(ds_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing index/docstore/meta under {index_dir}")

    index = faiss.read_index(index_path)
    if not isinstance(index, faiss.IndexIDMap2):
        index = faiss.IndexIDMap2(index)

    docstore = []
    with open(ds_path, "r", encoding="utf-8") as f:
        for line in f:
            docstore.append(json.loads(line))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, docstore, meta


def load_subindex(index_dir: str, name: str) -> Tuple[Optional[faiss.IndexIDMap2], Optional[List[Dict[str, Any]]]]:
    ipath = os.path.join(index_dir, f"index_{name}.faiss")
    dpath = os.path.join(index_dir, f"docstore_{name}.jsonl")
    if not (os.path.exists(ipath) and os.path.exists(dpath)):
        return None, None
    ix = faiss.read_index(ipath)
    if not isinstance(ix, faiss.IndexIDMap2):
        ix = faiss.IndexIDMap2(ix)
    ds = []
    with open(dpath, "r", encoding="utf-8") as f:
        for line in f:
            ds.append(json.loads(line))
    return ix, ds


def load_id_maps(index_dir: str) -> Dict[str, Any]:
    p = os.path.join(index_dir, "id_maps.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"variant_id_to_idx": {}, "rsid_to_idx": {}, "gene_symbol_to_idxs": {}}


# def load_embedder(model_name: str) -> SentenceTransformer:
#     """
#     Load a local SentenceTransformer model.
#     - `model_name` is usually a relative path from the container WORKDIR,
#       e.g. "models/sentence-transformers/all-MiniLM-L6-v2".
#     - We assume models are already present locally (offline).
#     """
#
#     # Enforce offline behavior once here (no proxies, no HF hub calls)
#     for key in [
#         "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
#         "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
#     ]:
#         os.environ.pop(key, None)
#     os.environ["HF_HUB_OFFLINE"] = "1"
#     os.environ["TRANSFORMERS_OFFLINE"] = "1"
#
#     p = Path(model_name)
#     if not p.is_absolute():
#         p = (Path.cwd() / p).resolve()
#
#     if not p.exists():
#         raise RuntimeError(
#             f"Embedding model path does not exist: {p}\n"
#             "Check your config['embedding']['model_name'] and container layout."
#         )
#
#     # Optional: keep the 'modules.json' check if you rely on local SentenceTransformers format
#     modules_file = p / "modules.json"
#     if not modules_file.exists():
#         raise RuntimeError(
#             f"Local SentenceTransformer seems incomplete at: {p}\n"
#             f"Missing file: {modules_file}"
#         )
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = SentenceTransformer(str(p), device=device)
#     # avoid remote model card fetch
#     model._model_card = None
#     return model


def load_rewrite_prompts(lang: str = "fr", base_dir: str | Path = "prompts") -> dict:
    """
    Load rewrite prompts from e.g. <base_dir>/rewrite_fr.yaml.
    `base_dir` will typically come from config["paths"]["prompts_dir"].
    """
    base_dir = Path(base_dir)
    path = base_dir / f"rewrite_{lang}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # minimal validation
    for k in ("system", "fewshots", "output_tags"):
        if k not in data:
            raise ValueError(f"Missing key '{k}' in {path}")
    if "open" not in data["output_tags"] or "close" not in data["output_tags"]:
        raise ValueError(f"Missing output_tags.open/close in {path}")
    return data


def load_router_prompt(lang: str = "en", base_dir: str | Path = "prompts") -> dict:
    """
    Load router prompt from e.g. <base_dir>/router_en.yaml.
    """
    base_dir = Path(base_dir)
    path = base_dir / f"router_{lang}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Router prompt not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    for k in ("system", "user_template", "output_format"):
        if k not in data:
            raise ValueError(f"Missing key '{k}' in router prompt {path}")
    return data


def load_ner(model_name: str = "en_core_web_sm"):
    """
    Tries sciSpaCy first if present, otherwise falls back to spaCy small.
    Install suggestions:
      pip install spacy
      python -m spacy download en_core_web_sm
    For biomedical: pip install scispacy && pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
    """
    try:
        import spacy
        return spacy.load(model_name)
    except Exception:
        try:
            import spacy
            return spacy.load("en_core_web_sm")
        except Exception:
            return None  # app will gracefully fallback


# ----------------------------
# Retrieval utilities
# ----------------------------
def pretty_citation(md: Dict[str, Any]) -> str:
    src = (md or {}).get("source")
    if src == "genereviews":
        sec = md.get("section") or "Section"
        page = md.get("page")
        title = md.get("doc_title") or md.get("doc_id") or "Document"
        cite = f"{title} – {sec}"
        if page:
            cite += f", p.{page}"
        return cite
    if src == "clinvar":
        vid = md.get("variant_id")
        gs = md.get("gene_symbol")
        return f"ClinVar: {vid or 'variant'} in {gs or 'gene'}"
    if src == "mitocarta":
        return f"Mitocarta gene: {md.get('symbol')}"
    # Legacy fallback keys
    if src == "pdf":
        sec = md.get("section") or "Section"
        page = md.get("page")
        title = md.get("doc_title") or md.get("doc_id") or "Document"
        cite = f"{title} – {sec}"
        if page:
            cite += f", p.{page}"
        return cite
    if src == "json_variants":
        return f"Variant {md.get('variant_id')} in {md.get('gene_symbol')}"
    if src == "json_genes":
        return f"Gene {md.get('symbol')}"
    # Fallback to source_file basename
    sf = md.get("source_file")
    return Path(sf).name if sf else "Source"


GUIDANCE_KEYWORDS_DEFAULT = [
    "frequency","surveillance","monitor","monitoring","follow-up","follow up",
    "management","treatment","avoid","contraindicated","recommend","recommended",
    "ekg","ecg","echocardiogram","holter","blood pressure","annually","every 6"
]


def classify_guidance_query(q: str, retr_cfg: Dict[str, Any]) -> bool:
    if not retr_cfg.get("enforce_pdf_for_guidance", True):
        return False
    kws = retr_cfg.get("guidance_keywords") or GUIDANCE_KEYWORDS_DEFAULT
    ql = q.lower()
    return any(k.lower() in ql for k in kws)


def _normalize_boosts(source_boosts: Dict[str, float]) -> Dict[str, float]:
    boosts = dict(source_boosts or {})
    # Map legacy keys -> new keys
    if "pdf" in boosts and "genereviews" not in boosts:
        boosts["genereviews"] = boosts["pdf"]
    if "json_variants" in boosts and "clinvar" not in boosts:
        boosts["clinvar"] = boosts["json_variants"]
    if "json_genes" in boosts and "mitocarta" not in boosts:
        boosts["mitocarta"] = boosts["json_genes"]
    return boosts


def apply_source_boosts(hits: List[Dict[str, Any]], retr_cfg: Dict[str, Any], prefer_genereviews: bool) -> None:
    boosts = _normalize_boosts(retr_cfg.get("source_boosts") or {})
    if prefer_genereviews:
        boosts["genereviews"] = boosts.get("genereviews", 0.0) + 0.30
        boosts["clinvar"] = boosts.get("clinvar", 0.0) - 0.10
        boosts["mitocarta"] = boosts.get("mitocarta", 0.0) - 0.05
    for h in hits:
        base = float(h.get("rerank_score", h.get("score", 0.0)))
        src = (h.get("metadata") or {}).get("source")
        h["_adj_score"] = base + float(boosts.get(src, 0.0))


def lexical_signal_genereviews(query: str, text: str, retr_cfg: Dict[str, Any]) -> float:
    if not retr_cfg.get("hybrid_lexical", True):
        return 0.0
    w = float(retr_cfg.get("lexical_weight_pdf", 0.02))  # keep key name for compat
    tl = text.lower()
    count = 0
    for t in GUIDANCE_KEYWORDS_DEFAULT:
        count += len(re.findall(r"\b" + re.escape(t.lower()) + r"\b", tl))
    return w * count


def add_lexical_boosts_for_genereviews(hits: List[Dict[str, Any]], query: str, retr_cfg: Dict[str, Any]) -> None:
    for h in hits:
        md = h.get("metadata") or {}
        if md.get("source") == "genereviews":
            h["_adj_score"] = h.get("_adj_score", float(h.get("rerank_score", h.get("score", 0.0)))) \
                              + lexical_signal_genereviews(query, h["text"], retr_cfg)


def _cap_values(retr_cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    caps = retr_cfg.get("per_source_cap") or {}
    gr_min = int(caps.get("genereviews_min", caps.get("pdf_min", 3)))
    clinvar_max = int(caps.get("clinvar_max", caps.get("json_variants_max", 2)))
    mitocarta_max = int(caps.get("mitocarta_max", caps.get("json_genes_max", 2)))
    return gr_min, clinvar_max, mitocarta_max


def rebalance_by_source(hits: List[Dict[str, Any]], retr_cfg: Dict[str, Any], final_k: int) -> List[Dict[str, Any]]:
    gr_min, clinvar_max, mitocarta_max = _cap_values(retr_cfg)
    hits_sorted = sorted(
        hits, key=lambda x: x.get("_adj_score", x.get("rerank_score", x.get("score", 0.0))), reverse=True
    )
    gr = [h for h in hits_sorted if (h.get("metadata") or {}).get("source") == "genereviews"]
    cl = [h for h in hits_sorted if (h.get("metadata") or {}).get("source") == "clinvar"]
    mt = [h for h in hits_sorted if (h.get("metadata") or {}).get("source") == "mitocarta"]
    other = [h for h in hits_sorted if (h.get("metadata") or {}).get("source") not in {"genereviews","clinvar","mitocarta"}]

    result, used = [], set()
    # 1) ensure minimum GeneReviews
    for h in gr[:min(gr_min, len(gr))]:
        if h["idx"] not in used:
            result.append(h); used.add(h["idx"])
    # 2) limited JSONs
    def take_cap(pool, cap):
        src = (pool[0].get("metadata") or {}).get("source") if pool else None
        added = 0
        for h in pool:
            if added >= cap:
                break
            if h["idx"] in used:
                continue
            result.append(h); used.add(h["idx"]); added += 1
    take_cap(cl, clinvar_max)
    take_cap(mt, mitocarta_max)
    # 3) fill remaining
    def take(pool):
        for h in pool:
            if len(result) >= final_k:
                break
            if h["idx"] in used:
                continue
            result.append(h); used.add(h["idx"])
    if len(result) < final_k: take(gr)
    if len(result) < final_k: take(other)
    if len(result) < final_k: take(cl)
    if len(result) < final_k: take(mt)
    return result[:final_k]


# Extract identifiers inside sentences
VARIANT_ID_RE = re.compile(r"\b(?:(?:chr)?([0-9]{1,2}|X|Y|MT)):(\d+):([ACGT]):([ACGT])\b", re.I)
RSID_RE = re.compile(r"\brs\d+\b", re.I)


def detect_identifiers(query: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for m in VARIANT_ID_RE.finditer(query):
        chrom = m.group(1).upper(); pos = m.group(2)
        ref = m.group(3).upper(); alt = m.group(4).upper()
        out.append(("variant_id", f"{chrom}:{pos}:{ref}:{alt}"))
    for m in RSID_RE.finditer(query):
        out.append(("rsid", m.group(0).lower()))
    m = re.search(r"\bgene\s+([A-Z0-9\-]{2,10})\b", query)
    if m:
        out.append(("gene_symbol", m.group(1).upper()))
    elif query.strip().isupper() and 2 <= len(query.strip()) <= 10:
        out.append(("gene_symbol", query.strip().upper()))
    return out


def structured_lookup_first(query: str, id_maps: Dict[str, Any], docstore: List[Dict[str, Any]], limit_per_key: int = 3) -> List[Dict[str, Any]]:
    idents = detect_identifiers(query)
    if not idents:
        return []
    hits: List[Dict[str, Any]] = []
    seen_idx: set[int] = set()
    for which, key in idents:
        if which == "variant_id":
            idxs = id_maps.get("variant_id_to_idx", {}).get(key, [])
        elif which == "rsid":
            m = id_maps.get("rsid_to_idx", {})
            idxs = m.get(key, []) + m.get(key.replace("rs", ""), [])
        elif which == "gene_symbol":
            idxs = id_maps.get("gene_symbol_to_idxs", {}).get(key, [])
        else:
            idxs = []
        for i in idxs[:limit_per_key]:
            if i in seen_idx:
                continue
            rec = docstore[i]
            hits.append({
                "score": 1.0,
                "idx": int(i),
                "text": rec["text"],
                "metadata": rec.get("metadata", {}),
                "_adj_score": 999.0,
            })
            seen_idx.add(i)
    return hits


def search(
    query: str,
    index: faiss.IndexIDMap2,
    docstore: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    top_k: int,
    enable_reranker: bool,
    rerank_top_k: int,
    reranker_model: str,
    retr_cfg: Optional[Dict[str, Any]] = None,
    id_maps: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    retr_cfg = retr_cfg or {}

    # 0) Structured fast-path
    struct_hits = structured_lookup_first(query, id_maps or {}, docstore, limit_per_key=3)

    # 1) Dense vector search
    qemb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(qemb, top_k)
    I = I[0]; D = D[0]

    dense_hits = []
    for score, idx in zip(D, I):
        if idx == -1: continue
        rec = docstore[idx]
        dense_hits.append({"score": float(score), "idx": int(idx), "text": rec["text"], "metadata": rec.get("metadata", {})})

    # 2) Optional reranker
    if enable_reranker and _RERANK_OK and rerank_top_k and dense_hits:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ce = CrossEncoder(reranker_model, device=device)
            rr = ce.predict([(query, h["text"]) for h in dense_hits], batch_size=128, show_progress_bar=False)
            for i, s in enumerate(rr): dense_hits[i]["rerank_score"] = float(s)
        except Exception:
            for h in dense_hits: h["rerank_score"] = h["score"]
    else:
        for h in dense_hits: h["rerank_score"] = h["score"]

    # 3) Merge struct + dense, dedupe
    hits = []
    seen = set()
    for h in struct_hits + dense_hits[: (rerank_top_k or len(dense_hits))]:
        k = _hit_key(h)
        if k in seen: continue
        hits.append(h); seen.add(k)

    # 4) Boosts + lexical
    prefer_gr = classify_guidance_query(query, retr_cfg)
    apply_source_boosts(hits, retr_cfg, prefer_gr)
    add_lexical_boosts_for_genereviews(hits, query, retr_cfg)

    # 5) Per-source quotas
    final_k = int(retr_cfg.get("final_k", rerank_top_k or len(hits)))
    hits = rebalance_by_source(hits, retr_cfg, final_k=final_k)
    return hits


def search_multi_stage(
    query: str,
    embedder: SentenceTransformer,
    per_source_indices: Dict[str, Tuple[Optional[faiss.IndexIDMap2], Optional[List[Dict[str, Any]]]]],
    per_source_k: Dict[str, int],
    final_k: int,
    enable_reranker: bool,
    reranker_model: str,
    retr_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    qemb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    def search_ix(ix, ds, k):
        if not ix or not ds or k <= 0: return []
        D, I = ix.search(qemb, k); D, I = D[0], I[0]
        out = []
        for score, idx in zip(D, I):
            if idx == -1: continue
            rec = ds[idx]
            out.append({"score": float(score), "idx": int(idx), "text": rec["text"], "metadata": rec.get("metadata", {})})
        return out

    # normalize keys
    key_map = {"genereviews": "genereviews", "pdf": "genereviews", "clinvar": "clinvar", "json_variants": "clinvar", "mitocarta": "mitocarta", "json_genes": "mitocarta"}
    hits = []
    for k_raw, k in key_map.items():
        if k_raw in per_source_k:
            ix, ds = per_source_indices.get(k, (None, None))
            hits.extend(search_ix(ix, ds, int(per_source_k[k_raw])))

    if not hits: return []

    # Rerank
    if enable_reranker and _RERANK_OK:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ce = CrossEncoder(reranker_model, device=device)
            rr = ce.predict([(query, h["text"]) for h in hits], batch_size=128, show_progress_bar=False)
            for i, s in enumerate(rr): hits[i]["rerank_score"] = float(s)
        except Exception:
            for h in hits: h["rerank_score"] = h["score"]
    else:
        for h in hits: h["rerank_score"] = h["score"]

    # Boosts + lexical + quotas
    prefer_gr = classify_guidance_query(query, retr_cfg)
    apply_source_boosts(hits, retr_cfg, prefer_gr)
    add_lexical_boosts_for_genereviews(hits, query, retr_cfg)
    hits = rebalance_by_source(hits, retr_cfg, final_k=final_k)
    return hits


def diagnostic_search_multi_stage(
    query: str,
    embedder: SentenceTransformer,
    per_source_indices: Dict[str, Tuple[Optional[faiss.IndexIDMap2], Optional[List[Dict[str, Any]]]]],
    per_source_k: Dict[str, int],
    final_k: int,
    enable_reranker: bool,
    reranker_model: str,
    retr_cfg: Dict[str, Any],
    id_maps: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Same strategy as the app's search, but returns full diagnostics:
      - structured_hits  (ID fast-path)
      - by_source        (raw dense top-k per source *before* boosts/quotas)
      - final            (after rerank + boosts + quotas, i.e., what the app actually uses)
    """
    # 0) structured fast-path
    structured_hits = structured_lookup_first(query, id_maps or {}, per_source_indices.get("genereviews", (None, None))[1] \
                                             or per_source_indices.get("clinvar", (None, None))[1] \
                                             or per_source_indices.get("mitocarta", (None, None))[1] \
                                             or [], limit_per_key=3)

    # 1) per-source dense
    qemb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    def search_ix(ix, ds, k):
        if not ix or not ds or not k or k <= 0:
            return []
        D, I = ix.search(qemb, k); D, I = D[0], I[0]
        out = []
        for score, idx in zip(D, I):
            if idx == -1:
                continue
            rec = ds[idx]
            out.append({
                "score": float(score),
                "idx": int(idx),
                "text": rec.get("text", ""),
                "metadata": rec.get("metadata", {}) or {}
            })
        return out

    # normalize keys same way as app
    key_map = {"genereviews":"genereviews","pdf":"genereviews",
               "clinvar":"clinvar","json_variants":"clinvar",
               "mitocarta":"mitocarta","json_genes":"mitocarta"}

    by_source: Dict[str, List[Dict[str, Any]]] = {"genereviews":[], "clinvar":[], "mitocarta":[]}
    for k_raw, k_norm in key_map.items():
        k = per_source_k.get(k_raw)
        if k is None:  # ignore alias if not present
            continue
        ix, ds = per_source_indices.get(k_norm, (None, None))
        by_source[k_norm].extend(search_ix(ix, ds, int(k)))

    # 2) merge into one list
    def _hit_key(h):
        md = h.get("metadata") or {}
        src = md.get("source")
        return (src, md.get("variant_id") or md.get("symbol") or md.get("doc_id"),
                md.get("page"), md.get("section"))

    hits = []
    seen = set()
    for h in structured_hits + by_source["genereviews"] + by_source["clinvar"] + by_source["mitocarta"]:
        k = _hit_key(h)
        if k in seen:
            continue
        hits.append(h); seen.add(k)

    # 3) rerank
    try:
        from sentence_transformers import CrossEncoder
        if enable_reranker and final_k and hits:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ce = CrossEncoder(reranker_model, device=device)
            rr = ce.predict([(query, h["text"]) for h in hits], batch_size=128, show_progress_bar=False)
            for i, s in enumerate(rr): hits[i]["rerank_score"] = float(s)
        else:
            for h in hits: h["rerank_score"] = h["score"]
    except Exception:
        for h in hits: h["rerank_score"] = h.get("score", 0.0)

    # 4) boosts + lexical + quotas (exactly like app)
    prefer_gr = classify_guidance_query(query, retr_cfg)
    apply_source_boosts(hits, retr_cfg, prefer_gr)
    add_lexical_boosts_for_genereviews(hits, query, retr_cfg)
    final_hits = rebalance_by_source(hits, retr_cfg, final_k=final_k)

    return {
        "structured_hits": structured_hits,  # list of dicts
        "by_source": by_source,              # dict: source -> list of dicts
        "final": final_hits                  # list of dicts
    }


# ------------------------------
# LLM prompts builders & helpers
# ------------------------------
def rewrite_query_llm_from_yaml(
    chat_history: list[dict],
    user_input: str,
    llm_generate: Callable[[list[dict], int, float], str],
    prompts: dict,
) -> tuple[str, dict]:
    sys_prompt = prompts["system"].strip()
    fewshots = prompts["fewshots"].strip()
    max_turns = int(prompts.get("max_turns", 4))
    open_tag = prompts["output_tags"]["open"]
    close_tag = prompts["output_tags"]["close"]
    tag_re = re.compile(re.escape(open_tag) + r"(.*?)" + re.escape(close_tag), re.DOTALL)

    recent = chat_history[-max_turns*2:] if chat_history else []
    lines = []
    for m in recent:
        if m.get("role") in ("user", "assistant"):
            role = "Utilisateur" if m["role"] == "user" else "Assistant"
            lines.append(f"{role}: {m.get('content','').strip()}")
    history_text = "\n".join(lines)

    user_prompt = f"""{fewshots}

Recent history:
{history_text}

Current question:
Utilisateur: {user_input}

Rewrite the current question so that it is self-contained while strictly following the constraints.
Remember: no invention. Keep the original language. Preserve genes/variants/acronyms exactly as they are.
Output (strictly): {open_tag}…{close_tag}
""".strip()

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    debug = {"messages": messages, "raw": None, "error": None}

    try:
        text = llm_generate(messages, max_tokens=120, temperature=0.0) or ""
        debug["raw"] = text

        m = tag_re.search(text.strip())
        if m:
            rewritten = m.group(1).strip()
            if rewritten:
                return rewritten, debug

        fallback = text.strip()
        if fallback and fallback != user_input:
            return fallback, debug
        return user_input, debug

    except Exception as e:
        debug["error"] = str(e)
        return user_input, debug


def render_router_user(router_cfg: dict, query: str, hits: List[Dict[str, Any]]) -> str:
    # Format passages as: [i] • score • source • text
    parts = []
    for i, h in enumerate(hits, 1):
        score = h.get("rerank_score", h.get("score", 0.0))
        src = pretty_citation(h.get("metadata", {}) or {})
        txt = (h.get("text") or "").strip()
        # keep each snippet short for routing
        snippet = txt[:800]
        parts.append(f"[{i}] • {score:.3f} • {src}\n{snippet}")
    passages = "\n\n".join(parts) if parts else "(none)"
    return router_cfg["user_template"].format(query=query, passages=passages).strip()


def call_router_llm(
    router_prompt: dict,
    query: str,
    hits: List[Dict[str, Any]],
    llm_generate: Callable[[list[dict], int, float], str],
) -> dict:
    """
    Decide RAG / LLM / HYBRID using the same `llm_generate` hook as the rewriter.
    """
    system = router_prompt["system"].strip()
    user = render_router_user(router_prompt, query, hits)

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    raw = None
    try:
        raw = llm_generate(messages, max_tokens=128, temperature=0.0) or ""
    except Exception:
        raw = None

    # Parse a single-line JSON object anywhere in the text
    out = {"mode": "RAG" if hits else "LLM", "reason": "fallback", "use_ids": []}
    if not raw:
        return out

    m = re.search(r"\{.*\}", raw.replace("\n", " ").strip())
    if not m:
        return out
    try:
        j = json.loads(m.group(0))
        mode = str(j.get("mode", "")).upper()
        if mode not in {"RAG", "LLM", "HYBRID"}:
            return out
        use_ids = j.get("use_ids") or []
        use_ids = [int(i) for i in use_ids if isinstance(i, int) and 1 <= i <= len(hits)]
        return {"mode": mode, "reason": j.get("reason", ""), "use_ids": use_ids}
    except Exception:
        return out


# ----------------------------
# Misc helpers
# ----------------------------
CITE_RE = re.compile(r"\s*\[\d+\]")  # matches ' [1]', ' [23]' anywhere
REFS_RE = re.compile(r"(?is)\n?\s*references\s*:.*$")  # strip a tailing 'References:' section

def strip_citations(text: str) -> str:
    if not text:
        return text
    # Remove any [number] citations
    text = CITE_RE.sub("", text)
    # Remove trailing 'References:' block if the model added one
    text = REFS_RE.sub("", text)
    return text.strip()


def trim_to_token_budget(text: str, max_tokens: Optional[int]) -> str:
    """
    Trims `text` so that it roughly fits into `max_tokens` tokens.
    Uses tiktoken if available, otherwise a crude char-based heuristic.
    """
    if not max_tokens:
        return text
    if not _enc:
        # rough fallback: ~4 chars per token
        return text[: max_tokens * 4]
    toks = _enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return _enc.decode(toks[:max_tokens])



def build_messages_llm_only(query: str, system_prompt: str, chat_history: List[Dict[str, str]], history_turns: int = 3):
    conv_pairs = []
    for m in chat_history[-history_turns*2:]:
        if m["role"] in ("user", "assistant"):
            role = "Utilisateur" if m["role"] == "user" else "Assistant"
            conv_pairs.append(f"{role}: {m['content']}")
    conversation = trim_to_token_budget("\n".join(conv_pairs), max_tokens=256)

    user_prompt = (
        "Answer the user's question from your general knowledge. "
        "Do NOT use any external context.\n"
        "Do NOT include citations, bracketed numbers like [1], [2], or a References section.\n"
        "If the user greets or asks conversationally, respond politely.\n\n"
        f"Conversation (recent):\n{conversation}\n\n"
        f"Question: {query}\nAnswer:"
    )
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def build_messages_hybrid(
    query: str,
    hits: List[Dict[str, Any]],
    system_prompt: str,
    num_ctx: int,
    chat_history: List[Dict[str, str]],
    use_ids: Optional[List[int]] = None,
) -> List[Dict[str, str]]:
    # keep only selected ids if provided
    use = hits
    if use_ids:
        use = [h for i, h in enumerate(hits, 1) if i in use_ids]

    blocks = []
    for i, h in enumerate(use, 1):
        src = pretty_citation(h["metadata"])
        blocks.append(f"[{i}] {src}\n{h['text']}")
    ctx = "\n\n".join(blocks)
    ctx = trim_to_token_budget(ctx, int(num_ctx * 0.5) if num_ctx else None)

    conv_pairs = []
    for m in chat_history[-6:]:
        if m["role"] in ("user", "assistant"):
            role = "Utilisateur" if m["role"] == "user" else "Assistant"
            conv_pairs.append(f"{role}: {m['content']}")
    conversation = trim_to_token_budget("\n".join(conv_pairs), max_tokens=256)

    user_prompt = (
        "Use the documentary context IF it is relevant; otherwise answer from your general knowledge. "
        "Cite sources with bracketed numbers [1], [2] ONLY if you used the context. "
        "Never fabricate citations.\n\n"
        f"Conversation (recent):\n{conversation}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\nAnswer:"
    )
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages



def build_messages_with_history(
    query: str,
    hits: List[Dict[str, Any]],
    system_prompt: str,
    num_ctx: int,
    chat_history: List[Dict[str, str]],
    history_turns: int = 3,
) -> List[Dict[str, str]]:
    conv_pairs = []
    for m in chat_history[-history_turns*2:]:
        if m["role"] in ("user", "assistant"):
            role = "Utilisateur" if m["role"] == "user" else "Assistant"
            conv_pairs.append(f"{role}: {m['content']}")
    conversation = "\n".join(conv_pairs)
    conversation = trim_to_token_budget(conversation, max_tokens=256)

    blocks = []
    for i, h in enumerate(hits, 1):
        src = pretty_citation(h["metadata"])
        blocks.append(f"[{i}] {src}\n{h['text']}")
    ctx = "\n\n".join(blocks)
    ctx_budget = int(num_ctx * 0.55) if num_ctx else None
    ctx = trim_to_token_budget(ctx, ctx_budget)

    user_prompt = (
        "Use only the documentary context to answer. "
        "If the information is not present, say that you do not know.\n"
        "Cite the sources with numbers in square brackets [1], [2], etc.\n\n"
        f"Conversation context (recent):\n{conversation}\n\n"
        f"Documentary context:\n{ctx}\n\n"
        f"Question: {query}\nAnswer:"
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages
