from __future__ import annotations
import os, re, html, json, unicodedata, io, base64
from typing import Any, Dict, List, Tuple
from pathlib import Path
import streamlit as st

# ---------- small text utils ----------
def pretty_citation(md: Dict[str, Any]) -> str:
    src = (md or {}).get("source")
    if src == "genereviews":
        title = md.get("doc_title") or md.get("doc_id") or "GeneReviews"
        sec = md.get("section")
        page = md.get("page")
        s = f"{title}"
        if sec: s += f" — {sec}"
        if page is not None: s += f", p.{page}"
        return s
    if src == "clinvar":
        vid = md.get("variant_id") or md.get("json_key") or "variant"
        gs = md.get("gene_symbol") or "gene"
        return f"ClinVar — {vid} ({gs})"
    if src == "mitocarta":
        sym = md.get("symbol") or md.get("json_key") or "gene"
        return f"Mitocarta — {sym}"
    title = md.get("doc_title") or md.get("doc_id") or Path(md.get("source_file","")).name or "Source"
    page = md.get("page")
    return f"{title}, p.{page}" if page is not None else title

def _extract_terms(query_fr: str, answer_en: str = "") -> List[str]:
    q = query_fr or ""
    terms: List[str] = []
    terms += re.findall(r"\brs\d+\b", q, flags=re.I)
    terms += re.findall(r"\b(?:chr)?(?:[0-9]{1,2}|X|Y|MT):\d+:[ACGT]:[ACGT]\b", q, flags=re.I)
    terms += [t for t in re.findall(r"\b[A-Z0-9\-]{2,12}\b", q) if not t.isdigit()]
    for s in ["mitochond", "mitochondrie", "mutation", "variant", "gène", "gene", "pathog", "diagnostic", "management", "treatment"]:
        if s in q.lower() or s in (answer_en or "").lower():
            terms.append(s)
    seen, out = set(), []
    for t in terms:
        tl = t.lower()
        if tl not in seen:
            out.append(t)
            seen.add(tl)
    return out[:20]

def highlight_html(text: str, terms: List[str]) -> str:
    if not text:
        return ""
    safe = html.escape(text).replace("\n", "<br/>")
    if not terms:
        return safe
    pats = []
    for t in terms:
        if not t: continue
        pats.append(re.escape(html.escape(t)))
        if t.lower() != t:
            pats.append(re.escape(html.escape(t.lower())))
    if not pats:
        return safe
    rx = re.compile("(" + "|".join(pats) + ")", re.IGNORECASE)
    return rx.sub(r"<mark>\1</mark>", safe)

def linkify_citations(answer_text: str, n_hits: int) -> str:
    if not answer_text or n_hits <= 0:
        return answer_text
    def repl(m):
        try:
            k = int(m.group(1))
            if 1 <= k <= n_hits:
                return f'<a href="#src-{k}" style="text-decoration:none;">[{k}]</a>'
        except Exception:
            pass
        return m.group(0)
    return re.sub(r"\[(\d{1,3})\]", repl, answer_text)

# ---------- PDF helpers (optional) ----------
def _render_pdf_highlight_b64(pdf_path: str, page_index: int, para_text: str, zoom: float = 2.0) -> str | None:
    try:
        import fitz
        from PIL import Image, ImageDraw
    except Exception:
        return None
    if not (pdf_path and os.path.exists(pdf_path) and para_text):
        return None
    try:
        with fitz.open(pdf_path) as doc:
            pidx = max(0, min(int(page_index) - 1 if page_index else 0, len(doc)-1))
            pg = doc[pidx]
            needle = (para_text or "").strip().replace("\n"," ")[:800]
            rects = pg.search_for(needle, quads=False, flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_LIGATURES)
            if not rects:
                # fallback: first sentence
                s1 = re.split(r'(?<=[.!?])\s', needle, maxsplit=1)[0]
                if len(s1) > 20:
                    rects = pg.search_for(s1, quads=False, flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_LIGATURES)
            m = fitz.Matrix(zoom, zoom)
            pix = pg.get_pixmap(matrix=m, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
            if rects:
                X0 = min(r.x0 for r in rects); Y0 = min(r.y0 for r in rects)
                X1 = max(r.x1 for r in rects); Y1 = max(r.y1 for r in rects)
                x0, y0, x1, y1 = int(X0*zoom), int(Y0*zoom), int(X1*zoom), int(Y1*zoom)
                overlay = Image.new("RGBA", img.size, (0,0,0,0))
                draw = ImageDraw.Draw(overlay, "RGBA")
                draw.rectangle((x0-4, y0-4, x1+4, y1+4), fill=(255,255,0,60), outline=(220,180,0,200), width=3)
                img = Image.alpha_composite(img, overlay)
            buf = io.BytesIO(); img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None

# ---------- panel ----------
def _score_for_sort(h: Dict[str, Any]) -> float:
    return float(h.get("_adj_score", h.get("rerank_score", h.get("score", 0.0))))

def render_section_block(text: str):
    safe = html.escape(text or "").replace("\n", "<br/>")
    st.markdown(
        f"""
<div style="border:1px solid #e6e6e6;border-radius:10px;padding:12px 14px;background:#fffbe6;">
  <div style="font-size:0.92rem; line-height:1.55;">
    {safe}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

def render_sources_panel(
    hits: List[Dict[str, Any]],
    query_fr: str,
    answer_en: str = "",
    title: str = "📚 Sources",
    expanded: bool = False,           # top-level expander (collapsed by default)
    expand_children: bool = False,    # inner cards (collapsed by default)
):
    if not hits:
        return

    terms = _extract_terms(query_fr, answer_en)
    hits_sorted = sorted(hits, key=_score_for_sort, reverse=True)

    with st.expander(f"{title} ({len(hits_sorted)})", expanded=expanded):
        for i, h in enumerate(hits_sorted, 1):
            md = h.get("metadata", {}) or {}
            st.markdown(f'<div id="src-{i}"></div>', unsafe_allow_html=True)

            cite = pretty_citation(md)
            score = _score_for_sort(h)

            with st.expander(f"[{i}] {cite} — score {score:.3f}", expanded=expand_children):
                md = h.get("metadata", {}) or {}
                src = (md.get("source") or "").lower()
                pdf_path = md.get("source_file")
                page = md.get("page")
                section_text = md.get("section_text")
                is_pdf = bool(pdf_path and str(pdf_path).lower().endswith(".pdf") and os.path.exists(pdf_path))
                is_genereviews = (src == "genereviews")  # your PDF source

                # ---------- CASE A: GeneReviews (PDF): two-pane ----------
                if is_genereviews and is_pdf:
                    left, right = st.columns([12, 2], vertical_alignment="top")

                    # LEFT — Document d'origine (large)
                    with left:
                        st.markdown(":orange[**Document d'origine**]")
                        shown_left = False
                        if section_text:
                            b64 = _render_pdf_highlight_b64(pdf_path, page or 1, section_text)
                            if b64:
                                st.markdown(
                                    f'<img src="data:image/png;base64,{b64}" style="width:100%;height:auto;" />',
                                    unsafe_allow_html=True,
                                )
                                shown_left = True
                        if not shown_left:
                            # Fallbacks if no highlight or no section_text
                            if md.get("source_url"):
                                st.link_button("Ouvrir la source", md["source_url"], use_container_width=True)
                            else:
                                st.code(str(pdf_path))

                    # RIGHT — Texte extrait (in a collapsed expander by default)
                    with right:
                        with st.popover(":orange[**Texte extrait**]"):
                            if section_text:
                                st.markdown(highlight_html(section_text, terms), unsafe_allow_html=True)
                            elif md.get("json_entry") is not None:
                                pretty = json.dumps(md["json_entry"], ensure_ascii=False, indent=2)
                                st.code(pretty, language="json")
                            else:
                                snippet = (h.get("text") or "").strip()
                                st.markdown(highlight_html(snippet, terms), unsafe_allow_html=True)

                # ---------- CASE B: ClinVar / MitoCarta (non-PDF): single main panel ----------
                else:
                    st.markdown(":orange[**Texte extrait**]")
                    if section_text:
                        st.markdown(highlight_html(section_text, terms), unsafe_allow_html=True)
                    elif md.get("json_entry") is not None:
                        pretty = json.dumps(md["json_entry"], ensure_ascii=False, indent=2)
                        st.code(pretty, language="json")
                    else:
                        snippet = (h.get("text") or "").strip()
                        st.markdown(highlight_html(snippet, terms), unsafe_allow_html=True)

                # Optional: small details popover
                with st.popover("Détails"):
                    st.json(md)

# def render_source_card(hit: Dict[str, Any], i: int, query_fr: str):
#     """Render one source with consistent layout and an anchor for [i]."""
#     md = hit.get("metadata", {}) or {}
#     score = hit.get("rerank_score", hit.get("score", 0.0))
#     cite = pretty_citation(md)
#     st.markdown(f'<div id="src-{i}"></div>', unsafe_allow_html=True)
#     exp = st.expander(f"[{i}] {cite} • score={float(score):.3f}", expanded=False)
#     with exp:
#         c1, c2 = st.columns([3, 2])
#         with c1:
#             snippet = (hit.get("text") or "").strip()
#             st.markdown(highlight_text(snippet, query_fr), unsafe_allow_html=True)
#         with c2:
#             page = md.get("page")
#             sec = md.get("section")
#             src_url = md.get("source_url")
#             src_file = md.get("source_file")
#             if page is not None:
#                 st.write(f"Page : {page}")
#             if sec:
#                 st.write(f"Section : {sec}")
#             if src_url:
#                 st.write(f"[Ouvrir la source]({src_url})")
#             elif src_file:
#                 st.code(str(src_file))
#             lic = md.get("license")
#             if lic:
#                 st.caption(lic)
#
#
# def render_sources_panel(hits: List[Dict[str, Any]], query_fr: str, title: str = "📚 Sources"):
#     if not hits:
#         return
#     st.markdown(f"### {title}")
#     for i, h in enumerate(hits, 1):
#         render_source_card(h, i, query_fr)
