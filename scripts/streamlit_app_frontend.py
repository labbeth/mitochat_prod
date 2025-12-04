from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st



# --- make sure we can import pdf_rendering in the same folder ---
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))
import pdf_rendering as pr  # <- as you asked

APP_TITLE = "Mito Chat"
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:9000")

# Bypass corp proxies for localhost
for var in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"]:
    os.environ.pop(var, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"


def post_rag(backend_url: str, chat_history: List[Dict[str, str]], query_fr: str, timeout: int = 120) -> Dict[str, Any]:
    payload = {"chat_history": chat_history, "query_fr": query_fr}
    url = backend_url.rstrip("/") + "/rag"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main():
    st.set_page_config(page_title="Local RAG Q&A (frontend)", layout="wide")

    st.markdown("""
    <style>
    .sent-block{
      background:#fffbe6;
      border:1px solid #f0e6b3;
      border-radius:8px;
      padding:6px 8px;
      display:inline;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    c1, c2 = st.columns([1, 8], vertical_alignment="center")
    with c1:
        try:
            st.image("http://localhost:9000/assets/logo.png", use_container_width=True)
        except Exception:
            st.empty()
    with c2:
        st.title(APP_TITLE)

    # Sidebar (= knobs you actually need client-side)
    with st.sidebar:
        if st.button("Nouveau chat", use_container_width=True):
            st.session_state.chat = []
            st.rerun()

        backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL)
        show_debug = st.checkbox("Show rewrite/debug info", value=False)

        # # DEBUG
        # st.subheader("Backend debug")
        # dbg_q = st.text_input("Query (FR) for /rag_debug", value="Quels systèmes corporels sont affectés par la DM1 ?")
        # if st.button("Run /rag_debug"):
        #     try:
        #         resp = requests.post(
        #             backend_url.rstrip("/") + "/rag_debug",
        #             json={"chat_history": [], "query_fr": dbg_q},
        #             timeout=120,
        #         )
        #         resp.raise_for_status()
        #         st.json(resp.json())
        #     except Exception as e:
        #         st.error(f"/rag_debug failed: {e}")
        # # END DEBUG

        st.divider()
        st.caption("Santé du backend :")
        try:
            hb = requests.get(backend_url.rstrip("/") + "/health", timeout=5).json()
            st.success(f"OK: {hb}")
        except Exception as e:
            st.error(f"Health check failed: {e}")

        # DEBUG
        if st.button("Ping backend"):
            try:
                health = requests.get(backend_url.rstrip("/") + "/health", timeout=5).json()
                who = requests.get(backend_url.rstrip("/") + "/whoami", timeout=5).json()
                st.success(f"Health: {health}")
                st.info(f"WhoAmI: {who}")
            except Exception as e:
                st.error(f"Ping failed: {e}")

    # Chat state
    if "chat" not in st.session_state:
        # {"role": "user"/"assistant", "content": str, optional: rewrite/router/hits/answer_en}
        st.session_state.chat = []

    # Replay history like your local app
    live = st.session_state.get("_rendering_live", False)

    for i, m in enumerate(st.session_state.chat):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            # Under user message, show rewrite + router if present
            if m["role"] == "user" and show_debug:
                if m.get("rewrite"):
                    with st.expander("Requête utilisée pour la recherche", expanded=False):
                        st.code(m["rewrite"], language="text")
                if m.get("router"):
                    with st.expander("Router decision", expanded=False):
                        st.json(m["router"])
            # Under assistant message, show sources (skip live one to avoid duplicate-on-rerun)
            # if m["role"] == "assistant" and m.get("hits"):
            #     if live and i == len(st.session_state.chat) - 1:
            #         continue
            #     pr.render_sources_panel(
            #         m["hits"],
            #         query_fr=st.session_state.chat[i-1]["content"] if (i>0 and st.session_state.chat[i-1]["role"]=="user") else "",
            #         answer_en=m.get("answer_en", ""),
            #         title="📚 Sources",
            #         expanded=False,
            #         expand_children=False,
            #     )
            if m["role"] == "assistant":
                has_hits = bool(m.get("hits"))
                is_llm = (m.get("mode", "").upper() == "LLM")

                # Avoid duplicate rendering for the live last assistant
                if live and i == len(st.session_state.chat) - 1:
                    continue

                # Only render sources if not LLM mode
                if has_hits and not is_llm:
                    pr.render_sources_panel(
                        m["hits"],
                        query_fr=st.session_state.chat[i-1]["content"] if (i > 0 and st.session_state.chat[i-1]["role"] == "user") else "",
                        answer_en=m.get("answer_en", ""),
                        title="📚 Sources",
                        expanded=False,
                        expand_children=False,
                    )


    # Input
    user_msg = st.chat_input("Posez une question sur les maladies mitochondriales…")
    if user_msg:
        # (1) push user turn
        st.session_state.chat.append({"role": "user", "content": user_msg})
        user_idx = len(st.session_state.chat) - 1
        with st.chat_message("user"):
            st.markdown(user_msg)

        # (2) call backend (does rewrite + retrieval + router + generation)
        with st.chat_message("assistant"):
            with st.spinner("Collecte d'information…"):
                st.caption(f"→ POST {backend_url.rstrip('/')}/rag")  # DEBUG
                try:
                    resp = post_rag(backend_url, st.session_state.chat[:-1], user_msg)
                except Exception as e:
                    st.error(f"Erreur d'appel backend: {e}")
                    st.stop()

        # (3) unpack response
        answer_fr = resp.get("answer_fr") or "(pas de réponse)"
        answer_en = resp.get("answer_en") or ""
        hits = resp.get("hits", []) or []
        router = resp.get("router") or {}
        rewrite_en = resp.get("rewrite_en") or ""
        mode = resp.get("mode", "RAG")

        # Linkify [1], [2] → #src-i anchors
        answer_fr_linked = pr.linkify_citations(answer_fr, len(hits))

        # (4) store rewrite/router on the user turn (so they render under the user bubble)
        st.session_state.chat[user_idx]["rewrite"] = rewrite_en
        st.session_state.chat[user_idx]["router"] = {
            "mode": mode,
            "use_ids": router.get("use_ids", []),
            "raw": router,
            "hits_count": len(hits),
        }

        # (5) append assistant turn with hits, rerun to render like the local app
        st.session_state.chat.append({
            "role": "assistant",
            "content": answer_fr_linked,
            "hits": hits,
            "answer_en": answer_en,
            "rewrite": rewrite_en,
            "mode": mode,
            "router": st.session_state.chat[user_idx]["router"],
        })

        # Tiny inline debug line (helps confirm backend was called)
        st.info(f"Mode: {mode} • Hits: {len(hits)}")

        st.session_state.pop("_rendering_live", None)
        st.rerun()


if __name__ == "__main__":
    main()
