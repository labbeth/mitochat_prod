'''Translation script using local models'''


from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Union
import os
import warnings
from pathlib import Path

import torch
import yaml
from transformers import MarianMTModel, MarianTokenizer


# ---------- Config loading ----------

def _config_path() -> str:
    # Allows override with APP_CONFIG env var
    return os.getenv("APP_CONFIG", os.path.join(os.path.dirname(__file__), "config.yaml"))


def _load_config() -> dict:
    path = _config_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_cfg(cfg: dict, *keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------- Utilities ----------
def _resolve_model_path(p: str) -> str:
    """
    Resolve a model path string from config to an absolute path.

    - If `p` is already absolute, keep it.
    - If `p` is relative, interpret it relative to the project root
      (parent of this file's directory, i.e. rag_app/).
    """
    path = Path(p)
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[1]  # rag_app/
        path = (root / path).resolve()
    return str(path)


def _device(auto: Optional[str] = None) -> str:
    if auto:
        return auto
    return "cuda" if torch.cuda.is_available() else "cpu"


# Check SentencePiece (required for Marian)
try:
    import sentencepiece  # noqa: F401
    _SPM_OK = True
except Exception:
    _SPM_OK = False


# ---------- Marian Translator ----------

class HFTranslator:
    """
    MarianMT FR<->EN translator (offline, safetensors-only).

    - Loads strictly from local directories (no internet).
    - Uses CUDA if available, else CPU.
    - Requires SentencePiece.
    - Security: use_safetensors=True avoids pickle-based .bin weights.
    """

    def __init__(
        self,
        fr_en_model: str,
        en_fr_model: str,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        num_beams: int = 4,
        local_files_only: bool = True,
    ):
        if not _SPM_OK:
            raise RuntimeError("SentencePiece is required. Please `pip install sentencepiece`.")

        self.device = _device(device)
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)
        self.local_files_only = bool(local_files_only)

        # --- Tokenizers ---
        self.tok_fr_en = MarianTokenizer.from_pretrained(
            fr_en_model,
            local_files_only=self.local_files_only,
        )
        self.tok_en_fr = MarianTokenizer.from_pretrained(
            en_fr_model,
            local_files_only=self.local_files_only,
        )

        # --- Models (use safetensors for security) ---
        self.mod_fr_en = MarianMTModel.from_pretrained(
            fr_en_model,
            use_safetensors=True,
            low_cpu_mem_usage=False,
            local_files_only=self.local_files_only,
        ).to(self.device).eval()

        self.mod_en_fr = MarianMTModel.from_pretrained(
            en_fr_model,
            use_safetensors=True,
            low_cpu_mem_usage=False,
            local_files_only=self.local_files_only,
        ).to(self.device).eval()

    @torch.inference_mode()
    def translate(self, text: Union[str, List[str]], src: str, tgt: str) -> Union[str, List[str]]:
        if not text or src == tgt:
            return text

        if src.lower().startswith("fr") and tgt.lower().startswith("en"):
            tok, mod = self.tok_fr_en, self.mod_fr_en
        elif src.lower().startswith("en") and tgt.lower().startswith("fr"):
            tok, mod = self.tok_en_fr, self.mod_en_fr
        else:
            return text  # unsupported pair

        return self._translate_any(text, tok, mod)

    def _translate_any(
        self,
        text: Union[str, List[str]],
        tok: MarianTokenizer,
        mod: MarianMTModel,
    ) -> Union[str, List[str]]:
        if isinstance(text, str):
            s = (text or "").strip()
            if not s:
                return text
            enc = tok(s, return_tensors="pt", truncation=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = mod.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
            )
            return tok.batch_decode(out, skip_special_tokens=True)[0]

        batch = [t.strip() for t in (text or [])]
        if not batch:
            return text
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = mod.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            early_stopping=True,
        )
        return tok.batch_decode(out, skip_special_tokens=True)


# ---------- Factory & helpers ----------

@lru_cache(maxsize=1)
def get_translator(
    device: Optional[str] = None,
    max_new_tokens: int = 512,
    num_beams: int = 4,
    cache_bust: str = "v1",
) -> HFTranslator:
    """
    Build and cache a Marian translator using paths from config.yaml.
    """
    cfg = _load_config()
    fr_en_model = _get_cfg(cfg, "translation", "fr_en_model")
    en_fr_model = _get_cfg(cfg, "translation", "en_fr_model")

    if not fr_en_model or not en_fr_model:
        raise ValueError(
            "Missing translation model paths in config.yaml.\n"
            "Expected keys:\n"
            "  translation.fr_en_model: ./models/Helsinki-NLP/opus-mt-fr-en\n"
            "  translation.en_fr_model: ./models/Helsinki-NLP/opus-mt-en-fr"
        )

    # Resolve to absolute paths relative to project root if needed
    fr_en_model = _resolve_model_path(fr_en_model)
    en_fr_model = _resolve_model_path(en_fr_model)

    if not Path(fr_en_model).exists():
        raise FileNotFoundError(f"Translation model path does not exist: {fr_en_model}")
    if not Path(en_fr_model).exists():
        raise FileNotFoundError(f"Translation model path does not exist: {en_fr_model}")

    # Force offline safety defaults
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    local_only = os.getenv("HF_LOCAL_ONLY", "1").strip() != "0"

    return HFTranslator(
        fr_en_model=fr_en_model,
        en_fr_model=en_fr_model,
        device=device,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        local_files_only=local_only,
    )


def translate_text(
    text: Union[str, List[str]],
    src: str,
    tgt: str,
    translator: Optional[HFTranslator] = None,
) -> Union[str, List[str]]:
    tr = translator or get_translator()
    try:
        return tr.translate(text, src, tgt)
    except Exception as e:
        warnings.warn(f"Translation failed; returning input. Error: {e}", RuntimeWarning)
        return text


# ---------- CLI quick test ----------

if __name__ == "__main__":
    t = get_translator()
    print(t.translate("Bonjour, comment allez-vous ?", src="fr", tgt="en"))
    print(t.translate("The patient received amoxicillin.", src="en", tgt="fr"))
