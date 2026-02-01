from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import os
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from ..utils import Condition

DEFAULT_PROMPT = """Decompose the following scientific claim/question into a small set of minimal conditions required for a valid conclusion.
Return each condition on its own line. Mark critical conditions with '[CRITICAL]'.
Text: {text}
"""
NUMERIC_QA_PROMPT = """You are solving a materials science problem.
Return ONLY the final answer in the exact format below.

If the answer is a tuple, return a tuple of numbers.
If the answer has units, do NOT include units inside ANSWER.
If you cannot solve, return UNKNOWN.

FORMAT:
ANSWER: <number or tuple or UNKNOWN>

Question: {q}
Unit hint: {unit_hint}
"""


class ConditionDecomposer:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = "cuda",
        backend: str = "hf_seq2seq",
        temperature: float = 0.0,
        api_base_url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.backend = backend
        self.temperature = temperature
        self.api_base_url = api_base_url

        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.tok = None
        self.model = None
        self.is_seq2seq = False
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if self.backend in ("hf_seq2seq", "hf_chat"):
            self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

            if self.backend == "hf_seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
                self.is_seq2seq = True
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
                self.is_seq2seq = False

            self.model.eval()


    @torch.no_grad()
    def decompose(self, text: str, max_new_tokens: int = 128) -> List[Condition]:
        prompt = DEFAULT_PROMPT.format(text=text.strip())

        if self.backend == "hf_seq2seq":
            enc = self.tok(prompt, return_tensors="pt", truncation=True).to(self.device)
            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature if self.temperature > 0 else None,
            )
            out = self.tok.decode(gen[0], skip_special_tokens=True).strip()

        elif self.backend == "hf_chat":
            messages = [
                {"role": "system", "content": "Decompose scientific claims into minimal auditable conditions. Output one condition per line. Mark critical ones with [CRITICAL]."},
                {"role": "user", "content": prompt},
            ]
            if hasattr(self.tok, "apply_chat_template"):
                rendered = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                rendered = "SYSTEM: " + messages[0]["content"] + "\nUSER: " + messages[1]["content"] + "\nASSISTANT:"

            enc = self.tok(rendered, return_tensors="pt", truncation=True).to(self.device)
            input_len = enc["input_ids"].shape[1]
            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tok.eos_token_id,
            )
            out = self.tok.decode(gen[0][input_len:], skip_special_tokens=True).strip()

        elif self.backend == "openai":
            out = self._decompose_openai(prompt)

        elif self.backend == "deepseek":
            out = self._decompose_deepseek(prompt)
        
        elif self.backend == "openrouter":
            out = self._decompose_openrouter(prompt)


        else:
            raise ValueError(f"Unknown decomposer backend: {self.backend}")

        conds = _parse_conditions(out)
        if not conds:
            conds = [Condition(text=text.strip(), critical=True)]
        return conds
    def _decompose_openrouter(self, prompt: str) -> str:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Optional tracking fields (OpenRouter likes these but they can be blank)
        site_url = os.environ.get("OPENROUTER_SITE_URL")
        site_name = os.environ.get("OPENROUTER_SITE_NAME")
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-Title"] = site_name

        payload = {
            "model": self.model_name,  # e.g. "openai/gpt-4o-mini" or "meta-llama/llama-3.3-70b-instruct"
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Decompose scientific claims into minimal auditable conditions. "
                        "Output one condition per line. Mark critical ones with [CRITICAL]."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": float(self.temperature),
        }

        # r = requests.post(
        #     "https://openrouter.ai/api/v1/chat/completions",
        #     headers=headers,
        #     json=payload,
        #     timeout=120,
        # )
        session = getattr(self, "_or_session", None)
        if session is None:
            session = self._session_with_retries()
            self._or_session = session
        
        r = session.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=(30, 180),  # (connect timeout, read timeout) — increase read timeout
        )

        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    
    def _answer_openrouter(self, prompt: str) -> str:
        # Same endpoint as decomposition, different prompt
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(getattr(self, "temperature", 0.0)),
            "max_tokens": int(getattr(self, "max_new_tokens", 128)),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Optional metadata (safe even if empty)
        if getattr(self, "http_referer", None):
            headers["HTTP-Referer"] = self.http_referer
        if getattr(self, "x_title", None):
            headers["X-Title"] = self.x_title

        session = getattr(self, "_or_session", None)
        if session is None:
            session = requests.Session()
            self._or_session = session

        r = session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=(30, 300),
        )
        r.raise_for_status()
        data = r.json()

        # OpenRouter response shape
        return data["choices"][0]["message"]["content"]

    
    # @torch.no_grad()
    @torch.no_grad()
    def answer_numeric(
        self,
        question: str,
        unit_hint: str = "",
        max_new_tokens: int = 96,
    ) -> str:
        """
        Numeric QA helper for MatSciBench-like tasks.

        Returns a single-line string in the strict format:
            ANSWER: <number | tuple | UNKNOWN>

        Works for:
        - HF seq2seq models (e.g., google/flan-t5-*)
        - OpenRouter / OpenAI / DeepSeek / HF-chat backends (if you have those methods)
        """
        import re

        q = (question or "").strip()
        u = (unit_hint or "").strip()

        prompt = (
            "You are solving a scientific / materials science problem.\n"
            "Return ONLY the final answer in EXACTLY this format:\n"
            "ANSWER: <number or tuple or UNKNOWN>\n\n"
            "Rules:\n"
            "- If the answer is a tuple, output a tuple of numbers like (a, b, c).\n"
            "- Do NOT include units inside ANSWER.\n"
            "- Do NOT write any explanation.\n"
            "- If you cannot determine the answer, output: ANSWER: UNKNOWN\n\n"
            f"Question: {q}\n"
            f"Unit hint: {u}\n"
        )

        backend = getattr(self, "backend", "hf_seq2seq")
        backend = (backend or "").lower().strip()

        # --------- 1) GET RAW TEXT OUTPUT (out) ----------
        if backend == "openrouter":
            out = self._answer_openrouter(prompt)
        elif backend == "openai":
            out = self._answer_openai(prompt)
        elif backend == "deepseek":
            out = self._answer_deepseek(prompt)
        elif backend in ("hf_chat", "chat", "hf"):
            out = self._answer_hf_chat(prompt)
        else:
            # HF seq2seq fallback (your flan-t5 path)
            # Use the same pattern as decompose(): tokenizer + generate
            enc = self.tok(prompt, return_tensors="pt", truncation=True).to(self.device)
            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            out = self.tok.decode(gen[0], skip_special_tokens=True)

        out = (out or "").strip()

        # --------- 2) NORMALIZE TO STRICT "ANSWER:" FORMAT ----------
        # helpers
        def _parse_numbers_any(s: str):
            if not s:
                return []
            s = str(s).strip()
            s = s.replace(",", "")
            s = s.replace("×10^", "e").replace("x10^", "e").replace("X10^", "e")
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            vals = []
            for t in nums:
                try:
                    vals.append(float(t))
                except Exception:
                    pass
            return vals

        def _fmt_num(x: float) -> str:
            return str(int(x)) if abs(x - int(x)) < 1e-12 else str(x)

        # If model included ANSWER:, keep only the last/first clean payload
        if "answer:" in out.lower():
            # take the first "ANSWER:" occurrence's payload
            payload = out.split("ANSWER:", 1)[-1].strip()
            # in case it continued with extra stuff, cut at newline
            payload = payload.splitlines()[0].strip()
        else:
            payload = out.splitlines()[0].strip() if out else ""

        # kill common placeholder junk
        if payload.lower().startswith(("number>", "<number", "number", "num>")):
            payload = ""

        # unknown handling
        if "unknown" in payload.lower():
            return "ANSWER: UNKNOWN"

        # tuple handling
        if payload.startswith("(") and ")" in payload:
            # try to extract numbers inside tuple-ish text
            nums = _parse_numbers_any(payload)
            if nums:
                return "ANSWER: (" + ", ".join(_fmt_num(x) for x in nums) + ")"
            return "ANSWER: UNKNOWN"

        # single or multiple numbers
        nums = _parse_numbers_any(payload)
        if not nums:
            # salvage from whole output if first line had no number
            nums = _parse_numbers_any(out)

        if not nums:
            return "ANSWER: UNKNOWN"
        if len(nums) == 1:
            return f"ANSWER: {_fmt_num(nums[0])}"
        return "ANSWER: (" + ", ".join(_fmt_num(x) for x in nums) + ")"



    def _session_with_retries(self) -> requests.Session:
        s = requests.Session()
        retry = Retry(
            total=8,
            connect=8,
            read=8,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s


# def _parse_conditions(generated: str) -> List[Condition]:
#     lines = [l.strip("-• \t") for l in generated.splitlines() if l.strip()]
#     conds: List[Condition] = []
#     for l in lines:
#         crit = False
#         if "[CRITICAL]" in l.upper():
#             crit = True
#             l = re.sub(r"\[\s*critical\s*\]", "", l, flags=re.I).strip()

#         # Clean up common FLAN formatting like leading ":" and skip empty conditions
#         l = l.lstrip(":").strip()
#         if not l:
#             continue

#         # if model didn't mark, treat as critical by default
#         conds.append(Condition(text=l, critical=True if crit or len(lines) <= 3 else False))

#     # Fallback: if everything got filtered out, treat the whole generated text as one condition
#     if not conds:
#         cleaned = generated.strip()
#         if cleaned:
#             conds = [Condition(text=cleaned, critical=True)]
#         else:
#             conds = [Condition(text="", critical=True)]

#     # ensure at least one critical
#     if not any(c.critical for c in conds):
#         conds[0].critical = True
#     return conds

def _parse_conditions(generated: str) -> List[Condition]:
    lines = [l.strip("-• \t") for l in generated.splitlines() if l.strip()]
    conds: List[Condition] = []

    for l in lines:
        crit = False

        # detect + remove marker
        if "[CRITICAL]" in l.upper():
            crit = True
            l = re.sub(r"\[\s*critical\s*\]", "", l, flags=re.I).strip()

        # cleanup common formatting
        l = l.lstrip(":").strip()

        # drop empty / marker-only lines
        if not l or l.upper() == "[CRITICAL]":
            continue

        # if model didn't mark, treat as critical by default when few lines
        conds.append(Condition(text=l, critical=True if crit or len(lines) <= 3 else False))

    # if everything got filtered out, fall back to using the whole output as one condition
    if not conds:
        fallback = re.sub(r"\[\s*critical\s*\]", "", generated, flags=re.I).strip()
        fallback = fallback.lstrip(":").strip()
        conds = [Condition(text=fallback if fallback else "", critical=True)]

    # ensure at least one critical
    if not any(c.critical for c in conds):
        conds[0].critical = True

    return conds

def _parse_answer_line(text: str) -> str:
    """
    Extracts the required format:
      ANSWER: ... | UNIT: ...
    If model outputs extra text, we pull the first line that contains 'ANSWER:'.
    """
    if not text:
        return "ANSWER: UNKNOWN | UNIT:"

    # Take first line that contains ANSWER:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines:
        if "ANSWER:" in l.upper():
            # normalize spacing
            l = l.replace("answer:", "ANSWER:").replace("unit:", "UNIT:")
            # if missing UNIT part, add it
            if "| UNIT:" not in l.upper():
                # try split by UNIT:
                if "UNIT:" in l.upper():
                    # ensure delimiter
                    l = l.replace("UNIT:", "| UNIT:")
                else:
                    l = l + " | UNIT:"
            # if answer is empty, set unknown
            if "ANSWER:" in l and l.split("ANSWER:", 1)[1].strip().startswith("|"):
                return "ANSWER: UNKNOWN | UNIT:"
            return l

    # fallback if no ANSWER line
    return "ANSWER: UNKNOWN | UNIT:"



