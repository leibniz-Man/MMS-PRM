import argparse
import json
import os
import time
import base64
import mimetypes
import random
import re
from typing import List, Dict, Any, Optional
try:
    # 更智能的进度条（在终端/Notebook 都有良好显示）
    from tqdm.auto import tqdm
except Exception:
    # 若未安装 tqdm，则优雅降级为普通迭代
    def tqdm(iterable, total=None, desc=None, unit=None):
        return iterable
# Reuse existing parsing utilities if available
try:
    from tools.check_dpo_reasoning_length import extract_reasoning, parse_steps
except Exception:
    # Minimal fallbacks if import fails
    import re

    def extract_reasoning(text: str) -> str:
        if not isinstance(text, str):
            return ""
        start_tag = "<|reasoning_start|>"
        end_tag = "<|reasoning_end|>"
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx + len(start_tag): end_idx]
        return text

    def parse_steps(reasoning_block: str) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        if not reasoning_block:
            return steps
        step_pat = re.compile(r"<\|reasoning_step_start\|>(.*?)<\|reasoning_step_end\|>", re.DOTALL)
        for seg in step_pat.findall(reasoning_block):
            def _field(start_tag: str, end_tag: str) -> Optional[str]:
                m = re.search(rf"{re.escape(start_tag)}(.*?){re.escape(end_tag)}", seg, re.DOTALL)
                return m.group(1).strip() if m else None

            step = {
                "name": _field("<|reasoning_step_name_start|>", "<|reasoning_step_name_end|>"),
                "thought": _field("<|reasoning_step_thought_start|>", "<|reasoning_step_thought_end|>"),
                "reflection": _field("<|reasoning_step_reflection_start|>", "<|reasoning_step_reflection_end|>"),
            }
            if not step["name"] and not step["thought"] and not step["reflection"]:
                step["raw"] = seg.strip()
            steps.append(step)
        return steps


def step_to_text(step: Dict[str, Any]) -> str:
    """Convert parsed step dict into a single human-readable string."""
    parts = []
    if step.get("name"):
        parts.append(f"[NAME] {step['name']}")
    if step.get("thought"):
        parts.append(f"[THOUGHT] {step['thought']}")
    if step.get("reflection"):
        parts.append(f"[REFLECTION] {step['reflection']}")
    if not parts and step.get("raw"):
        parts.append(step["raw"])
    return " \n".join(parts).strip()


def extract_steps(text: Optional[str]) -> List[str]:
    """Extract reasoning steps as text lines from a raw chosen/rejected field."""
    if not text:
        return []
    rb = extract_reasoning(text)
    parsed = parse_steps(rb)
    if parsed:
        return [step_to_text(s) for s in parsed]
    # Fallback: split lines heuristically
    lines = [ln.strip() for ln in rb.splitlines() if ln.strip()]
    return lines


def _is_url(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return s.startswith("http://") or s.startswith("https://") or s.startswith("data:")


def _encode_image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _to_image_part(image: str, image_root: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not image:
        return None
    url = None
    if _is_url(image):
        url = image
    else:
        p = image
        if image_root and not os.path.isabs(p):
            p = os.path.join(image_root, p)
        if os.path.exists(p):
            try:
                url = _encode_image_to_data_url(p)
            except Exception:
                url = None
    if not url:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _extract_images(obj: Dict[str, Any]) -> List[str]:
    images: List[str] = []
    candidate_keys = [
        "image", "image_url", "image_path",
        "images", "image_urls", "image_paths",
        "picture", "pictures",
    ]
    for k in candidate_keys:
        if k in obj and obj[k] is not None:
            val = obj[k]
            if isinstance(val, str):
                images.append(val)
            elif isinstance(val, list):
                for it in val:
                    if isinstance(it, str):
                        images.append(it)
                    elif isinstance(it, dict):
                        # try typical dict forms {url: ..., path: ...}
                        u = it.get("url") or it.get("path")
                        if isinstance(u, str):
                            images.append(u)
            elif isinstance(val, dict):
                u = val.get("url") or val.get("path")
                if isinstance(u, str):
                    images.append(u)
    # de-duplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for img in images:
        if img not in seen:
            uniq.append(img)
            seen.add(img)
    return uniq


def load_criteria_pairs(path: Optional[str]) -> Dict[str, List[str]]:
    """Load named parent->children pairs from lines like `Parent：c1、c2`."""
    if not path or not os.path.exists(path):
        return {}
    out: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "：" in line:
                parent, children = line.split("：", 1)
            elif ":" in line:
                parent, children = line.split(":", 1)
            else:
                continue
            parent = parent.strip()
            parts = [x.strip() for x in re.split(r"[、,;；]", children) if x.strip()]
            if parent and parts:
                dedup = []
                seen = set()
                for p in parts:
                    if p not in seen:
                        dedup.append(p)
                        seen.add(p)
                out[parent] = dedup
    return out


class BGEEmbedder:
    """BGE text encoder used for criteria/parent similarity in paper-style pipeline."""
    def __init__(self, model_name: str, batch_size: int = 32, device: Optional[str] = None):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except Exception as e:
            raise RuntimeError("BGE mode requires `torch` and `transformers` installed.") from e
        self.torch = torch
        self.batch_size = max(1, int(batch_size))
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.cache: Dict[str, Any] = {}

    def _embed_batch(self, texts: List[str]) -> Any:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            masked = hidden * mask
            summed = masked.sum(dim=1)
            counts = mask.sum(dim=1)
            pooled = summed / self.torch.clamp(counts, min=1)
            vec = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
        return vec.detach().cpu()

    def encode_many(self, texts: List[str]) -> None:
        pending = [t for t in texts if isinstance(t, str) and t and t not in self.cache]
        if not pending:
            return
        for i in range(0, len(pending), self.batch_size):
            batch = pending[i:i + self.batch_size]
            vecs = self._embed_batch(batch)
            for j, t in enumerate(batch):
                self.cache[t] = vecs[j]

    def cosine(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        self.encode_many([a, b])
        va = self.cache.get(a)
        vb = self.cache.get(b)
        if va is None or vb is None:
            return 0.0
        return float((va * vb).sum().item())


def build_phi_messages(step_text: str, parent: str, candidate_hints: List[str], max_criteria_per_step: int) -> List[Dict[str, Any]]:
    system = (
        "You are an analysis function Phi for fine-grained reward construction. "
        "Given one reasoning step and a selected coarse parent reward dimension, "
        "generate up to 5 candidate textual evaluation criteria focused on this parent only. "
        "Return STRICT JSON only: {\"candidate_criteria\": [\"...\", ...]}."
    )
    user_payload = {
        "step_text": step_text,
        "selected_parent": parent,
        "candidate_hints": candidate_hints[:20],
        "max_candidates": min(5, max_criteria_per_step),
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}]}]


def generate_phi_candidates(
    step_text: str,
    parent: str,
    candidate_hints: List[str],
    max_criteria_per_step: int,
    model: str,
    temperature: float,
    base_url: Optional[str],
    max_tokens: Optional[int],
) -> List[str]:
    if not parent:
        return []
    msgs = build_phi_messages(step_text, parent, candidate_hints, max_criteria_per_step)
    resp = call_gpt4o_until_success(
        msgs,
        model=model,
        temperature=temperature,
        base_url=base_url,
        max_tokens=max_tokens,
        response_json=True,
        expected_keys=["candidate_criteria"],
    )
    c = resp.get("candidate_criteria")
    if not isinstance(c, list):
        return []
    out: List[str] = []
    seen = set()
    for x in c:
        if isinstance(x, str):
            v = x.strip()
            if v and v not in seen:
                out.append(v)
                seen.add(v)
        if len(out) >= min(5, max_criteria_per_step):
            break
    return out


def select_parent_and_candidates(
    step_text: str,
    criteria_catalog: Dict[str, List[str]],
    embedder: BGEEmbedder,
    model: str,
    temperature: float,
    base_url: Optional[str],
    max_tokens: Optional[int],
    max_criteria_per_step: int,
    parent_distance_threshold: float,
) -> Dict[str, Any]:
    """Select a coarse parent and pre-filter candidate criteria by distance."""
    if not criteria_catalog:
        return {"selected_parent": None, "candidate_criteria": []}

    # Parent selection via BGE cosine similarity between step and top-level parent names.
    best_parent = None
    best_sim = -1.0
    parents = list(criteria_catalog.keys())
    embedder.encode_many([step_text] + parents)
    for parent in parents:
        sim = embedder.cosine(step_text, parent)
        if sim > best_sim:
            best_sim = sim
            best_parent = parent
    if best_parent is None:
        return {"selected_parent": None, "candidate_criteria": []}

    # Phi generates step-conditioned candidate criteria under the selected parent.
    parent_children = criteria_catalog.get(best_parent, [])
    phi_candidates = generate_phi_candidates(
        step_text=step_text,
        parent=best_parent,
        candidate_hints=parent_children,
        max_criteria_per_step=max_criteria_per_step,
        model=model,
        temperature=temperature,
        base_url=base_url,
        max_tokens=max_tokens,
    )
    if not phi_candidates:
        phi_candidates = list(parent_children)

    # Filter by parent distance using paper-style cosine distance.
    zeta = max(0.0, min(1.0, parent_distance_threshold))
    filtered: List[str] = []
    embedder.encode_many(phi_candidates + [best_parent])
    for c in phi_candidates:
        sim = embedder.cosine(c, best_parent)
        dist = 1.0 - sim
        if dist <= zeta:
            filtered.append(c)
    if not filtered:
        filtered = list(phi_candidates)

    return {
        "selected_parent": best_parent,
        "candidate_criteria": filtered[:max_criteria_per_step],
    }


def build_step_payloads(
    steps: List[str],
    criteria_catalog: Dict[str, List[str]],
    embedder: BGEEmbedder,
    model: str,
    temperature: float,
    base_url: Optional[str],
    max_tokens: Optional[int],
    max_criteria_per_step: int,
    parent_distance_threshold: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, s in enumerate(steps, start=1):
        alloc = select_parent_and_candidates(
            step_text=s,
            criteria_catalog=criteria_catalog,
            embedder=embedder,
            model=model,
            temperature=temperature,
            base_url=base_url,
            max_tokens=max_tokens,
            max_criteria_per_step=max_criteria_per_step,
            parent_distance_threshold=parent_distance_threshold,
        )
        out.append({
            "step_index": i,
            "step_text": s,
            "selected_parent": alloc.get("selected_parent"),
            "candidate_criteria": alloc.get("candidate_criteria", []),
        })
    return out


def build_messages(
    query: Optional[str],
    chosen_steps: List[Dict[str, Any]],
    rejected_steps: List[Dict[str, Any]],
    image_parts: Optional[List[Dict[str, Any]]] = None,
    compact: bool = False,
    min_criteria_per_step: int = 3,
    max_criteria_per_step: int = 5,
) -> List[Dict[str, Any]]:
    if compact:
        system = (
            "You are a strict step-level judge for math and diagram reasoning. "
            "Evaluate each step using ONLY criteria from the provided CANDIDATE LIST if available. "
            "Select 3-5 most relevant criteria per step, assign integer scores in [1,10], and return JSON only."
        )
    else:
        system = (
            "You are a strict step-level judge for math and diagram reasoning. "
            "For each step in chosen/rejected chains, evaluate with CANDIDATE LIST from the selected parent dimension. "
            "Calibration (use full 1-10 range): 1-2 off-topic/hallucinated; 3-4 clear error; 5 neutral setup; "
            "6-7 minor progress with issues; 8 solid correct progress; 9-10 exceptional verified progress. "
            f"Selection: choose {min_criteria_per_step}-{max_criteria_per_step} criteria per step (max {max_criteria_per_step}). "
            "If list is too coarse, refine at most two criteria into finer-grained ones. "
            "Return STRICT JSON only with keys chosen_step_reviews/rejected_step_reviews. "
            "Each review must include: step_index, step_text, selected_parent, why_good/why_bad, "
            "reward_categories (list), reward_scores (dict criterion->integer 1-10)."
        )

    user_payload = {
        "query": query or "",
        "chosen_steps": chosen_steps,
        "rejected_steps": rejected_steps,
        "constraints": {
            "min_criteria_per_step": min_criteria_per_step,
            "max_criteria_per_step": max_criteria_per_step,
            "use_candidate_list_only_if_available": True,
            "score_range": [1, 10],
        },
    }

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}]
    if image_parts:
        if compact:
            first = next((p for p in image_parts if p), None)
            if first:
                user_content.append(first)
        else:
            user_content.extend([p for p in image_parts if p])

    return [{"role": "system", "content": system}, {"role": "user", "content": user_content}]


def _coerce_score(v: Any) -> int:
    try:
        iv = int(round(float(v)))
    except Exception:
        iv = 5
    return max(1, min(10, iv))


def _normalize_step_reviews(
    reviews: Any,
    fallback_steps: List[Dict[str, Any]],
    min_criteria_per_step: int,
    max_criteria_per_step: int,
    polarity_key: str,
) -> List[Dict[str, Any]]:
    """Post-process model output into stable schema and enforce 3-5 criteria."""
    step_map = {int(x["step_index"]): x for x in fallback_steps if isinstance(x, dict) and "step_index" in x}
    normalized: List[Dict[str, Any]] = []
    raw_list = reviews if isinstance(reviews, list) else []

    for item in raw_list:
        if not isinstance(item, dict):
            continue
        step_index = int(item.get("step_index") or 0)
        fb = step_map.get(step_index, {})
        step_text = item.get("step_text") or fb.get("step_text") or ""
        selected_parent = item.get("selected_parent") or fb.get("selected_parent")
        candidate_criteria = fb.get("candidate_criteria", []) if isinstance(fb.get("candidate_criteria", []), list) else []

        cats = item.get("reward_categories")
        if not isinstance(cats, list):
            cats = []
        dedup_cats: List[str] = []
        seen = set()
        for c in cats:
            if isinstance(c, str):
                c = c.strip()
                if c and c not in seen:
                    dedup_cats.append(c)
                    seen.add(c)

        score_map = item.get("reward_scores")
        if not isinstance(score_map, dict):
            score_map = {}
        final_scores: Dict[str, int] = {}
        for c in dedup_cats:
            final_scores[c] = _coerce_score(score_map.get(c, 5))

        # Backfill to at least min_criteria_per_step, prioritizing candidate list.
        if len(dedup_cats) < min_criteria_per_step:
            for c in candidate_criteria:
                if len(dedup_cats) >= min_criteria_per_step:
                    break
                if isinstance(c, str) and c and c not in seen:
                    dedup_cats.append(c)
                    seen.add(c)
                    final_scores[c] = 5

        # Final truncation to max_criteria_per_step.
        dedup_cats = dedup_cats[:max_criteria_per_step]
        final_scores = {k: final_scores.get(k, 5) for k in dedup_cats}

        normalized_scores = {k: (v - 1) / 9.0 for k, v in final_scores.items()}
        mean_normalized = (sum(normalized_scores.values()) / len(normalized_scores)) if normalized_scores else 0.0

        out = {
            "step_index": step_index if step_index > 0 else None,
            "step_text": step_text,
            "selected_parent": selected_parent,
            polarity_key: item.get(polarity_key, ""),
            "reward_categories": dedup_cats,
            "reward_scores": final_scores,
            "reward_scores_normalized": normalized_scores,
            "mean_normalized_score": mean_normalized,
        }
        normalized.append(out)

    # Fallback: ensure every parsed step has one review entry.
    existing = {x.get("step_index") for x in normalized}
    for fb in fallback_steps:
        idx = fb.get("step_index")
        if idx in existing:
            continue
        candidate_criteria = fb.get("candidate_criteria", []) if isinstance(fb.get("candidate_criteria", []), list) else []
        cats = [c for c in candidate_criteria if isinstance(c, str) and c][:max_criteria_per_step]
        if len(cats) < min_criteria_per_step:
            while len(cats) < min_criteria_per_step:
                cats.append(f"Generic Criterion {len(cats)+1}")
        scores = {c: 5 for c in cats}
        normalized.append({
            "step_index": idx,
            "step_text": fb.get("step_text", ""),
            "selected_parent": fb.get("selected_parent"),
            polarity_key: "",
            "reward_categories": cats,
            "reward_scores": scores,
            "reward_scores_normalized": {k: (v - 1) / 9.0 for k, v in scores.items()},
            "mean_normalized_score": 4.0 / 9.0 if scores else 0.0,
        })

    normalized.sort(key=lambda x: (x.get("step_index") is None, x.get("step_index") or 0))
    return normalized


def call_gpt4o(
    messages: List[Dict[str, Any]],
    model: str = "gpt-4o",
    temperature: float = 0.2,
    base_url: Optional[str] = None,
    max_tokens: Optional[int] = None,
    response_json: bool = False,
) -> Dict[str, Any]:
    """Call GPT-4o and return parsed JSON from the assistant's reply."""
    # Prefer official client; fallback to requests if unavailable
    try:
        from openai import OpenAI
        # Prefer provided base_url or environment variables
        env_base = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE")
            or os.getenv("BASE_URL")
        )
        env_api_key = os.getenv("OPENAI_API_KEY")
        if env_base:
            # Ensure trailing /v1 for compatibility
            normalized_base = env_base.rstrip("/")
            if not normalized_base.endswith("/v1"):
                normalized_base = normalized_base + "/v1"
            client = OpenAI(api_key=env_api_key, base_url=normalized_base)
        else:
            client = OpenAI(api_key=env_api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            **({"response_format": {"type": "json_object"}} if response_json else {}),
        )
        content = resp.choices[0].message.content
    except Exception:
        import requests
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        env_base = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE")
            or os.getenv("BASE_URL")
            or "https://api.openai.com"
        )
        normalized_base = env_base.rstrip("/")
        # Avoid duplicating /v1 if it's already present in base
        if normalized_base.endswith("/v1"):
            url = f"{normalized_base}/chat/completions"
        else:
            url = f"{normalized_base}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_json:
            payload["response_format"] = {"type": "json_object"}
        r = requests.post(url, headers=headers, data=json.dumps(payload))
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]

    # Parse JSON content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Attempt to extract JSON substring
        import re
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            return json.loads(m.group(0))
        raise


def call_gpt4o_until_success(
    messages: List[Dict[str, Any]],
    model: str = "gpt-4o",
    temperature: float = 0.2,
    base_url: Optional[str] = None,
    max_tokens: Optional[int] = None,
    response_json: bool = False,
    expected_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Retry calling until a valid JSON with expected keys is returned.

    Implements exponential backoff capped at 60 seconds between retries.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            review = call_gpt4o(
                messages,
                model=model,
                temperature=temperature,
                base_url=base_url,
                max_tokens=max_tokens,
                response_json=response_json,
            )
            if not isinstance(review, dict):
                raise ValueError("Response is not JSON object.")
            keys = expected_keys or ["chosen_step_reviews", "rejected_step_reviews"]
            ok = True
            for k in keys:
                if k not in review:
                    ok = False
                    break
            if ok:
                return review
            raise ValueError(f"Response missing expected keys: {keys}")
        except Exception as e:
            wait = min(60.0, float(2 ** min(attempt, 6)))
            try:
                print(f"[judge-retry] attempt={attempt} wait={wait}s error={e}")
            except Exception:
                pass
            time.sleep(wait)


def process_jsonl(
    path: str,
    out_path: str,
    limit: Optional[int],
    sleep_s: float,
    model: str,
    temperature: float,
    base_url: Optional[str],
    image_root: Optional[str],
    criteria_catalog: Dict[str, List[str]],
    embedder: BGEEmbedder,
    min_criteria_per_step: int,
    max_criteria_per_step: int,
    parent_distance_threshold: float,
    compact: bool = False,
    max_tokens: Optional[int] = None,
    truncate_step_chars: Optional[int] = None,
) -> None:
    total = 0
    written = 0

    # 若指定了 limit，则先随机抽样将要处理的行索引（仅在 JSON 可解析的行中抽样）
    selected_indices: Optional[set[int]] = None
    total_estimated: Optional[int] = None
    if limit is not None:
        valid_indices: list[int] = []
        try:
            with open(path, "r", encoding="utf-8") as f_count:
                for i, ln in enumerate(f_count):
                    s = ln.strip()
                    if not s:
                        continue
                    try:
                        json.loads(s)
                    except json.JSONDecodeError:
                        continue
                    valid_indices.append(i)
        except Exception:
            valid_indices = []

        if not valid_indices:
            print("未在输入文件中找到有效的 JSON 行，退出。")
            return

        if limit >= len(valid_indices):
            selected_indices = set(valid_indices)
        else:
            selected_indices = set(random.sample(valid_indices, limit))

        total_estimated = len(selected_indices)
    else:
        # 未指定 limit 的情况下，估算总数用于进度条
        try:
            with open(path, "r", encoding="utf-8") as f_count:
                total_estimated = sum(1 for ln in f_count if ln.strip())
        except Exception:
            total_estimated = None

    with open(path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for idx, line in enumerate(tqdm(f_in, total=total_estimated, desc="处理进度", unit="条")):
            # 若启用随机抽样，仅处理被选中的行
            if selected_indices is not None and idx not in selected_indices:
                continue

            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            _id = obj.get("id")
            query = obj.get("query") or obj.get("question") or obj.get("prompt")
            chosen_raw = obj.get("chosen") or obj.get("accept") or obj.get("preferred")
            rejected_raw = obj.get("rejected") or obj.get("reject") or obj.get("dispreferred")

            chosen_steps = extract_steps(chosen_raw)
            rejected_steps = extract_steps(rejected_raw)

            # 紧凑模式下截断每个步骤文本，进一步减少输入
            if compact and truncate_step_chars and truncate_step_chars > 0:
                chosen_steps = [s[:truncate_step_chars] for s in chosen_steps]
                rejected_steps = [s[:truncate_step_chars] for s in rejected_steps]

            images = _extract_images(obj)
            image_parts = [_to_image_part(img, image_root=image_root) for img in images]
            chosen_step_payloads = build_step_payloads(
                chosen_steps,
                criteria_catalog=criteria_catalog,
                embedder=embedder,
                model=model,
                temperature=temperature,
                base_url=base_url,
                max_tokens=max_tokens,
                max_criteria_per_step=max_criteria_per_step,
                parent_distance_threshold=parent_distance_threshold,
            )
            rejected_step_payloads = build_step_payloads(
                rejected_steps,
                criteria_catalog=criteria_catalog,
                embedder=embedder,
                model=model,
                temperature=temperature,
                base_url=base_url,
                max_tokens=max_tokens,
                max_criteria_per_step=max_criteria_per_step,
                parent_distance_threshold=parent_distance_threshold,
            )
            messages = build_messages(
                query,
                chosen_step_payloads,
                rejected_step_payloads,
                image_parts=image_parts,
                compact=compact,
                min_criteria_per_step=min_criteria_per_step,
                max_criteria_per_step=max_criteria_per_step,
            )

            review = call_gpt4o_until_success(
                messages,
                model=model,
                temperature=temperature,
                base_url=base_url,
                max_tokens=max_tokens,
                response_json=True,
            )

            chosen_reviews = _normalize_step_reviews(
                review.get("chosen_step_reviews", []),
                fallback_steps=chosen_step_payloads,
                min_criteria_per_step=min_criteria_per_step,
                max_criteria_per_step=max_criteria_per_step,
                polarity_key="why_good",
            )
            rejected_reviews = _normalize_step_reviews(
                review.get("rejected_step_reviews", []),
                fallback_steps=rejected_step_payloads,
                min_criteria_per_step=min_criteria_per_step,
                max_criteria_per_step=max_criteria_per_step,
                polarity_key="why_bad",
            )

            out_obj = {
                "id": _id,
                "model": model,
                "chosen_step_reviews": chosen_reviews,
                "rejected_step_reviews": rejected_reviews,
                "meta": {
                    "query_present": bool(query),
                    "chosen_step_count": len(chosen_steps),
                    "rejected_step_count": len(rejected_steps),
                    "image_count": len(images),
                    "min_criteria_per_step": min_criteria_per_step,
                    "max_criteria_per_step": max_criteria_per_step,
                    "criteria_catalog_loaded": bool(criteria_catalog),
                },
            }
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            # 每完成一条即保存：刷新缓冲并尝试 fsync 落盘
            try:
                f_out.flush()
                os.fsync(f_out.fileno())
            except Exception:
                # 某些文件系统可能不支持 fsync，忽略以不中断流程
                pass
            written += 1
            total += 1

            # 若为随机抽样，按抽样数量终止；否则按传入 limit 终止
            if selected_indices is not None and written >= len(selected_indices):
                break
            if limit is None and sleep_s > 0:
                time.sleep(sleep_s)
            elif sleep_s > 0:
                time.sleep(sleep_s)

    print(f"Processed {written} samples (read {total} lines). Output -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Use a VLM judge to score step-wise reasoning for chosen vs rejected.")
    parser.add_argument("--path", default="datasets/dpo_data.jsonl", help="Input JSONL path")
    parser.add_argument("--output", default="tools/dpo_step_judgments.jsonl", help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to process")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-32B-Instruct", help="Model to use (paper default: Qwen2.5-VL-32B-Instruct)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    parser.add_argument("--base-url", type=str, default=None, help="Custom API base url, e.g., https://api.openai-proxy.org or https://api.openai-proxy.org/v1")
    parser.add_argument("--image-root", type=str, default=None, help="Base directory to resolve local image paths in dataset items")
    parser.add_argument("--compact", action="store_true", help="启用紧凑模式：极简提示、仅首图、why_*≤8词、JSON 格式输出")
    parser.add_argument("--max-tokens", type=int, default=None, help="限制模型最大输出 token，用于减少冗长输出")
    parser.add_argument("--truncate-step-chars", type=int, default=None, help="紧凑模式下每个步骤文本的最大字符数")
    parser.add_argument("--criteria-pairs", type=str, default=os.path.join("tools", "fgrs_pipeline", "output", "reward_hierarchy_named_pairs.txt"), help="Path to parent-child criteria pairs text file")
    parser.add_argument("--min-criteria-per-step", type=int, default=3, help="Minimum criteria selected per step")
    parser.add_argument("--max-criteria-per-step", type=int, default=5, help="Maximum criteria selected per step (paper: <=5)")
    parser.add_argument("--parent-distance-threshold", type=float, default=0.5, help="Filter candidate criteria by parent distance: keep if distance <= threshold")
    parser.add_argument("--bge-model", type=str, default="BAAI/bge-en-icl", help="Embedding model used for parent/criteria cosine distance")
    parser.add_argument("--bge-batch-size", type=int, default=32, help="Batch size for BGE embedding")
    parser.add_argument("--bge-device", type=str, default=None, help="Embedding device, e.g., cuda or cpu")
    args = parser.parse_args()

    # Basic guard for API key when using fallback
    if not os.getenv("OPENAI_API_KEY"):
        print("警告：未检测到 OPENAI_API_KEY 环境变量。若未安装官方 openai 客户端，将无法调用接口。")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # 紧凑模式默认采用更低温度与合理截断
    if args.compact and (args.temperature is None or args.temperature > 0.0):
        args.temperature = 0.0

    args.min_criteria_per_step = max(1, int(args.min_criteria_per_step))
    args.max_criteria_per_step = max(args.min_criteria_per_step, int(args.max_criteria_per_step))
    if args.max_criteria_per_step > 5:
        args.max_criteria_per_step = 5

    criteria_catalog = load_criteria_pairs(args.criteria_pairs)
    if not criteria_catalog:
        raise SystemExit(f"criteria catalog missing/empty: {args.criteria_pairs}. This is required in paper-style mode.")
    print(f"Loaded criteria catalog: {len(criteria_catalog)} parents from {args.criteria_pairs}")
    embedder = BGEEmbedder(args.bge_model, batch_size=args.bge_batch_size, device=args.bge_device)

    process_jsonl(
        path=args.path,
        out_path=args.output,
        limit=args.limit,
        sleep_s=args.sleep,
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        image_root=args.image_root,
        criteria_catalog=criteria_catalog,
        embedder=embedder,
        min_criteria_per_step=args.min_criteria_per_step,
        max_criteria_per_step=args.max_criteria_per_step,
        parent_distance_threshold=args.parent_distance_threshold,
        compact=args.compact,
        max_tokens=args.max_tokens,
        truncate_step_chars=args.truncate_step_chars if args.truncate_step_chars is not None else (200 if args.compact else None),
    )


if __name__ == "__main__":
    main()
