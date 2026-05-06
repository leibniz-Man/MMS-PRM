import argparse
import json
import os
import time
import base64
import mimetypes
import random
from typing import List, Dict, Any, Optional
try:
    # 更智能的进度条（在终端/Notebook 都有良好显示）
    from tqdm.auto import tqdm
except Exception:
    # 若未安装 tqdm，则优雅降级为普通迭代
    def tqdm(iterable, total=None, desc=None, unit=None):
        return iterable
OPENAI_BASE_URL="https://api.openai-proxy.org/v1"
OPENAI_API_KEY="sk-pfQFCXggVvHdL3R1O3k5chxbva6l7Ai4v5SkkPdKYDw2t4DY"
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


def build_messages(
    query: Optional[str],
    chosen_steps: List[str],
    rejected_steps: List[str],
    image_parts: Optional[List[Dict[str, Any]]] = None,
    compact: bool = False,
) -> List[Dict[str, Any]]:
    # 紧凑模式下极简 system 指令，减少输入 token
    if compact:
        system = (
            "You are a step-wise judge. Return JSON only. "
            "For each step, include why_* (≤8 words) and reward_categories (≤3 short nouns). "
            "No extra text, no scores."
        )
    else:
        system = (
            "You are a strict, step-wise reasoning judge. "
            "Given a query and two step-by-step reasoning chains (chosen vs rejected), assess EACH step. "
            "For every step, produce BOTH: (1) a concise textual rationale (why_good or why_bad) and (2) a list of invented reward categories (strings). "
            "Focus categories on vision-related aspects when images are involved. "
            "You MAY invent new categories as needed; DO NOT include any numeric scores. Return ONLY JSON."
        )

    # 将用户负载压缩为必要信息，避免冗长 schema 示例
    user_payload = {
        "query": query or "",
        "chosen_steps": chosen_steps,
        "rejected_steps": rejected_steps,
        "keys": {
            "chosen_step_reviews": ["step_index", "step_text", "why_good", "reward_categories"],
            "rejected_step_reviews": ["step_index", "step_text", "why_bad", "reward_categories"],
        },
    }

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}]
    if image_parts:
        # 紧凑模式下仅附带首张图片，以减少输入
        if compact:
            first = next((p for p in image_parts if p), None)
            if first:
                user_content.append(first)
        else:
            user_content.extend([p for p in image_parts if p])

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


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
            if (
                isinstance(review, dict) and
                isinstance(review.get("chosen_step_reviews"), list) and
                isinstance(review.get("rejected_step_reviews"), list)
            ):
                return review
            raise ValueError("Response missing expected keys: chosen_step_reviews/rejected_step_reviews")
        except Exception as e:
            wait = min(60.0, float(2 ** min(attempt, 6)))
            try:
                print(f"[judge-retry] attempt={attempt} wait={wait}s error={e}")
            except Exception:
                pass
            time.sleep(wait)


def process_jsonl(path: str, out_path: str, limit: Optional[int], sleep_s: float, model: str, temperature: float, base_url: Optional[str], image_root: Optional[str], compact: bool = False, max_tokens: Optional[int] = None, truncate_step_chars: Optional[int] = None) -> None:
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
            messages = build_messages(query, chosen_steps, rejected_steps, image_parts=image_parts, compact=compact)

            review = call_gpt4o_until_success(
                messages,
                model=model,
                temperature=temperature,
                base_url=base_url,
                max_tokens=max_tokens,
                response_json=True if compact else False,
            )

            out_obj = {
                "id": _id,
                "model": model,
                "chosen_step_reviews": review.get("chosen_step_reviews", []),
                "rejected_step_reviews": review.get("rejected_step_reviews", []),
                "meta": {
                    "query_present": bool(query),
                    "chosen_step_count": len(chosen_steps),
                    "rejected_step_count": len(rejected_steps),
                    "image_count": len(images),
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
    parser = argparse.ArgumentParser(description="Use GPT-4o to judge step-wise reasoning for chosen vs rejected.")
    parser.add_argument("--path", default="datasets/dpo_data.jsonl", help="Input JSONL path")
    parser.add_argument("--output", default="tools/dpo_step_judgments.jsonl", help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to process")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use (e.g., gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    parser.add_argument("--base-url", type=str, default=None, help="Custom API base url, e.g., https://api.openai-proxy.org or https://api.openai-proxy.org/v1")
    parser.add_argument("--image-root", type=str, default=None, help="Base directory to resolve local image paths in dataset items")
    parser.add_argument("--compact", action="store_true", help="启用紧凑模式：极简提示、仅首图、why_*≤8词、JSON 格式输出")
    parser.add_argument("--max-tokens", type=int, default=None, help="限制模型最大输出 token，用于减少冗长输出")
    parser.add_argument("--truncate-step-chars", type=int, default=None, help="紧凑模式下每个步骤文本的最大字符数")
    args = parser.parse_args()

    # Basic guard for API key when using fallback
    if not os.getenv("OPENAI_API_KEY"):
        print("警告：未检测到 OPENAI_API_KEY 环境变量。若未安装官方 openai 客户端，将无法调用接口。")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # 紧凑模式默认采用更低温度与合理截断
    if args.compact and (args.temperature is None or args.temperature > 0.0):
        args.temperature = 0.0

    process_jsonl(
        path=args.path,
        out_path=args.output,
        limit=args.limit,
        sleep_s=args.sleep,
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        image_root=args.image_root,
        compact=args.compact,
        max_tokens=args.max_tokens,
        truncate_step_chars=args.truncate_step_chars if args.truncate_step_chars is not None else (200 if args.compact else None),
    )


if __name__ == "__main__":
    main()