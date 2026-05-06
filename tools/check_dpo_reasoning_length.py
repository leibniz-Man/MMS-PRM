import argparse
import json
import random
import re
from typing import Tuple, Optional, List, Dict, Any


def extract_reasoning(text: str) -> str:
    """Extract content between <|reasoning_start|> and <|reasoning_end|>.

    If markers are missing, fall back to the full text.
    """
    if not isinstance(text, str):
        return ""

    start_tag = "<|reasoning_start|>"
    end_tag = "<|reasoning_end|>"

    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx + len(start_tag): end_idx]
    return text


def extract_all_reasoning_blocks(text: str) -> List[str]:
    """Extract all reasoning blocks delimited by <|reasoning_start|> ... <|reasoning_end|>.

    Some entries may contain multiple reasoning segments separated by <|split_token|>.
    """
    if not isinstance(text, str) or not text:
        return []
    pattern = re.compile(r"<\|reasoning_start\|>(.*?)<\|reasoning_end\|>", re.DOTALL)
    return [m.strip() for m in pattern.findall(text)]


def extract_answer(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None
    m = re.search(r"<\|answer_start\|>(.*?)<\|answer_end\|>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def parse_steps(reasoning_block: str) -> List[Dict[str, Any]]:
    """Parse a single reasoning block into structured steps using step delimiters.

    Each step is within <|reasoning_step_start|> ... <|reasoning_step_end|> and may include
    name/thought/reflection subfields.
    """
    steps: List[Dict[str, Any]] = []
    if not reasoning_block:
        return steps

    # Find all step segments
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
        # Fallback: if no subfields, keep raw segment
        if not step["name"] and not step["thought"] and not step["reflection"]:
            step["raw"] = seg.strip()
        steps.append(step)

    return steps


def print_structured_reasoning(tag: str, text: Optional[str]) -> None:
    """Pretty-print reasoning for a field (chosen/rejected) in Chinese.

    Splits by reasoning blocks, then prints steps with name/thought/reflection.
    """
    print(f"【{tag}】")
    if not text:
        print("  无内容")
        return

    blocks = extract_all_reasoning_blocks(text)
    if not blocks:
        # If no explicit blocks, try to show the whole extracted reasoning
        fallback = extract_reasoning(text)
        print("  无显式分隔块，展示推理片段：")
        print("  ----")
        print(f"  {fallback.strip()[:1000]}")
        print("  ----")
        return

    for bi, block in enumerate(blocks, start=1):
        print(f"  推理块 {bi}：")
        steps = parse_steps(block)
        if not steps:
            print("    （无步骤分隔，原始内容）")
            print("    ----")
            print("    " + block.strip()[:1000].replace("\n", "\n    "))
            print("    ----")
        else:
            for si, st in enumerate(steps, start=1):
                print(f"    步骤 {si}：")
                if st.get("name"):
                    print(f"      名称：{st['name']}")
                if st.get("thought"):
                    print("      思路：")
                    print("        " + st["thought"].replace("\n", "\n        "))
                if st.get("reflection"):
                    print("      反思：")
                    print("        " + st["reflection"].replace("\n", "\n        "))
                if st.get("raw"):
                    print("      原始片段：")
                    print("        " + st["raw"].replace("\n", "\n        "))
        print("  ----")


def measure_lengths(text: str) -> Tuple[int, int]:
    """Return (char_len, token_len) for the given text.

    token length is computed by simple whitespace splitting.
    """
    if text is None:
        return 0, 0
    s = str(text)
    char_len = len(s)
    token_len = len(s.split())
    return char_len, token_len


def process_file(path: str, limit: Optional[int] = None, show_all: bool = False) -> None:
    total = 0
    same_char = 0
    same_token = 0
    diffs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                diffs.append({
                    "id": None,
                    "error": "Invalid JSON",
                })
                continue

            _id = obj.get("id")
            chosen = obj.get("chosen")
            rejected = obj.get("rejected")

            chosen_reasoning = extract_reasoning(chosen)
            rejected_reasoning = extract_reasoning(rejected)

            c_char, c_tok = measure_lengths(chosen_reasoning)
            r_char, r_tok = measure_lengths(rejected_reasoning)

            char_equal = c_char == r_char
            tok_equal = c_tok == r_tok

            if char_equal:
                same_char += 1
            if tok_equal:
                same_token += 1

            # Record mismatch or all entries depending on flags
            if show_all or not (char_equal and tok_equal):
                # Only store limited number if limit provided
                if limit is None or len(diffs) < limit:
                    diffs.append({
                        "id": _id,
                        "char_len": {"chosen": c_char, "rejected": r_char, "equal": char_equal},
                        "token_len": {"chosen": c_tok, "rejected": r_tok, "equal": tok_equal},
                    })

    # Summary
    print("摘要：")
    print(f"  总行数: {total}")
    print(f"  字符长度相同: {same_char} ({same_char/total*100:.2f}% if total>0 else 0)")
    print(f"  分词长度相同: {same_token} ({same_token/total*100:.2f}% if total>0 else 0)")

    if diffs:
        print("\n样例：")
        for item in diffs:
            print(json.dumps(item, ensure_ascii=False))
    else:
        print("\n在当前展示条件下未发现不一致。")


def sample_and_show(path: str, n: int, seed: Optional[int] = None) -> None:
    """Randomly sample n entries and show structured reasoning for chosen/rejected."""
    # Read all lines as JSON objects
    objs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not objs:
        print("文件为空或无法解析为JSON对象。")
        return

    if seed is not None:
        random.seed(seed)

    k = min(n, len(objs))
    samples = random.sample(objs, k)

    for i, obj in enumerate(samples, start=1):
        _id = obj.get("id")
        print("==============================")
        print(f"样本 {i}/{k}  ID: {_id}")
        print_structured_reasoning("chosen", obj.get("chosen"))
        print_structured_reasoning("rejected", obj.get("rejected"))
        ch_ans = extract_answer(obj.get("chosen"))
        rj_ans = extract_answer(obj.get("rejected"))
        if ch_ans is not None or rj_ans is not None:
            print("答案片段：")
            print(f"  chosen: {ch_ans}")
            print(f"  rejected: {rj_ans}")
        print("==============================\n")


def main():
    parser = argparse.ArgumentParser(description="Check if chosen and rejected reasoning lengths are equal.")
    parser.add_argument(
        "--path",
        default="datasets/dpo_data.jsonl",
        help="Path to the JSONL file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit number of sample outputs (mismatches or all if --show-all)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show samples for all lines instead of only mismatches",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Randomly sample N entries and display structured reasoning",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random sampling seed for reproducibility",
    )

    args = parser.parse_args()
    if args.sample and args.sample > 0:
        sample_and_show(args.path, n=args.sample, seed=args.seed)
    else:
        process_file(args.path, limit=args.limit, show_all=args.show_all)


if __name__ == "__main__":
    main()