import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, total=None, desc=None, unit=None):
        return iterable

def call_gpt4o(messages: List[Dict[str, Any]], model: str = "gpt-4o", temperature: float = 0.0, base_url: Optional[str] = None, max_tokens: Optional[int] = None, response_json: bool = True) -> Dict[str, Any]:
    try:
        from openai import OpenAI
        env_base = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE") or os.getenv("BASE_URL")
        env_api_key = os.getenv("OPENAI_API_KEY")
        if env_base:
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
        env_base = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE") or os.getenv("BASE_URL") or "https://api.openai.com"
        normalized_base = env_base.rstrip("/")
        url = f"{normalized_base}/chat/completions" if normalized_base.endswith("/v1") else f"{normalized_base}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_json:
            payload["response_format"] = {"type": "json_object"}
        r = requests.post(url, headers=headers, data=json.dumps(payload))
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            return json.loads(m.group(0))
        raise

def call_gpt4o_until_success(messages: List[Dict[str, Any]], model: str = "gpt-4o", temperature: float = 0.0, base_url: Optional[str] = None, max_tokens: Optional[int] = None, response_json: bool = True) -> Dict[str, Any]:
    attempt = 0
    while True:
        attempt += 1
        try:
            obj = call_gpt4o(messages, model=model, temperature=temperature, base_url=base_url, max_tokens=max_tokens, response_json=response_json)
            if isinstance(obj, dict) and obj.get("name"):
                return obj
            raise ValueError("Invalid response")
        except Exception as e:
            wait = min(60.0, float(2 ** min(attempt, 6)))
            try:
                print(f"[name-retry] attempt={attempt} wait={wait}s error={e}")
            except Exception:
                pass
            time.sleep(wait)

def build_messages(children_names: List[str]) -> List[Dict[str, Any]]:
    system = (
        "You are a cluster naming assistant for visual reasoning reward categories."
        " Summarize the common theme across the given child category names into a short English cluster name (≤5 words)."
        " Prefer precise, academic-style vocabulary such as Logical Consistency, Mathematical Manipulation, Conceptual Understanding, Reasoning Accuracy, Visual Recognition Correctness, Temporal Reasoning, Spatial Understanding."
        " Output STRICT JSON only: {\"name\": \"<cluster name>\"}."
    )
    user_payload = {"children_names": children_names}
    return [{"role": "system", "content": system}, {"role": "user", "content": [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}]}]

def build_filter_messages(name: str, children_names: List[str]) -> List[Dict[str, Any]]:
    system = (
        "You are a reward category filtering assistant."
        " Given a cluster name and its candidate child category names, deduplicate and select the most useful reward children."
        " Useful criteria: semantically precise, informative, evaluable/observable, non-redundant, and not overly generic."
        " Output STRICT JSON only: {\"name\": \"<name>\", \"children\": [\"<filtered child>\", ...]}."
    )
    user_payload = {"name": name, "children_names": children_names}
    return [{"role": "system", "content": system}, {"role": "user", "content": [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}]}]

def _collect_leaf_names(node: Any, acc: List[str]) -> None:
    if isinstance(node, dict):
        if "children" in node and isinstance(node["children"], list):
            for ch in node["children"]:
                _collect_leaf_names(ch, acc)
        else:
            n = node.get("name")
            if isinstance(n, str):
                acc.append(n)
    elif isinstance(node, list):
        for ch in node:
            _collect_leaf_names(ch, acc)
    elif isinstance(node, str):
        acc.append(node)

def process_tree(inp: Dict[str, Any], model: str, temperature: float, base_url: Optional[str], max_tokens: Optional[int]) -> Dict[str, Any]:
    root = inp.get("root")
    if not isinstance(root, dict):
        return inp
    children = root.get("children")
    if not isinstance(children, list):
        return inp
    out_children: List[Dict[str, Any]] = []
    for cluster in tqdm(children, total=len(children), desc="Naming clusters", unit="cluster"):
        if not isinstance(cluster, dict):
            out_children.append(cluster)
            continue
        cluster_children = cluster.get("children", [])
        if isinstance(cluster_children, list) and len(cluster_children) == 1:
            out_children.append(cluster)
            continue
        names: List[str] = []
        _collect_leaf_names(cluster_children, names)
        if not names:
            out_children.append(cluster)
            continue
        messages = build_messages(sorted(set(names))[:200])
        review = call_gpt4o_until_success(messages, model=model, temperature=temperature, base_url=base_url, max_tokens=max_tokens, response_json=True)
        new_cluster = dict(cluster)
        new_cluster["orig_name"] = new_cluster.get("name")
        new_cluster["name"] = review.get("name")
        if "keywords" in new_cluster:
            del new_cluster["keywords"]
        out_children.append(new_cluster)
    out_root = dict(root)
    out_root["children"] = out_children
    out_obj = dict(inp)
    out_obj["root"] = out_root
    return out_obj

def main():
    parser = argparse.ArgumentParser(description="Summarize cluster children and name clusters via GPT-4o")
    parser.add_argument("--input", type=str, default=os.path.join("tools", "fgrs_pipeline", "output", "reward_hierarchy.json"))
    parser.add_argument("--output", type=str, default=os.path.join("tools", "fgrs_pipeline", "output", "reward_hierarchy_named.json"))
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        print("警告：未检测到 OPENAI_API_KEY 环境变量。若未安装官方 openai 客户端，将无法调用接口。")
    with open(args.input, "r", encoding="utf-8") as f:
        tree = json.load(f)
    named_tree = process_tree(tree, model=args.model, temperature=args.temperature, base_url=args.base_url, max_tokens=args.max_tokens)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(named_tree, f, ensure_ascii=False, indent=2)
    print(f"Wrote named hierarchy to: {args.output}")

    pairs_path = os.path.splitext(args.output)[0] + "_pairs.txt"
    lines: List[str] = []
    root = named_tree.get("root") if isinstance(named_tree, dict) else None
    children = root.get("children") if isinstance(root, dict) else None
    if isinstance(children, list):
        for cluster in children:
            if not isinstance(cluster, dict):
                continue
            if "orig_name" not in cluster:
                continue
            cluster_children = cluster.get("children", [])
            names: List[str] = []
            _collect_leaf_names(cluster_children, names)
            if names and isinstance(cluster.get("name"), str):
                line = f"{cluster.get('name')}：" + "、".join(sorted(set(names)))
                lines.append(line)
    with open(pairs_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote named pairs to: {pairs_path}")

    merged: Dict[str, List[str]] = {}
    for ln in lines:
        try:
            name, childs = ln.split("：", 1)
        except ValueError:
            continue
        items = [x.strip() for x in childs.split("、") if x.strip()]
        prev = merged.get(name) or []
        s = set(prev)
        for it in items:
            if it not in s:
                prev.append(it)
                s.add(it)
        merged[name] = prev

    filtered_lines: List[str] = []
    use_gpt = bool(os.getenv("OPENAI_API_KEY"))
    for name, childs in sorted(merged.items(), key=lambda kv: kv[0]):
        final_children: List[str]
        if use_gpt:
            msgs = build_filter_messages(name, childs[:200])
            resp = call_gpt4o_until_success(msgs, model=args.model, temperature=args.temperature, base_url=args.base_url, max_tokens=args.max_tokens, response_json=True)
            c = resp.get("children")
            if isinstance(c, list) and c:
                final_children = [str(x) for x in c if isinstance(x, str)]
            else:
                final_children = childs
        else:
            final_children = childs
        filtered_lines.append(f"{name}：" + "、".join(sorted(set(final_children))))

    with open(pairs_path, "w", encoding="utf-8") as f:
        f.write("\n".join(filtered_lines))
    print(f"Merged and filtered pairs written to: {pairs_path}")

if __name__ == "__main__":
    main()