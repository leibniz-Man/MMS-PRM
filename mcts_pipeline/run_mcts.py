import argparse
import json
import os
import random
from typing import List, Optional, Tuple

import torch
from PIL import Image

from internvl.conversation import get_conv_template
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess


class MCTSNode:
    def __init__(self, reasoning_prefix: str, depth: int, parent: Optional["MCTSNode"] = None):
        self.reasoning_prefix = reasoning_prefix
        self.depth = depth
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value = 0.0
        self.fully_expanded = False

    def add_child(self, child: "MCTSNode"):
        self.children.append(child)


def build_pixel_values(image_path: Optional[str], image_size: int, dynamic_image_size: bool, use_thumbnail: bool, max_num: int) -> Tuple[Optional[torch.Tensor], List[int]]:
    if not image_path:
        return None, []
    image = Image.open(image_path).convert("RGB")
    if dynamic_image_size:
        images = dynamic_preprocess(image, image_size=image_size, use_thumbnail=use_thumbnail, max_num=max_num)
    else:
        images = [image]
    transform = build_transform(is_train=False, input_size=image_size)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    num_patches_list = [pixel_values.shape[0]]
    return pixel_values, num_patches_list


def build_query(model, tokenizer, question: str, reasoning_prefix: str, pixel_values: Optional[torch.Tensor], num_patches_list: Optional[List[int]]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if pixel_values is not None and "<image>" not in question:
        question = "<image>\n" + question

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], reasoning_prefix)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

    for num_patches in num_patches_list:
        image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
        query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    return input_ids, attention_mask, eos_token_id


def generate_next_step(model, tokenizer, pixel_values, input_ids, attention_mask, eos_token_id, max_new_tokens: int, temperature: float) -> str:
    generation_config = dict(
        num_beams=1,
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        eos_token_id=eos_token_id,
    )
    outputs = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=None,
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    sep_str = get_conv_template(model.template).sep.strip()
    response = response.split(sep_str)[0].strip()
    return response


def extract_single_step(delta: str) -> str:
    end_tag = "<|reasoning_step_end|>"
    proceed_tag = "<|reasoning_proceed|>"
    end_pos = delta.find(end_tag)
    if end_pos != -1:
        return delta[: end_pos + len(end_tag)]
    proceed_pos = delta.find(proceed_tag)
    if proceed_pos != -1:
        return delta[: proceed_pos]
    return delta


def mcts_search(model, tokenizer, question: str, pixel_values: Optional[torch.Tensor], num_patches_list: Optional[List[int]],
                max_depth: int, branching_factor: int, simulations: int, max_new_tokens: int, temperature: float) -> str:
    root_prefix = "<|reasoning_start|>\n"
    root = MCTSNode(reasoning_prefix=root_prefix, depth=0, parent=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pixel_values is not None:
        pixel_values = pixel_values.to(torch.bfloat16).to(device)

    def ucb(node: MCTSNode, parent_visits: int, c: float = 1.4) -> float:
        if node.visits == 0:
            return float("inf")
        return node.value / node.visits + c * (math.log(parent_visits) / node.visits) ** 0.5

    import math

    for _ in range(simulations):
        node = root
        path = [node]

        while node.depth < max_depth:
            if len(node.children) < branching_factor:
                input_ids, attention_mask, eos_id = build_query(model, tokenizer, question, node.reasoning_prefix, pixel_values, num_patches_list)
                delta = generate_next_step(model, tokenizer, pixel_values, input_ids, attention_mask, eos_id, max_new_tokens, temperature)
                step_text = extract_single_step(delta)
                next_prefix = node.reasoning_prefix + step_text + "\n<|reasoning_proceed|>\n"
                child = MCTSNode(reasoning_prefix=next_prefix, depth=node.depth + 1, parent=node)
                node.add_child(child)
                node = child
                path.append(node)
            else:
                best = max(node.children, key=lambda ch: ucb(ch, node.visits if node.visits > 0 else 1))
                node = best
                path.append(node)

        reward = len(path) - 1
        for n in path:
            n.visits += 1
            n.value += reward

    best_leaf = max((n for n in collect_leaves(root)), key=lambda x: x.visits)
    final_reasoning = best_leaf.reasoning_prefix + "<|reasoning_end|>"
    return final_reasoning


def collect_leaves(root: MCTSNode) -> List[MCTSNode]:
    out = []
    stack = [root]
    while stack:
        n = stack.pop()
        if not n.children:
            out.append(n)
        else:
            for ch in n.children:
                stack.append(ch)
    return out


def parse_item(line: str) -> Tuple[str, Optional[str]]:
    item = json.loads(line)
    q = item.get("question")
    img = item.get("image")
    if q is None and "conversations" in item:
        for m in item["conversations"]:
            if m.get("from") == "human":
                q = m.get("value")
                break
    return q, img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="huggingface/sft_mpo_5e_6")
    parser.add_argument("--input", type=str, default="datasets/dpo_data.jsonl")
    parser.add_argument("--output", type=str, default="mcts_pipeline/mcts_outputs.jsonl")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--use-thumbnail", action="store_true")
    parser.add_argument("--max-num", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--branching", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    class A:
        pass

    a = A()
    a.checkpoint = args.checkpoint
    a.load_in_8bit = args.load_in_8bit
    a.load_in_4bit = args.load_in_4bit
    a.auto = args.auto
    model, tokenizer = load_model_and_tokenizer(a)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = open(args.output, "w")

    lines = open(args.input).readlines()
    random.seed(0)
    for line in lines:
        question, image = parse_item(line)
        if question is None:
            continue
        pixel_values, num_patches_list = build_pixel_values(image, args.image_size, args.dynamic, args.use_thumbnail, args.max_num)
        reasoning = mcts_search(
            model=model,
            tokenizer=tokenizer,
            question=question,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            max_depth=args.max_depth,
            branching_factor=args.branching,
            simulations=args.simulations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        writer.write(json.dumps({"question": question, "reasoning": reasoning}, ensure_ascii=False) + "\n")

    writer.close()


if __name__ == "__main__":
    main()