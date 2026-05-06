import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from internvl.conversation import get_conv_template
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess


DEFAULT_REWARD_DIMENSIONS = [
    "Visual Grounding Correctness",
    "Logical Consistency",
    "Mathematical or Semantic Correctness",
    "Stepwise Coherence",
    "Conciseness",
]


class MCTSNode:
    """A node is one partial reasoning trajectory."""

    _next_id = 0

    def __init__(
        self,
        reasoning_prefix: str,
        depth: int,
        parent: Optional["MCTSNode"] = None,
        step_text: str = "",
        reward_vector: Optional[List[float]] = None,
        raw_scores: Optional[Dict[str, float]] = None,
        terminal: bool = False,
    ):
        self.node_id = MCTSNode._next_id
        MCTSNode._next_id += 1

        self.reasoning_prefix = reasoning_prefix
        self.depth = depth
        self.parent = parent
        self.step_text = step_text
        self.children: List["MCTSNode"] = []

        # MCTS statistics.
        self.visits = 0
        self.value_sum = 0.0

        # Multi-dimensional step reward in [0, 1].
        self.reward_vector = reward_vector
        self.raw_scores = raw_scores or {}
        self.terminal = terminal

    @property
    def value_mean(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def add_child(self, child: "MCTSNode") -> None:
        self.children.append(child)


def build_pixel_values(
    image_path: Optional[str],
    image_size: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
    max_num: int,
) -> Tuple[Optional[torch.Tensor], List[int]]:
    if not image_path:
        return None, []

    image = Image.open(image_path).convert("RGB")
    if dynamic_image_size:
        images = dynamic_preprocess(
            image,
            image_size=image_size,
            use_thumbnail=use_thumbnail,
            max_num=max_num,
        )
    else:
        images = [image]

    transform = build_transform(is_train=False, input_size=image_size)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    num_patches_list = [pixel_values.shape[0]]
    return pixel_values, num_patches_list


def build_query(
    model,
    tokenizer,
    question: str,
    reasoning_prefix: str,
    pixel_values: Optional[torch.Tensor],
    num_patches_list: Optional[List[int]],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if pixel_values is not None and "<image>" not in question:
        question = "<image>\n" + question

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    sep_token = template.sep.strip()
    eos_token_id = tokenizer.convert_tokens_to_ids(sep_token) if sep_token else tokenizer.eos_token_id
    if eos_token_id is None or eos_token_id < 0:
        eos_token_id = tokenizer.eos_token_id

    template.append_message(template.roles[0], question)
    if reasoning_prefix:
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


def generate_text(
    model,
    tokenizer,
    pixel_values,
    input_ids,
    attention_mask,
    eos_token_id,
    max_new_tokens: int,
    temperature: float,
) -> str:
    generation_config = dict(
        num_beams=1,
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        eos_token_id=eos_token_id,
    )

    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=None,
    )
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values

    with torch.inference_mode():
        try:
            outputs = model.generate(**kwargs, generation_config=generation_config)
        except TypeError:
            # Some HF-style generate implementations expect generation kwargs directly.
            kwargs.pop("output_hidden_states", None)
            outputs = model.generate(**kwargs, **generation_config)

    # HF generate usually returns prompt + continuation. InternVL variants may return only
    # continuation. Slicing is safe when the full sequence is returned.
    if hasattr(outputs, "shape") and outputs.shape[-1] > input_ids.shape[-1]:
        gen_ids = outputs[:, input_ids.shape[-1]:]
    else:
        gen_ids = outputs

    response = tokenizer.batch_decode(gen_ids, skip_special_tokens=False)[0]
    sep_str = get_conv_template(model.template).sep.strip()
    if sep_str:
        response = response.split(sep_str)[0]
    return response.strip()


def generate_next_step(
    model,
    tokenizer,
    pixel_values,
    input_ids,
    attention_mask,
    eos_token_id,
    max_new_tokens: int,
    temperature: float,
) -> str:
    response = generate_text(
        model=model,
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return extract_single_step(response)


def extract_single_step(delta: str) -> str:
    """Keep exactly one reasoning step."""

    end_tag = "<|reasoning_step_end|>"
    proceed_tag = "<|reasoning_proceed|>"
    reasoning_end_tag = "<|reasoning_end|>"

    reasoning_end_pos = delta.find(reasoning_end_tag)
    if reasoning_end_pos != -1:
        return delta[: reasoning_end_pos + len(reasoning_end_tag)].strip()

    end_pos = delta.find(end_tag)
    if end_pos != -1:
        return delta[: end_pos + len(end_tag)].strip()

    proceed_pos = delta.find(proceed_tag)
    if proceed_pos != -1:
        return delta[:proceed_pos].strip()

    # Conservative fallback: one paragraph / short completion as a single step.
    lines = [ln for ln in delta.splitlines() if ln.strip()]
    if len(lines) > 1:
        return lines[0].strip()
    return delta.strip()


def is_terminal_step(step_text: str) -> bool:
    lower = step_text.lower()
    return (
        "<|reasoning_end|>" in step_text
        or "final answer" in lower
        or "the answer is" in lower
    )


def normalize_raw_score(raw: float) -> float:
    # Paper: raw score s in [1, 10] is mapped to (s - 1) / 9.
    clipped = max(1.0, min(10.0, float(raw)))
    return (clipped - 1.0) / 9.0


def augmented_chebyshev(
    reward_vector: List[float],
    ideal_point: List[float],
    rho: float,
) -> float:
    """g_aug(v; z*) = - max_j(z*_j - v_j) - rho * sum_j(z*_j - v_j)."""

    gaps = [z - v for z, v in zip(ideal_point, reward_vector)]
    return -max(gaps) - rho * sum(gaps)


def update_ideal_point(
    ideal_point: List[float],
    reward_vector: List[float],
    ideal_lambda: float,
) -> List[float]:
    """z*_j <- (1-lambda) z*_j + lambda v_j."""

    return [
        (1.0 - ideal_lambda) * z + ideal_lambda * v
        for z, v in zip(ideal_point, reward_vector)
    ]


def trajectory_nodes(node: MCTSNode) -> List[MCTSNode]:
    out: List[MCTSNode] = []
    cur: Optional[MCTSNode] = node
    while cur is not None:
        out.append(cur)
        cur = cur.parent
    return list(reversed(out))


def trajectory_reward_vector(node: MCTSNode, num_dims: int) -> List[float]:
    step_vectors = [
        n.reward_vector
        for n in trajectory_nodes(node)
        if n.reward_vector is not None
    ]
    if not step_vectors:
        return [0.5] * num_dims

    return [
        sum(vec[j] for vec in step_vectors) / len(step_vectors)
        for j in range(num_dims)
    ]


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^a-z0-9.\-/%]", "", text)
    return text


def answer_reward(reasoning: str, answer: Optional[str]) -> float:
    """Simple exact/contains reward. Replace this with task-specific evaluation when available."""

    if answer is None or str(answer).strip() == "":
        return 0.0
    gold = normalize_answer(str(answer))
    pred_context = normalize_answer(reasoning[-2000:])
    if not gold:
        return 0.0
    return 1.0 if gold in pred_context else 0.0


class StepRewardScorer:
    """Step-level multi-dimensional reward scorer.

    `backend=self` uses the currently loaded VLM as a strict step-level judge.
    For paper-faithful reproduction, point this class to the same judge model
    used in the paper, e.g. Qwen2.5-VL-32B-Instruct, or serve it behind a
    compatible wrapper and replace `score_with_self_model`.
    """

    def __init__(
        self,
        dimensions: List[str],
        backend: str = "self",
        judge_max_new_tokens: int = 256,
        judge_temperature: float = 0.0,
    ):
        if not dimensions:
            raise ValueError("At least one reward dimension is required.")
        self.dimensions = dimensions
        self.backend = backend
        self.judge_max_new_tokens = judge_max_new_tokens
        self.judge_temperature = judge_temperature

    def score_step(
        self,
        model,
        tokenizer,
        question: str,
        reasoning_prefix: str,
        step_text: str,
        pixel_values: Optional[torch.Tensor],
        num_patches_list: Optional[List[int]],
    ) -> Tuple[Dict[str, float], List[float], str]:
        if self.backend == "heuristic":
            raw = self.score_with_heuristics(step_text)
            return raw, self.raw_to_vector(raw), ""

        if self.backend == "neutral":
            raw = {dim: 5.0 for dim in self.dimensions}
            return raw, self.raw_to_vector(raw), ""

        raw, judge_text = self.score_with_self_model(
            model=model,
            tokenizer=tokenizer,
            question=question,
            reasoning_prefix=reasoning_prefix,
            step_text=step_text,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
        )
        if not raw:
            # Do not silently crash long MCTS runs on one malformed judge output.
            raw = self.score_with_heuristics(step_text)
        return raw, self.raw_to_vector(raw), judge_text

    def score_with_self_model(
        self,
        model,
        tokenizer,
        question: str,
        reasoning_prefix: str,
        step_text: str,
        pixel_values: Optional[torch.Tensor],
        num_patches_list: Optional[List[int]],
    ) -> Tuple[Dict[str, float], str]:
        criteria = "\n".join(f"- {dim}" for dim in self.dimensions)
        prompt = f"""You are a strict step-level judge for math and diagram reasoning.

Given the original question, previous reasoning prefix, and one candidate next reasoning step,
evaluate the candidate step using ONLY criteria from the CANDIDATE LIST.

Calibration, use the full 1-10 range:
1-2: off-topic, hallucinated, or contradicts known facts.
3-4: clear error(s) or unjustified leap; harms solution.
5: neutral restatement/setup with no measurable progress.
6-7: minor progress with issue(s) or unverifiable claim(s).
8: solid, correct progress; improves solution measurably.
9-10: exceptional, critical progress; precise, verified, and necessary.

Guidelines:
- Prefer criteria that directly match the step content.
- Penalize fillers, repetition, vague statements, and unsupported claims.
- If any error/irrelevance/contradiction exists, reflect it in at least one low score.
- Avoid clustering all scores between 7-9 unless the step is truly excellent.
- Score EVERY criterion in the CANDIDATE LIST. If a criterion is not applicable, give 5.

CANDIDATE LIST:
{criteria}

ORIGINAL QUESTION:
{question}

PREVIOUS REASONING PREFIX:
{reasoning_prefix}

CANDIDATE NEXT STEP:
{step_text}

Final output requirements:
Produce a single final line listing the criteria and their scores.
Format strictly as: Criterion A: [[X]] - Criterion B: [[Y]] - ...
[[...]] MUST appear only in the final line; no extra text.
"""
        input_ids, attention_mask, eos_id = build_query(
            model=model,
            tokenizer=tokenizer,
            question=prompt,
            reasoning_prefix="",
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
        )
        judge_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=eos_id,
            max_new_tokens=self.judge_max_new_tokens,
            temperature=self.judge_temperature,
        )
        raw_scores = self.parse_scores(judge_text)
        return raw_scores, judge_text

    def score_with_heuristics(self, step_text: str) -> Dict[str, float]:
        """Fallback only. It keeps the pipeline runnable but is not a real PRM."""

        text = step_text.strip()
        lower = text.lower()
        token_count = max(1, len(text.split()))

        base = 6.0
        raw = {dim: base for dim in self.dimensions}

        for dim in self.dimensions:
            d = dim.lower()
            if "concise" in d:
                if token_count <= 80:
                    raw[dim] = 8.0
                elif token_count <= 180:
                    raw[dim] = 6.0
                else:
                    raw[dim] = 4.0
            elif "visual" in d or "ground" in d:
                visual_terms = [
                    "image", "figure", "diagram", "chart", "graph", "table",
                    "angle", "line", "point", "axis", "bar", "color", "shown",
                    "visible", "intersect", "triangle", "circle",
                ]
                raw[dim] = 7.0 if any(w in lower for w in visual_terms) else 5.0
            elif "logical" in d or "coherence" in d:
                logic_terms = [
                    "because", "therefore", "since", "so", "thus", "implies",
                    "hence", "then", "as a result",
                ]
                raw[dim] = 7.0 if any(w in lower for w in logic_terms) else 5.5
            elif "correct" in d or "semantic" in d or "math" in d:
                raw[dim] = 6.5 if re.search(r"\d|=|\+|\-|\*|/|°|angle", lower) else 5.5

        if not text or "i don't know" in lower or "cannot" in lower:
            raw = {dim: 3.0 for dim in self.dimensions}
        return raw

    def parse_scores(self, judge_text: str) -> Dict[str, float]:
        raw: Dict[str, float] = {}
        # Supports both hyphen and en dash separators.
        matches = re.findall(r"([^:\n\-–]+):\s*\[\[\s*([0-9]+(?:\.[0-9]+)?)\s*\]\]", judge_text)
        for name, val in matches:
            canonical = self.match_dimension(name.strip())
            if canonical is None:
                continue
            raw[canonical] = max(1.0, min(10.0, float(val)))

        # If the judge omitted a dimension, keep it neutral rather than making
        # the vector shorter and breaking Chebyshev scalarization.
        for dim in self.dimensions:
            raw.setdefault(dim, 5.0)
        return raw

    def match_dimension(self, name: str) -> Optional[str]:
        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", s.lower())

        target = norm(name)
        for dim in self.dimensions:
            if target == norm(dim):
                return dim
        # Fuzzy containment helps when the judge shortens names.
        for dim in self.dimensions:
            nd = norm(dim)
            if target and (target in nd or nd in target):
                return dim
        return None

    def raw_to_vector(self, raw_scores: Dict[str, float]) -> List[float]:
        return [normalize_raw_score(raw_scores.get(dim, 5.0)) for dim in self.dimensions]


def uct_score(
    child: MCTSNode,
    parent_visits: int,
    ideal_point: List[float],
    rho: float,
    exploration_c: float,
) -> float:
    # Paper-style selection is Chebyshev step quality plus UCT exploration.
    if child.reward_vector is None:
        exploitation = child.value_mean
    elif child.visits == 0:
        exploitation = augmented_chebyshev(child.reward_vector, ideal_point, rho)
    else:
        # Backpropagated trajectory value is used once available.
        exploitation = child.value_mean

    exploration = exploration_c * math.sqrt(
        math.log(max(parent_visits, 1) + 1.0) / (child.visits + 1.0)
    )
    return exploitation + exploration


def expand_node(
    node: MCTSNode,
    model,
    tokenizer,
    question: str,
    pixel_values: Optional[torch.Tensor],
    num_patches_list: Optional[List[int]],
    branching_factor: int,
    max_new_tokens: int,
    temperature: float,
    scorer: StepRewardScorer,
    ideal_point: List[float],
    ideal_lambda: float,
    reward_logs: List[Dict],
) -> List[float]:
    """Sample k candidate next steps, score each as a reward vector, update z*."""

    remaining = branching_factor - len(node.children)
    if remaining <= 0:
        return ideal_point

    input_ids, attention_mask, eos_id = build_query(
        model=model,
        tokenizer=tokenizer,
        question=question,
        reasoning_prefix=node.reasoning_prefix,
        pixel_values=pixel_values,
        num_patches_list=num_patches_list,
    )

    seen_steps = {ch.step_text for ch in node.children}
    for _ in range(remaining):
        step_text = generate_next_step(
            model=model,
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=eos_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        if not step_text or step_text in seen_steps:
            continue
        seen_steps.add(step_text)

        terminal = is_terminal_step(step_text)
        if terminal:
            next_prefix = node.reasoning_prefix + step_text + "\n"
        else:
            next_prefix = node.reasoning_prefix + step_text + "\n<|reasoning_proceed|>\n"

        raw_scores, reward_vector, judge_text = scorer.score_step(
            model=model,
            tokenizer=tokenizer,
            question=question,
            reasoning_prefix=node.reasoning_prefix,
            step_text=step_text,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
        )

        ideal_point = update_ideal_point(ideal_point, reward_vector, ideal_lambda)

        child = MCTSNode(
            reasoning_prefix=next_prefix,
            depth=node.depth + 1,
            parent=node,
            step_text=step_text,
            reward_vector=reward_vector,
            raw_scores=raw_scores,
            terminal=terminal,
        )
        node.add_child(child)

        reward_logs.append(
            {
                "parent_id": node.node_id,
                "node_id": child.node_id,
                "depth": child.depth,
                "step_text": step_text,
                "raw_scores": raw_scores,
                "reward_vector": reward_vector,
                "judge_text": judge_text,
            }
        )

    return ideal_point


def collect_nodes(root: MCTSNode) -> List[MCTSNode]:
    out: List[MCTSNode] = []
    stack = [root]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children)
    return out


def collect_candidate_nodes(root: MCTSNode) -> List[MCTSNode]:
    nodes = collect_nodes(root)
    candidates = [
        n for n in nodes
        if n.depth > 0 and (not n.children or n.terminal)
    ]
    if not candidates:
        candidates = [n for n in nodes if n.depth > 0]
    return candidates


def fused_trajectory_score(
    node: MCTSNode,
    ideal_point: List[float],
    rho: float,
    eta: float,
    answer: Optional[str],
    num_dims: int,
) -> Tuple[float, List[float], float, float]:
    r_traj = trajectory_reward_vector(node, num_dims)
    g_aug = augmented_chebyshev(r_traj, ideal_point, rho)
    r_ans = answer_reward(node.reasoning_prefix, answer)
    score = eta * g_aug + (1.0 - eta) * r_ans
    return score, r_traj, g_aug, r_ans


def backpropagate(path: List[MCTSNode], value: float) -> None:
    for n in path:
        n.visits += 1
        n.value_sum += value


def mcts_search(
    model,
    tokenizer,
    question: str,
    pixel_values: Optional[torch.Tensor],
    num_patches_list: Optional[List[int]],
    answer: Optional[str],
    max_depth: int,
    branching_factor: int,
    simulations: int,
    max_new_tokens: int,
    temperature: float,
    scorer: StepRewardScorer,
    exploration_c: float,
    rho: float,
    eta: float,
    ideal_lambda: float,
) -> Tuple[MCTSNode, Dict]:
    root_prefix = "<|reasoning_start|>\n"
    root = MCTSNode(reasoning_prefix=root_prefix, depth=0, parent=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pixel_values is not None:
        pixel_values = pixel_values.to(torch.bfloat16).to(device)

    # Because normalized rewards lie in [0, 1], 1.0 is a natural utopia point.
    ideal_point = [1.0] * len(scorer.dimensions)
    reward_logs: List[Dict] = []
    simulation_logs: List[Dict] = []

    for sim_idx in range(simulations):
        node = root
        path = [node]

        while node.depth < max_depth and not node.terminal:
            if len(node.children) < branching_factor:
                ideal_point = expand_node(
                    node=node,
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    pixel_values=pixel_values,
                    num_patches_list=num_patches_list,
                    branching_factor=branching_factor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    scorer=scorer,
                    ideal_point=ideal_point,
                    ideal_lambda=ideal_lambda,
                    reward_logs=reward_logs,
                )
                if not node.children:
                    break

            node = max(
                node.children,
                key=lambda ch: uct_score(
                    child=ch,
                    parent_visits=node.visits,
                    ideal_point=ideal_point,
                    rho=rho,
                    exploration_c=exploration_c,
                ),
            )
            path.append(node)

        fused_score, r_traj, g_aug, r_ans = fused_trajectory_score(
            node=node,
            ideal_point=ideal_point,
            rho=rho,
            eta=eta,
            answer=answer,
            num_dims=len(scorer.dimensions),
        )
        ideal_point = update_ideal_point(ideal_point, r_traj, ideal_lambda)
        backpropagate(path, fused_score)

        simulation_logs.append(
            {
                "simulation": sim_idx,
                "leaf_id": node.node_id,
                "depth": node.depth,
                "fused_score": fused_score,
                "chebyshev_score": g_aug,
                "answer_reward": r_ans,
                "trajectory_reward": r_traj,
                "ideal_point": ideal_point,
            }
        )

    candidates = collect_candidate_nodes(root)
    scored_candidates = []
    for n in candidates:
        score, r_traj, g_aug, r_ans = fused_trajectory_score(
            node=n,
            ideal_point=ideal_point,
            rho=rho,
            eta=eta,
            answer=answer,
            num_dims=len(scorer.dimensions),
        )
        scored_candidates.append((score, n, r_traj, g_aug, r_ans))

    best_score, best_node, best_r_traj, best_g_aug, best_r_ans = max(
        scored_candidates,
        key=lambda item: item[0],
    )

    metadata = {
        "ideal_point": ideal_point,
        "best_score": best_score,
        "best_chebyshev_score": best_g_aug,
        "best_answer_reward": best_r_ans,
        "best_trajectory_reward": best_r_traj,
        "reward_dimensions": scorer.dimensions,
        "reward_logs": reward_logs,
        "simulation_logs": simulation_logs,
        "candidate_scores": [
            {
                "node_id": n.node_id,
                "depth": n.depth,
                "score": score,
                "chebyshev_score": g_aug,
                "answer_reward": r_ans,
                "trajectory_reward": r_traj,
                "visits": n.visits,
            }
            for score, n, r_traj, g_aug, r_ans in scored_candidates
        ],
    }
    return best_node, metadata


def make_preference_pairs(
    root: MCTSNode,
    question: str,
    image: Optional[str],
    answer: Optional[str],
    ideal_point: List[float],
    rho: float,
    eta: float,
    num_dims: int,
    margin: float,
    max_pairs: int,
) -> List[Dict]:
    candidates = collect_candidate_nodes(root)
    scored = []
    for node in candidates:
        score, r_traj, g_aug, r_ans = fused_trajectory_score(
            node=node,
            ideal_point=ideal_point,
            rho=rho,
            eta=eta,
            answer=answer,
            num_dims=num_dims,
        )
        scored.append((score, node, r_traj, g_aug, r_ans))

    scored.sort(key=lambda x: x[0], reverse=True)
    pairs: List[Dict] = []
    for i, (chosen_score, chosen, chosen_r, chosen_g, chosen_ans) in enumerate(scored):
        for rejected_score, rejected, rejected_r, rejected_g, rejected_ans in reversed(scored[i + 1:]):
            if chosen_score - rejected_score < margin:
                continue
            pairs.append(
                {
                    "prompt": question,
                    "image": image,
                    "answer": answer,
                    "chosen": chosen.reasoning_prefix + "<|reasoning_end|>",
                    "rejected": rejected.reasoning_prefix + "<|reasoning_end|>",
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "chosen_trajectory_reward": chosen_r,
                    "rejected_trajectory_reward": rejected_r,
                    "chosen_chebyshev_score": chosen_g,
                    "rejected_chebyshev_score": rejected_g,
                    "chosen_answer_reward": chosen_ans,
                    "rejected_answer_reward": rejected_ans,
                }
            )
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def load_criteria(path: Optional[str]) -> List[str]:
    if not path:
        return DEFAULT_REWARD_DIMENSIONS
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        for key in ("criteria", "dimensions", "reward_dimensions"):
            if key in obj and isinstance(obj[key], list):
                return [str(x) for x in obj[key]]
    raise ValueError(f"Cannot parse criteria file: {path}")


def parse_item(line: str, answer_field: str, input_dir: Optional[Path]) -> Tuple[Dict, str, Optional[str], Optional[str]]:
    item = json.loads(line)
    q = item.get("question")
    img = item.get("image")
    answer = item.get(answer_field)
    if answer is None:
        answer = item.get("answer", item.get("label", item.get("gt_answer")))

    if q is None and "conversations" in item:
        for m in item["conversations"]:
            if m.get("from") == "human":
                q = m.get("value")
                break

    if img and input_dir is not None and not os.path.isabs(img) and not os.path.exists(img):
        candidate = input_dir / img
        if candidate.exists():
            img = str(candidate)

    return item, q, img, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="huggingface/sft_mpo_5e_6")
    parser.add_argument("--input", type=str, default="datasets/dpo_data.jsonl")
    parser.add_argument("--output", type=str, default="mcts_pipeline/chebyshev_mcts_outputs.jsonl")
    parser.add_argument("--preference-output", type=str, default="mcts_pipeline/chebyshev_dpo_pairs.jsonl")
    parser.add_argument("--log-output", type=str, default="mcts_pipeline/chebyshev_mcts_logs.jsonl")

    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--use-thumbnail", action="store_true")
    parser.add_argument("--max-num", type=int, default=6)

    # Paper defaults: branch factor k=3, search depth D=10.
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--branching", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)

    # Paper reward/scalarization parameters: eta=0.5, rho=0.1, lambda=0.2.
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--ideal-lambda", type=float, default=0.2)
    parser.add_argument("--exploration-c", type=float, default=1.4)

    parser.add_argument("--pair-margin", type=float, default=0.05)
    parser.add_argument("--max-pairs-per-item", type=int, default=1)
    parser.add_argument("--answer-field", type=str, default="answer")

    parser.add_argument("--criteria-file", type=str, default=None)
    parser.add_argument(
        "--reward-backend",
        type=str,
        default="self",
        choices=["self", "heuristic", "neutral"],
        help="self: use loaded VLM as judge; heuristic/neutral: fallbacks only, not paper-faithful.",
    )
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    parser.add_argument("--judge-temperature", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    class A:
        pass

    a = A()
    a.checkpoint = args.checkpoint
    a.load_in_8bit = args.load_in_8bit
    a.load_in_4bit = args.load_in_4bit
    a.auto = args.auto
    model, tokenizer = load_model_and_tokenizer(a)

    dimensions = load_criteria(args.criteria_file)
    scorer = StepRewardScorer(
        dimensions=dimensions,
        backend=args.reward_backend,
        judge_max_new_tokens=args.judge_max_new_tokens,
        judge_temperature=args.judge_temperature,
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    pref_dir = os.path.dirname(args.preference_output)
    if pref_dir:
        os.makedirs(pref_dir, exist_ok=True)
    log_dir = os.path.dirname(args.log_output)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    input_path = Path(args.input)
    input_dir = input_path.parent if input_path.parent.exists() else None

    with open(args.input, "r", encoding="utf-8") as reader, \
        open(args.output, "w", encoding="utf-8") as writer, \
        open(args.preference_output, "w", encoding="utf-8") as pref_writer, \
        open(args.log_output, "w", encoding="utf-8") as log_writer:

        for idx, line in enumerate(reader):
            if args.limit is not None and idx >= args.limit:
                break
            if not line.strip():
                continue

            item, question, image, answer = parse_item(line, args.answer_field, input_dir)
            if question is None:
                continue

            pixel_values, num_patches_list = build_pixel_values(
                image_path=image,
                image_size=args.image_size,
                dynamic_image_size=args.dynamic,
                use_thumbnail=args.use_thumbnail,
                max_num=args.max_num,
            )

            best_node, metadata = mcts_search(
                model=model,
                tokenizer=tokenizer,
                question=question,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                answer=answer,
                max_depth=args.max_depth,
                branching_factor=args.branching,
                simulations=args.simulations,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                scorer=scorer,
                exploration_c=args.exploration_c,
                rho=args.rho,
                eta=args.eta,
                ideal_lambda=args.ideal_lambda,
            )

            final_reasoning = best_node.reasoning_prefix + "<|reasoning_end|>"
            writer.write(
                json.dumps(
                    {
                        "question": question,
                        "image": image,
                        "answer": answer,
                        "reasoning": final_reasoning,
                        "score": metadata["best_score"],
                        "chebyshev_score": metadata["best_chebyshev_score"],
                        "answer_reward": metadata["best_answer_reward"],
                        "trajectory_reward": metadata["best_trajectory_reward"],
                        "ideal_point": metadata["ideal_point"],
                        "reward_dimensions": metadata["reward_dimensions"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            writer.flush()

            pairs = make_preference_pairs(
                root=trajectory_nodes(best_node)[0],
                question=question,
                image=image,
                answer=answer,
                ideal_point=metadata["ideal_point"],
                rho=args.rho,
                eta=args.eta,
                num_dims=len(dimensions),
                margin=args.pair_margin,
                max_pairs=args.max_pairs_per_item,
            )
            for pair in pairs:
                pref_writer.write(json.dumps(pair, ensure_ascii=False) + "\n")
            pref_writer.flush()

            log_writer.write(
                json.dumps(
                    {
                        "index": idx,
                        "question": question,
                        "image": image,
                        "answer": answer,
                        **metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log_writer.flush()


if __name__ == "__main__":
    main()
