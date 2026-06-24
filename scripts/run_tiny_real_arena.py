#!/usr/bin/env python3
"""Run a tiny real-model arena smoke on CPU.

This script trains a very small dense LLaMA on synthetic patterned sequences,
then evaluates FP and ternary-converted candidates with resource-aware fitness.
It is intentionally small enough to run locally before moving larger jobs to
Colab.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402
from bitnet_llama.module import BitLinear  # noqa: E402
from scripts.estimate_memory_traffic import ModelShape, default_policies, estimate  # noqa: E402


EPS = 1e-12


@dataclass
class CandidateResult:
    name: str
    quality_source: str
    runtime_policy: str
    loss: float
    perplexity: float
    token_accuracy: float
    logit_kl_to_fp16: float
    bytes_per_token_mb: float
    ideal_latency_ms_at_100gbps: float
    peak_ram_proxy_mb: float
    fitness: float
    pareto_frontier: bool = False


def make_batch(batch_size: int, seq_len: int, vocab_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(3, vocab_size, (batch_size, 1), generator=generator)
    steps = torch.randint(1, 9, (batch_size, 1), generator=generator)
    bends = torch.randint(0, 5, (batch_size, 1), generator=generator)
    positions = torch.arange(seq_len).unsqueeze(0)
    values = starts + steps * positions + bends * ((positions % 7) ** 2)
    tokens = (values % (vocab_size - 3)) + 3
    tokens[:, 0] = 1
    tokens[:, -1] = 2
    return tokens.long()


def build_model(args: argparse.Namespace):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        max_position_embeddings=args.seq_len,
        tie_word_embeddings=False,
    )
    torch.manual_seed(args.seed)
    return LlamaForCausalLM(config), config


def train_model(model, args: argparse.Namespace) -> list[float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses: list[float] = []
    for step in range(args.train_steps):
        input_ids = make_batch(args.batch_size, args.seq_len, args.vocab_size, args.seed + 1000 + step)
        output = model(input_ids=input_ids, labels=input_ids)
        loss = output.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return losses


def project_model_to_s1(model) -> None:
    projected_state, _ = C.convert_state_dict(model.state_dict(), C.S1)
    model.load_state_dict(projected_state)


def is_target_linear_name(name: str) -> bool:
    return C.is_target_weight_key(f"{name}.weight")


def set_nested_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def replace_target_linears_with_bitlinear(model, num_groups: int) -> int:
    replacements = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not is_target_linear_name(name):
            continue
        ternary_code, _, _ = C.quantize_weight(module.weight, C.S1)
        replacement = BitLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            num_groups=num_groups,
        )
        replacement.to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            replacement.weight.copy_(
                ternary_code.to(device=module.weight.device, dtype=module.weight.dtype)
            )
            if module.bias is not None:
                replacement.bias.copy_(module.bias)
        set_nested_module(model, name, replacement)
        replacements += 1
    return replacements


def recover_s1_with_bitlinear_ste(model, args: argparse.Namespace):
    recovered = copy.deepcopy(model)
    replacement_count = replace_target_linears_with_bitlinear(recovered, args.ste_num_groups)
    recovered.train()
    optimizer = torch.optim.AdamW(recovered.parameters(), lr=args.ste_qat_learning_rate)
    losses: list[float] = []
    for step in range(args.ste_qat_steps):
        input_ids = make_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            seed=args.seed + 30000 + step,
        )
        output = recovered(input_ids=input_ids, labels=input_ids)
        loss = output.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    recovered.eval()
    return recovered, losses, replacement_count


def recover_s1_with_projected_qat(model, args: argparse.Namespace):
    recovered = copy.deepcopy(model)
    project_model_to_s1(recovered)
    recovered.train()
    optimizer = torch.optim.AdamW(recovered.parameters(), lr=args.qat_learning_rate)
    losses: list[float] = []
    for step in range(args.qat_steps):
        input_ids = make_batch(args.batch_size, args.seq_len, args.vocab_size, args.seed + 20000 + step)
        output = recovered(input_ids=input_ids, labels=input_ids)
        loss = output.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        project_model_to_s1(recovered)
        losses.append(float(loss.detach()))
    recovered.eval()
    return recovered, losses


@torch.no_grad()
def evaluate_model(model, input_ids: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    output = model(input_ids=input_ids, labels=input_ids)
    logits = output.logits.detach()
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    predictions = shift_logits.argmax(dim=-1)
    accuracy = float((predictions == shift_labels).float().mean())
    return float(output.loss), accuracy, logits


def clone_with_state(model, state_dict: dict[str, torch.Tensor]):
    cloned = copy.deepcopy(model)
    cloned.load_state_dict(state_dict)
    cloned.eval()
    return cloned


def logit_kl(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> float:
    ref = F.log_softmax(reference_logits.float(), dim=-1)
    cand = F.log_softmax(candidate_logits.float(), dim=-1)
    ref_prob = ref.exp()
    return float((ref_prob * (ref - cand)).sum(dim=-1).mean())


def traffic_by_policy(config, seq_len: int) -> dict[str, tuple[float, float]]:
    shape = ModelShape(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
    )
    traffic: dict[str, tuple[float, float]] = {}
    for policy in default_policies(16.0):
        item = estimate(shape, policy, seq_len, batch_size=1, bandwidth_gbps_values=[100.0])
        tps = item.ideal_tokens_per_second_at_bandwidth["100_GBps"]
        traffic[policy.name] = (item.total_mb_per_token, 1000.0 / max(tps, EPS))
    return traffic


def normalized_log(value: float, baseline: float) -> float:
    return math.log(max(value, EPS) / max(baseline, EPS))


def compute_fitness(
    token_accuracy: float,
    bytes_per_token_mb: float,
    latency_ms: float,
    peak_ram_proxy_mb: float,
    max_bytes: float,
    max_latency: float,
    max_ram: float,
    args: argparse.Namespace,
) -> float:
    return (
        token_accuracy
        - args.lambda_bytes * normalized_log(bytes_per_token_mb, max_bytes)
        - args.lambda_latency * normalized_log(latency_ms, max_latency)
        - args.lambda_ram * normalized_log(peak_ram_proxy_mb, max_ram)
    )


def dominates(left: CandidateResult, right: CandidateResult) -> bool:
    no_worse = (
        left.token_accuracy >= right.token_accuracy - EPS
        and left.bytes_per_token_mb <= right.bytes_per_token_mb + EPS
        and left.ideal_latency_ms_at_100gbps <= right.ideal_latency_ms_at_100gbps + EPS
        and left.peak_ram_proxy_mb <= right.peak_ram_proxy_mb + EPS
    )
    strict = (
        left.token_accuracy > right.token_accuracy + EPS
        or left.bytes_per_token_mb < right.bytes_per_token_mb - EPS
        or left.ideal_latency_ms_at_100gbps < right.ideal_latency_ms_at_100gbps - EPS
        or left.peak_ram_proxy_mb < right.peak_ram_proxy_mb - EPS
    )
    return no_worse and strict


def mark_pareto(results: list[CandidateResult]) -> None:
    for item in results:
        item.pareto_frontier = not any(
            dominates(other, item) for other in results if other.name != item.name
        )


def evaluate_candidates(
    model,
    config,
    eval_ids: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[CandidateResult], list[float], list[float], int]:
    fp_loss, fp_acc, fp_logits = evaluate_model(model, eval_ids)
    converted_s0, _ = C.convert_state_dict(model.state_dict(), C.S0)
    converted_s1, _ = C.convert_state_dict(model.state_dict(), C.S1)
    s0_model = clone_with_state(model, converted_s0)
    s1_model = clone_with_state(model, converted_s1)
    s0_loss, s0_acc, s0_logits = evaluate_model(s0_model, eval_ids)
    s1_loss, s1_acc, s1_logits = evaluate_model(s1_model, eval_ids)
    qat_model, qat_losses = recover_s1_with_projected_qat(model, args)
    qat_loss, qat_acc, qat_logits = evaluate_model(qat_model, eval_ids)
    ste_model, ste_losses, bitlinear_replacements = recover_s1_with_bitlinear_ste(model, args)
    ste_loss, ste_acc, ste_logits = evaluate_model(ste_model, eval_ids)

    traffic = traffic_by_policy(config, args.seq_len)
    raw = [
        {
            "name": "fp16_dense",
            "quality_source": "fp16",
            "runtime_policy": "fp16_baseline",
            "loss": fp_loss,
            "accuracy": fp_acc,
            "kl": 0.0,
            "ram_factor": 1.0,
        },
        {
            "name": "s0_ternary_ptq_packed",
            "quality_source": "S0",
            "runtime_policy": "packed_b1_58_weight_fp16_kv",
            "loss": s0_loss,
            "accuracy": s0_acc,
            "kl": logit_kl(fp_logits, s0_logits),
            "ram_factor": 0.35,
        },
        {
            "name": "s1_groupwise_ptq_packed",
            "quality_source": "S1",
            "runtime_policy": "packed_b1_58_weight_fp16_kv",
            "loss": s1_loss,
            "accuracy": s1_acc,
            "kl": logit_kl(fp_logits, s1_logits),
            "ram_factor": 0.35,
        },
        {
            "name": "s1_groupwise_ptq_int8_kv",
            "quality_source": "S1",
            "runtime_policy": "packed_b1_58_weight_int8_kv",
            "loss": s1_loss,
            "accuracy": s1_acc,
            "kl": logit_kl(fp_logits, s1_logits),
            "ram_factor": 0.28,
        },
        {
            "name": "s1_groupwise_ptq_int4_kv",
            "quality_source": "S1",
            "runtime_policy": "packed_b1_58_weight_int4_kv",
            "loss": s1_loss,
            "accuracy": s1_acc,
            "kl": logit_kl(fp_logits, s1_logits),
            "ram_factor": 0.23,
        },
        {
            "name": "s1_projected_qat_int8_kv",
            "quality_source": "S1_projected_QAT",
            "runtime_policy": "packed_b1_58_weight_int8_kv",
            "loss": qat_loss,
            "accuracy": qat_acc,
            "kl": logit_kl(fp_logits, qat_logits),
            "ram_factor": 0.28,
        },
        {
            "name": "s1_projected_qat_int4_kv",
            "quality_source": "S1_projected_QAT",
            "runtime_policy": "packed_b1_58_weight_int4_kv",
            "loss": qat_loss,
            "accuracy": qat_acc,
            "kl": logit_kl(fp_logits, qat_logits),
            "ram_factor": 0.23,
        },
        {
            "name": "s1_bitlinear_ste_int8_kv",
            "quality_source": "S1_BitLinear_STE",
            "runtime_policy": "packed_b1_58_weight_int8_kv",
            "loss": ste_loss,
            "accuracy": ste_acc,
            "kl": logit_kl(fp_logits, ste_logits),
            "ram_factor": 0.28,
        },
        {
            "name": "s1_bitlinear_ste_int4_kv",
            "quality_source": "S1_BitLinear_STE",
            "runtime_policy": "packed_b1_58_weight_int4_kv",
            "loss": ste_loss,
            "accuracy": ste_acc,
            "kl": logit_kl(fp_logits, ste_logits),
            "ram_factor": 0.23,
        },
    ]
    max_bytes = max(traffic[item["runtime_policy"]][0] for item in raw)
    max_latency = max(traffic[item["runtime_policy"]][1] for item in raw)
    max_ram = args.peak_ram_baseline_mb

    results: list[CandidateResult] = []
    for item in raw:
        bytes_per_token, latency = traffic[item["runtime_policy"]]
        ram = args.peak_ram_baseline_mb * item["ram_factor"]
        fitness = compute_fitness(
            item["accuracy"],
            bytes_per_token,
            latency,
            ram,
            max_bytes,
            max_latency,
            max_ram,
            args,
        )
        results.append(
            CandidateResult(
                name=item["name"],
                quality_source=item["quality_source"],
                runtime_policy=item["runtime_policy"],
                loss=item["loss"],
                perplexity=float(math.exp(min(item["loss"], 20.0))),
                token_accuracy=item["accuracy"],
                logit_kl_to_fp16=item["kl"],
                bytes_per_token_mb=bytes_per_token,
                ideal_latency_ms_at_100gbps=latency,
                peak_ram_proxy_mb=ram,
                fitness=fitness,
            )
        )
    mark_pareto(results)
    return results, qat_losses, ste_losses, bitlinear_replacements


def print_results(results: list[CandidateResult], train_losses: list[float], elapsed: float) -> None:
    print("Tiny real-model evolutionary arena smoke")
    print("=" * 48)
    print(f"train_loss: start={train_losses[0]:.4f} end={train_losses[-1]:.4f} elapsed={elapsed:.1f}s")
    print(
        f"{'candidate':>30} {'acc':>8} {'loss':>8} {'kl':>8} "
        f"{'MB/tok':>9} {'lat_ms':>9} {'RAM':>9} {'fitness':>9} {'pareto':>8}"
    )
    print("-" * 112)
    for item in sorted(results, key=lambda value: value.fitness, reverse=True):
        print(
            f"{item.name:>30} "
            f"{item.token_accuracy:>8.3f} "
            f"{item.loss:>8.3f} "
            f"{item.logit_kl_to_fp16:>8.3f} "
            f"{item.bytes_per_token_mb:>9.3f} "
            f"{item.ideal_latency_ms_at_100gbps:>9.4f} "
            f"{item.peak_ram_proxy_mb:>9.1f} "
            f"{item.fitness:>9.3f} "
            f"{str(item.pareto_frontier):>8}"
        )
    quality_winner = max(results, key=lambda item: item.token_accuracy)
    resource_winner = max(results, key=lambda item: item.fitness)
    print("\nchecks:")
    print(f"  quality winner          : {quality_winner.name}")
    print(f"  resource-aware winner   : {resource_winner.name}")
    print(f"  resource changes winner : {quality_winner.name != resource_winner.name}")
    print(f"  pareto frontier         : {', '.join(item.name for item in results if item.pareto_frontier)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--qat-steps", type=int, default=48)
    parser.add_argument("--qat-learning-rate", type=float, default=2e-3)
    parser.add_argument("--ste-qat-steps", type=int, default=48)
    parser.add_argument("--ste-qat-learning-rate", type=float, default=2e-3)
    parser.add_argument("--ste-num-groups", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--lambda-bytes", type=float, default=0.08)
    parser.add_argument("--lambda-latency", type=float, default=0.05)
    parser.add_argument("--lambda-ram", type=float, default=0.04)
    parser.add_argument("--peak-ram-baseline-mb", type=float, default=256.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json-out", type=Path, default=Path("reports/tiny_real_arena_smoke.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    model, config = build_model(args)
    start = time.perf_counter()
    train_losses = train_model(model, args)
    elapsed = time.perf_counter() - start
    eval_ids = make_batch(args.eval_batch_size, args.seq_len, args.vocab_size, args.seed + 9999)
    results, qat_losses, ste_losses, bitlinear_replacements = evaluate_candidates(
        model, config, eval_ids, args
    )
    print_results(results, train_losses, elapsed)
    if qat_losses:
        print(f"  projected_qat_loss: start={qat_losses[0]:.4f} end={qat_losses[-1]:.4f}")
    if ste_losses:
        print(f"  bitlinear_ste_loss: start={ste_losses[0]:.4f} end={ste_losses[-1]:.4f}")
        print(f"  bitlinear_replacements: {bitlinear_replacements}")

    payload = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_loss_start": train_losses[0],
        "train_loss_end": train_losses[-1],
        "projected_qat_loss_start": qat_losses[0] if qat_losses else None,
        "projected_qat_loss_end": qat_losses[-1] if qat_losses else None,
        "bitlinear_ste_loss_start": ste_losses[0] if ste_losses else None,
        "bitlinear_ste_loss_end": ste_losses[-1] if ste_losses else None,
        "bitlinear_replacements": bitlinear_replacements,
        "train_elapsed_seconds": elapsed,
        "results": [asdict(item) for item in results],
        "quality_winner": max(results, key=lambda item: item.token_accuracy).name,
        "resource_winner": max(results, key=lambda item: item.fitness).name,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")

    if args.strict:
        quality_winner = max(results, key=lambda item: item.token_accuracy)
        resource_winner = max(results, key=lambda item: item.fitness)
        checks = [
            train_losses[-1] < train_losses[0],
            any(item.pareto_frontier for item in results),
            resource_winner.name != "fp16_dense",
            bitlinear_replacements > 0,
            bool(ste_losses) and ste_losses[-1] < ste_losses[0],
        ]
        if not all(checks):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
