#!/usr/bin/env python3
"""Run a local feasibility smoke for an evolutionary low-resource LLM arena.

This does not train a model. It checks whether the proposed arena has enough
signal to rank model/runtime variants under quality and resource pressure.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DOMAINS = ["arithmetic", "retrieval", "code", "summary", "long_context"]
EPS = 1e-12


@dataclass(frozen=True)
class Candidate:
    name: str
    skills: dict[str, float]
    bytes_per_token_mb: float
    latency_ms: float
    peak_ram_mb: float
    robustness: float
    family: str


@dataclass(frozen=True)
class Task:
    task_id: str
    domain: str
    difficulty: float
    resource_stress: float
    novelty: float
    invalidity: float = 0.0
    split: str = "train"


@dataclass
class CandidateMetrics:
    name: str
    family: str
    quality_mean: float
    quality_se: float
    holdout_quality_mean: float
    bytes_per_token_mb: float
    latency_ms: float
    peak_ram_mb: float
    fitness: float
    fitness_se: float


@dataclass
class ArenaReport:
    signal_to_noise_pass: bool
    pareto_frontier_pass: bool
    resource_selection_pass: bool
    adversary_pass: bool
    holdout_pass: bool
    quality_winner: str
    resource_winner: str
    best_delta_over_second: float
    best_delta_se: float
    pareto_frontier: list[str]
    adversarial_mid_difficulty_rate: float
    adversarial_invalidity_mean: float
    holdout_delta_best: float


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def default_candidates() -> list[Candidate]:
    return [
        Candidate(
            name="fp16_tiny_baseline",
            family="dense",
            skills={
                "arithmetic": 0.72,
                "retrieval": 0.70,
                "code": 0.66,
                "summary": 0.74,
                "long_context": 0.68,
            },
            bytes_per_token_mb=49.0,
            latency_ms=38.0,
            peak_ram_mb=900.0,
            robustness=0.70,
        ),
        Candidate(
            name="int8_weight_baseline",
            family="quantized_dense",
            skills={
                "arithmetic": 0.70,
                "retrieval": 0.69,
                "code": 0.64,
                "summary": 0.72,
                "long_context": 0.67,
            },
            bytes_per_token_mb=30.0,
            latency_ms=29.0,
            peak_ram_mb=620.0,
            robustness=0.68,
        ),
        Candidate(
            name="current_py_bitlinear_reference",
            family="reference_bitlinear",
            skills={
                "arithmetic": 0.55,
                "retrieval": 0.54,
                "code": 0.50,
                "summary": 0.57,
                "long_context": 0.52,
            },
            bytes_per_token_mb=113.0,
            latency_ms=82.0,
            peak_ram_mb=980.0,
            robustness=0.46,
        ),
        Candidate(
            name="packed_b1_58_weight",
            family="packed_bitnet",
            skills={
                "arithmetic": 0.62,
                "retrieval": 0.61,
                "code": 0.58,
                "summary": 0.64,
                "long_context": 0.60,
            },
            bytes_per_token_mb=21.0,
            latency_ms=23.0,
            peak_ram_mb=360.0,
            robustness=0.56,
        ),
        Candidate(
            name="packed_b1_58_int8_kv",
            family="packed_bitnet_kv",
            skills={
                "arithmetic": 0.61,
                "retrieval": 0.61,
                "code": 0.57,
                "summary": 0.63,
                "long_context": 0.62,
            },
            bytes_per_token_mb=13.0,
            latency_ms=18.0,
            peak_ram_mb=290.0,
            robustness=0.57,
        ),
        Candidate(
            name="packed_b1_58_int4_kv",
            family="packed_bitnet_kv",
            skills={
                "arithmetic": 0.58,
                "retrieval": 0.58,
                "code": 0.54,
                "summary": 0.60,
                "long_context": 0.59,
            },
            bytes_per_token_mb=9.0,
            latency_ms=16.0,
            peak_ram_mb=245.0,
            robustness=0.51,
        ),
        Candidate(
            name="packed_b1_58_qat_recovery",
            family="packed_bitnet_qat",
            skills={
                "arithmetic": 0.68,
                "retrieval": 0.67,
                "code": 0.63,
                "summary": 0.70,
                "long_context": 0.65,
            },
            bytes_per_token_mb=13.5,
            latency_ms=18.5,
            peak_ram_mb=300.0,
            robustness=0.65,
        ),
    ]


def make_tasks(count_per_domain: int, rng: np.random.Generator, split: str) -> list[Task]:
    tasks: list[Task] = []
    for domain in DOMAINS:
        for index in range(count_per_domain):
            base = rng.uniform(0.28, 0.78)
            if domain == "long_context":
                stress = rng.uniform(0.65, 1.0)
                difficulty = min(0.95, base + rng.uniform(0.05, 0.16))
            elif domain == "code":
                stress = rng.uniform(0.35, 0.75)
                difficulty = min(0.92, base + rng.uniform(0.02, 0.12))
            else:
                stress = rng.uniform(0.15, 0.70)
                difficulty = base
            novelty = rng.uniform(0.1, 0.9)
            tasks.append(
                Task(
                    task_id=f"{split}_{domain}_{index:03d}",
                    domain=domain,
                    difficulty=float(difficulty),
                    resource_stress=float(stress),
                    novelty=float(novelty),
                    split=split,
                )
            )
    return tasks


def quality_probability(candidate: Candidate, task: Task) -> float:
    skill = candidate.skills[task.domain]
    stress_penalty = task.resource_stress * (1.0 - candidate.robustness) * 0.35
    invalidity_penalty = task.invalidity * 0.8
    margin = skill - task.difficulty - stress_penalty - invalidity_penalty
    return float(sigmoid(margin / 0.085))


def score_candidate(
    candidate: Candidate,
    tasks: Iterable[Task],
    repeats: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    task_list = list(tasks)
    probabilities = np.array([quality_probability(candidate, task) for task in task_list])
    noise = rng.normal(0.0, 0.035, size=(len(task_list), repeats))
    samples = np.clip(probabilities[:, None] + noise, 0.0, 1.0)
    return {
        "task_means": samples.mean(axis=1),
        "repeat_samples": samples,
        "probabilities": probabilities,
    }


def normalized_resource(value: float, baseline: float) -> float:
    return math.log(value / baseline)


def compute_metrics(
    candidates: list[Candidate],
    train_tasks: list[Task],
    holdout_tasks: list[Task],
    repeats: int,
    rng: np.random.Generator,
    lambda_bytes: float,
    lambda_latency: float,
    lambda_ram: float,
) -> tuple[list[CandidateMetrics], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    baseline_bytes = max(candidate.bytes_per_token_mb for candidate in candidates)
    baseline_latency = max(candidate.latency_ms for candidate in candidates)
    baseline_ram = max(candidate.peak_ram_mb for candidate in candidates)
    metrics: list[CandidateMetrics] = []
    train_scores: dict[str, np.ndarray] = {}
    train_fitness_scores: dict[str, np.ndarray] = {}
    holdout_scores: dict[str, np.ndarray] = {}

    for candidate in candidates:
        train_result = score_candidate(candidate, train_tasks, repeats, rng)
        holdout_result = score_candidate(candidate, holdout_tasks, repeats, rng)
        task_means = train_result["task_means"]
        holdout_means = holdout_result["task_means"]
        resource_penalty = (
            lambda_bytes * normalized_resource(candidate.bytes_per_token_mb, baseline_bytes)
            + lambda_latency * normalized_resource(candidate.latency_ms, baseline_latency)
            + lambda_ram * normalized_resource(candidate.peak_ram_mb, baseline_ram)
        )
        fitness_task_values = task_means - resource_penalty
        metrics.append(
            CandidateMetrics(
                name=candidate.name,
                family=candidate.family,
                quality_mean=float(task_means.mean()),
                quality_se=float(task_means.std(ddof=1) / math.sqrt(len(task_means))),
                holdout_quality_mean=float(holdout_means.mean()),
                bytes_per_token_mb=candidate.bytes_per_token_mb,
                latency_ms=candidate.latency_ms,
                peak_ram_mb=candidate.peak_ram_mb,
                fitness=float(fitness_task_values.mean()),
                fitness_se=float(fitness_task_values.std(ddof=1) / math.sqrt(len(fitness_task_values))),
            )
        )
        train_scores[candidate.name] = task_means
        train_fitness_scores[candidate.name] = fitness_task_values
        holdout_scores[candidate.name] = holdout_means

    return metrics, train_scores, train_fitness_scores, holdout_scores


def dominates(left: CandidateMetrics, right: CandidateMetrics) -> bool:
    no_worse = (
        left.quality_mean >= right.quality_mean - EPS
        and left.bytes_per_token_mb <= right.bytes_per_token_mb + EPS
        and left.latency_ms <= right.latency_ms + EPS
        and left.peak_ram_mb <= right.peak_ram_mb + EPS
    )
    strict = (
        left.quality_mean > right.quality_mean + EPS
        or left.bytes_per_token_mb < right.bytes_per_token_mb - EPS
        or left.latency_ms < right.latency_ms - EPS
        or left.peak_ram_mb < right.peak_ram_mb - EPS
    )
    return no_worse and strict


def pareto_frontier(metrics: list[CandidateMetrics]) -> list[str]:
    frontier: list[str] = []
    for candidate in metrics:
        if not any(dominates(other, candidate) for other in metrics if other.name != candidate.name):
            frontier.append(candidate.name)
    return frontier


def mutate_tasks(
    tasks: list[Task],
    candidates: list[Candidate],
    rng: np.random.Generator,
    count: int,
) -> list[Task]:
    scored: list[tuple[float, Task, float]] = []
    for task in tasks:
        pass_rate = float(np.mean([quality_probability(candidate, task) for candidate in candidates]))
        mid_difficulty_bonus = 1.0 - min(abs(pass_rate - 0.5) * 2.0, 1.0)
        failure_pressure = 1.0 - pass_rate
        adversary_score = 0.55 * mid_difficulty_bonus + 0.25 * failure_pressure + 0.20 * task.novelty
        scored.append((adversary_score, task, pass_rate))

    scored.sort(key=lambda item: item[0], reverse=True)
    seeds = scored[:count]
    mutated: list[Task] = []
    for index, (_, task, pass_rate) in enumerate(seeds):
        difficulty_delta = rng.uniform(-0.025, 0.045)
        if pass_rate < 0.35:
            difficulty_delta -= 0.10
        elif pass_rate > 0.65:
            difficulty_delta += 0.10
        mutated.append(
            Task(
                task_id=f"adv_{task.domain}_{index:03d}",
                domain=task.domain,
                difficulty=float(np.clip(task.difficulty + difficulty_delta, 0.10, 0.95)),
                resource_stress=float(np.clip(task.resource_stress + rng.uniform(0.0, 0.12), 0.0, 1.0)),
                novelty=float(np.clip(task.novelty + rng.uniform(0.05, 0.20), 0.0, 1.0)),
                invalidity=float(rng.beta(1.0, 20.0) * 0.2),
                split="adversarial",
            )
        )
    return mutated


def pairwise_signal(
    winner: CandidateMetrics,
    runner_up: CandidateMetrics,
    train_scores: dict[str, np.ndarray],
) -> tuple[float, float]:
    diff = train_scores[winner.name] - train_scores[runner_up.name]
    return float(diff.mean()), float(diff.std(ddof=1) / math.sqrt(len(diff)))


def build_report(
    candidates: list[Candidate],
    metrics: list[CandidateMetrics],
    train_scores: dict[str, np.ndarray],
    train_fitness_scores: dict[str, np.ndarray],
    holdout_scores: dict[str, np.ndarray],
    adversarial_tasks: list[Task],
    signal_k: float,
) -> ArenaReport:
    quality_winner = max(metrics, key=lambda item: item.quality_mean)
    sorted_by_fitness = sorted(metrics, key=lambda item: item.fitness, reverse=True)
    resource_winner = sorted_by_fitness[0]
    runner_up = sorted_by_fitness[1]
    delta, delta_se = pairwise_signal(resource_winner, runner_up, train_fitness_scores)
    frontier = pareto_frontier(metrics)

    adversarial_pass_rates = [
        float(np.mean([quality_probability(candidate, task) for candidate in candidates]))
        for task in adversarial_tasks
    ]
    mid_rate = float(np.mean([(0.2 <= rate <= 0.8) for rate in adversarial_pass_rates]))
    invalidity_mean = float(np.mean([task.invalidity for task in adversarial_tasks]))
    holdout_best = float(np.mean(holdout_scores[resource_winner.name]))
    train_best = float(np.mean(train_scores[resource_winner.name]))
    holdout_delta = holdout_best - train_best

    return ArenaReport(
        signal_to_noise_pass=abs(delta) > signal_k * max(delta_se, EPS),
        pareto_frontier_pass=len(frontier) >= 2,
        resource_selection_pass=quality_winner.name != resource_winner.name,
        adversary_pass=mid_rate >= 0.6 and invalidity_mean < 0.05,
        holdout_pass=holdout_delta >= -0.08,
        quality_winner=quality_winner.name,
        resource_winner=resource_winner.name,
        best_delta_over_second=delta,
        best_delta_se=delta_se,
        pareto_frontier=frontier,
        adversarial_mid_difficulty_rate=mid_rate,
        adversarial_invalidity_mean=invalidity_mean,
        holdout_delta_best=holdout_delta,
    )


def print_metrics(metrics: list[CandidateMetrics], report: ArenaReport) -> None:
    print("Evolutionary LLM arena feasibility smoke")
    print("=" * 48)
    print(
        f"{'candidate':>32} {'quality':>9} {'fit':>9} {'bytes':>9} "
        f"{'lat_ms':>9} {'ram_mb':>9} {'holdout':>9}"
    )
    print("-" * 96)
    for item in sorted(metrics, key=lambda metric: metric.fitness, reverse=True):
        print(
            f"{item.name:>32} "
            f"{item.quality_mean:>9.3f} "
            f"{item.fitness:>9.3f} "
            f"{item.bytes_per_token_mb:>9.1f} "
            f"{item.latency_ms:>9.1f} "
            f"{item.peak_ram_mb:>9.0f} "
            f"{item.holdout_quality_mean:>9.3f}"
        )

    print("\nChecks")
    print("-" * 48)
    print(f"quality winner:   {report.quality_winner}")
    print(f"resource winner:  {report.resource_winner}")
    print(f"pareto frontier:  {', '.join(report.pareto_frontier)}")
    print(
        "fitness signal:  "
        f"delta={report.best_delta_over_second:.4f}, "
        f"se={report.best_delta_se:.4f}, "
        f"pass={report.signal_to_noise_pass}"
    )
    print(f"resource changes winner: {report.resource_selection_pass}")
    print(f"pareto pass:             {report.pareto_frontier_pass}")
    print(
        "adversary pass:          "
        f"{report.adversary_pass} "
        f"(mid={report.adversarial_mid_difficulty_rate:.2f}, "
        f"invalid={report.adversarial_invalidity_mean:.3f})"
    )
    print(f"holdout pass:            {report.holdout_pass} (delta={report.holdout_delta_best:.3f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--tasks-per-domain", type=int, default=12)
    parser.add_argument("--holdout-tasks-per-domain", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--adversarial-count", type=int, default=20)
    parser.add_argument("--lambda-bytes", type=float, default=0.075)
    parser.add_argument("--lambda-latency", type=float, default=0.045)
    parser.add_argument("--lambda-ram", type=float, default=0.035)
    parser.add_argument("--signal-k", type=float, default=2.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json-out", type=Path, default=Path("reports/arena_feasibility_smoke.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    candidates = default_candidates()
    train_tasks = make_tasks(args.tasks_per_domain, rng, "train")
    holdout_tasks = make_tasks(args.holdout_tasks_per_domain, rng, "holdout")
    metrics, train_scores, train_fitness_scores, holdout_scores = compute_metrics(
        candidates=candidates,
        train_tasks=train_tasks,
        holdout_tasks=holdout_tasks,
        repeats=args.repeats,
        rng=rng,
        lambda_bytes=args.lambda_bytes,
        lambda_latency=args.lambda_latency,
        lambda_ram=args.lambda_ram,
    )
    adversarial_tasks = mutate_tasks(train_tasks, candidates, rng, args.adversarial_count)
    report = build_report(
        candidates,
        metrics,
        train_scores,
        train_fitness_scores,
        holdout_scores,
        adversarial_tasks,
        args.signal_k,
    )

    print_metrics(metrics, report)

    payload = {
        "seed": args.seed,
        "config": {
            "tasks_per_domain": args.tasks_per_domain,
            "holdout_tasks_per_domain": args.holdout_tasks_per_domain,
            "repeats": args.repeats,
            "adversarial_count": args.adversarial_count,
            "lambda_bytes": args.lambda_bytes,
            "lambda_latency": args.lambda_latency,
            "lambda_ram": args.lambda_ram,
            "signal_k": args.signal_k,
        },
        "candidates": [asdict(candidate) for candidate in candidates],
        "metrics": [asdict(metric) for metric in metrics],
        "report": asdict(report),
        "adversarial_tasks": [asdict(task) for task in adversarial_tasks],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")

    required = [
        report.signal_to_noise_pass,
        report.pareto_frontier_pass,
        report.resource_selection_pass,
        report.adversary_pass,
        report.holdout_pass,
    ]
    if args.strict and not all(required):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
