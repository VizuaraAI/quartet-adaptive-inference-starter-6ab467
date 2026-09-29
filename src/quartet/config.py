"""Configuration loading. One YAML file drives rollouts, training and eval."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "default.yaml",
)


@dataclass
class ModelConfig:
    small_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    large_model: str = "Qwen/Qwen2.5-14B-Instruct"
    dtype: str = "bfloat16"
    device: str = "auto"
    max_new_tokens_answer: int = 32
    max_new_tokens_reason: int = 384
    reason_samples: int = 5
    reason_temperature: float = 0.7


@dataclass
class RetrievalConfig:
    corpus: str = "wikipedia-paragraphs-1m"
    top_k: int = 3
    index_dir: str = "artifacts/bm25_index"
    max_passage_tokens: int = 160


@dataclass
class DataConfig:
    datasets: list[str] = field(default_factory=lambda: ["popqa", "gsm8k", "hotpotqa"])
    train_size: int = 4000
    val_size: int = 1000
    test_size: int = 2000
    seed: int = 0


@dataclass
class BudgetConfig:
    """Budgets are per-query averages. The policy is conditioned on them, so a
    single trained policy covers the whole sweep."""

    latency_ms: float = 900.0
    energy_j: float = 30.0
    peak_memory_mib: float = 12000.0
    #: Multipliers applied to the budgets when sweeping the Pareto front.
    sweep: list[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
    )


@dataclass
class TrainConfig:
    hidden_dim: int = 128
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    dual_lr: float = 0.05
    entropy_bonus: float = 0.01
    lambda_init: float = 0.0
    lambda_max: float = 50.0
    seed: int = 0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rollout_path: str = "artifacts/rollouts.jsonl"
    output_dir: str = "artifacts"
    #: Set false on a machine with no NVML/RAPL access; meters then fall back
    #: to an analytic energy model and say so in the logs.
    measure_energy: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        raw = copy.deepcopy(raw or {})
        sections = {
            "model": ModelConfig,
            "retrieval": RetrievalConfig,
            "data": DataConfig,
            "budget": BudgetConfig,
            "train": TrainConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in sections.items():
            kwargs[name] = klass(**raw.pop(name, {}) or {})
        for key in ("rollout_path", "output_dir", "measure_energy"):
            if key in raw:
                kwargs[key] = raw.pop(key)
        if raw:
            raise ValueError(f"unknown config keys: {sorted(raw)}")
        return cls(**kwargs)


def load_config(path: str | None = None) -> Config:
    """Read the YAML config. Falls back to the dataclass defaults if absent."""
    import yaml

    path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        return Config()
    with open(path, "r", encoding="utf-8") as handle:
        return Config.from_dict(yaml.safe_load(handle) or {})


def budget_vector(cfg: Config, scale: float = 1.0) -> list[float]:
    """Budgets as the normalised 3-vector fed to the policy as input."""
    return [
        cfg.budget.latency_ms * scale,
        cfg.budget.energy_j * scale,
        cfg.budget.peak_memory_mib * scale,
    ]
