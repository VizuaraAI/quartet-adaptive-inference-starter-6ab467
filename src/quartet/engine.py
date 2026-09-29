"""Execution engine: the small model, the large model, retrieval, and the four
actions built on top of them.

Design note that matters for the paper: the gating pass and the ANSWER action
share one forward pass. We prefill the question once, read the gate features
off that pass, and reuse the same draft tokens as the answer. That is what
keeps the gate overhead under the threshold reported in the overhead ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quartet.actions import Action, ActionResult
from quartet.data import Example, score
from quartet.meters import fallback_energy, measure

ANSWER_SYSTEM = (
    "You are a concise question answering assistant. "
    "Reply with the short answer only, no explanation."
)
REASON_SYSTEM = (
    "Think step by step, then give the final answer on a new line "
    "beginning with 'Answer:'."
)
RETRIEVE_SYSTEM = (
    "Answer the question using the evidence passages. "
    "Reply with the short answer only, no explanation."
)


@dataclass
class Draft:
    """Output of the shared gating prefill: a cheap answer plus its statistics."""

    text: str
    mean_logprob: float
    mean_entropy: float
    min_token_prob: float
    prompt_tokens: int
    completion_tokens: int
    hidden: list[float]
    latency_ms: float
    energy_j: float
    peak_memory_mib: float


class Engine:
    """Wraps both models and the retriever behind one small interface."""

    def __init__(self, cfg, retriever=None):
        self.cfg = cfg
        self.retriever = retriever
        self._small = None
        self._large = None
        self._tokenizer = None
        self._large_tokenizer = None

    # -------------------------------------------------------- model setup ---

    def _load(self, name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.cfg.model.dtype, torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=dtype, device_map=self.cfg.model.device
        )
        model.eval()
        return model, tokenizer

    @property
    def small(self):
        if self._small is None:
            self._small, self._tokenizer = self._load(self.cfg.model.small_model)
        return self._small

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            _ = self.small
        return self._tokenizer

    @property
    def large(self):
        if self._large is None:
            self._large, self._large_tokenizer = self._load(self.cfg.model.large_model)
        return self._large

    # ---------------------------------------------------------- generation ---

    def _chat(self, tokenizer, system: str, user: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        num_return_sequences: int = 1,
        want_hidden: bool = False,
    ) -> dict[str, Any]:
        import torch

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.95 if temperature > 0 else None,
                num_return_sequences=num_return_sequences,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=want_hidden,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        sequences = out.sequences[:, prompt_len:]
        texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)

        logprobs, entropies, min_prob = [], [], 1.0
        for step, logits in enumerate(out.scores):
            probs = torch.softmax(logits[0].float(), dim=-1)
            token_id = sequences[0, step]
            p = float(probs[token_id].clamp_min(1e-12))
            logprobs.append(float(torch.log(torch.tensor(p))))
            entropies.append(float(-(probs * probs.clamp_min(1e-12).log()).sum()))
            min_prob = min(min_prob, p)

        hidden: list[float] = []
        if want_hidden and getattr(out, "hidden_states", None):
            # Mid-layer representation of the last prompt token: a cheap probe
            # that is strictly free, since this forward pass happens anyway.
            layers = out.hidden_states[0]
            mid = layers[len(layers) // 2]
            hidden = mid[0, -1].float().tolist()

        return {
            "texts": [t.strip() for t in texts],
            "mean_logprob": sum(logprobs) / max(len(logprobs), 1),
            "mean_entropy": sum(entropies) / max(len(entropies), 1),
            "min_token_prob": min_prob,
            "prompt_tokens": prompt_len,
            "completion_tokens": int(sequences.shape[1]),
            "hidden": hidden,
        }

    # -------------------------------------------------------------- gating ---

    def draft(self, example: Example) -> Draft:
        """The shared gating pass. Everything downstream reuses this."""
        prompt = self._chat(self.tokenizer, ANSWER_SYSTEM, example.question)
        with measure(self.cfg.measure_energy) as m:
            out = self._generate(
                self.small,
                self.tokenizer,
                prompt,
                self.cfg.model.max_new_tokens_answer,
                want_hidden=True,
            )
        energy = m.energy_j or fallback_energy(out["completion_tokens"], "small")
        return Draft(
            text=out["texts"][0],
            mean_logprob=out["mean_logprob"],
            mean_entropy=out["mean_entropy"],
            min_token_prob=out["min_token_prob"],
            prompt_tokens=out["prompt_tokens"],
            completion_tokens=out["completion_tokens"],
            hidden=out["hidden"],
            latency_ms=m.latency_ms,
            energy_j=energy,
            peak_memory_mib=m.peak_memory_mib,
        )

    # ------------------------------------------------------------- actions ---

    def run_action(self, action: int, example: Example, draft: Draft) -> ActionResult:
        """Execute one action and return its prediction plus measured costs."""
        action = Action(action)
        if action is Action.ANSWER:
            return self._act_answer(example, draft)
        if action is Action.REASON:
            return self._act_reason(example, draft)
        if action is Action.RETRIEVE:
            return self._act_retrieve(example, draft)
        return self._act_delegate(example, draft)

    def _act_answer(self, example: Example, draft: Draft) -> ActionResult:
        # Free: the draft was already produced by the gating pass.
        return ActionResult(
            action=int(Action.ANSWER),
            prediction=draft.text,
            correct=score(example, draft.text),
            latency_ms=0.0,
            energy_j=0.0,
            peak_memory_mib=draft.peak_memory_mib,
            prompt_tokens=draft.prompt_tokens,
            completion_tokens=draft.completion_tokens,
            meta={"reused_draft": True},
        )

    def _act_reason(self, example: Example, draft: Draft) -> ActionResult:
        from collections import Counter

        prompt = self._chat(self.tokenizer, REASON_SYSTEM, example.question)
        with measure(self.cfg.measure_energy) as m:
            out = self._generate(
                self.small,
                self.tokenizer,
                prompt,
                self.cfg.model.max_new_tokens_reason,
                temperature=self.cfg.model.reason_temperature,
                num_return_sequences=self.cfg.model.reason_samples,
            )
        finals = [extract_final(t) for t in out["texts"]]
        finals = [f for f in finals if f]
        prediction = Counter(finals).most_common(1)[0][0] if finals else draft.text
        agreement = (finals.count(prediction) / len(finals)) if finals else 0.0
        tokens = out["completion_tokens"] * self.cfg.model.reason_samples
        return ActionResult(
            action=int(Action.REASON),
            prediction=prediction,
            correct=score(example, prediction),
            latency_ms=m.latency_ms,
            energy_j=m.energy_j or fallback_energy(tokens, "small"),
            peak_memory_mib=m.peak_memory_mib,
            prompt_tokens=out["prompt_tokens"],
            completion_tokens=tokens,
            meta={"self_consistency": agreement, "paths": len(finals)},
        )

    def _act_retrieve(self, example: Example, draft: Draft) -> ActionResult:
        if self.retriever is None:
            raise RuntimeError("RETRIEVE requested but no retriever was configured")
        with measure(self.cfg.measure_energy) as m:
            passages = self.retriever.search(example.question, self.cfg.retrieval.top_k)
            evidence = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
            user = f"Evidence:\n{evidence}\n\nQuestion: {example.question}"
            out = self._generate(
                self.small,
                self.tokenizer,
                self._chat(self.tokenizer, RETRIEVE_SYSTEM, user),
                self.cfg.model.max_new_tokens_answer,
            )
        prediction = out["texts"][0]
        return ActionResult(
            action=int(Action.RETRIEVE),
            prediction=prediction,
            correct=score(example, prediction),
            latency_ms=m.latency_ms,
            energy_j=m.energy_j or fallback_energy(out["completion_tokens"], "small"),
            peak_memory_mib=m.peak_memory_mib,
            prompt_tokens=out["prompt_tokens"],
            completion_tokens=out["completion_tokens"],
            meta={"num_passages": len(passages)},
        )

    def _act_delegate(self, example: Example, draft: Draft) -> ActionResult:
        large = self.large  # loading also populates self._large_tokenizer
        tokenizer = self._large_tokenizer
        with measure(self.cfg.measure_energy) as m:
            out = self._generate(
                large,
                tokenizer,
                self._chat(tokenizer, REASON_SYSTEM, example.question),
                self.cfg.model.max_new_tokens_reason,
            )
        prediction = extract_final(out["texts"][0]) or out["texts"][0]
        return ActionResult(
            action=int(Action.DELEGATE),
            prediction=prediction,
            correct=score(example, prediction),
            latency_ms=m.latency_ms,
            energy_j=m.energy_j or fallback_energy(out["completion_tokens"], "large"),
            peak_memory_mib=m.peak_memory_mib,
            prompt_tokens=out["prompt_tokens"],
            completion_tokens=out["completion_tokens"],
            meta={"model": self.cfg.model.large_model},
        )


def extract_final(text: str) -> str:
    """Pull the short answer out of a chain-of-thought completion."""
    marker = "answer:"
    lowered = text.lower()
    if marker in lowered:
        return text[lowered.rindex(marker) + len(marker) :].strip().split("\n")[0].strip()
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    return lines[-1] if lines else ""
