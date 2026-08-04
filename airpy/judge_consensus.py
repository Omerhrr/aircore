"""JudgeConsensus: an LLM-as-judge consensus strategy, entirely inside airpy.

aircore's `ConsensusGroup.strategy` is typed as
`Callable[[Sequence[Any]], Any]` (aircore/consensus.py) -- any callable with
that shape is a valid strategy. `majority`/`unanimous` are suited to
discrete, exact-match outputs (tool results, categorical votes); they are
the wrong tool for free-text model outputs, where three independent calls
to the same prompt almost never produce byte-identical strings even when
they agree in substance (see examples/parallel_consensus.py, which surfaced
this against a real provider).

JudgeConsensus is the free-text-shaped alternative: it asks a model to read
the voters' outputs and either select the best one verbatim, or synthesize
one merged answer out of all of them. It is deliberately NOT part of
aircore -- the runtime's job is to run whatever strategy it's given and fail
the step gracefully if the strategy raises (see scheduler.py's
`_run_consensus_group`, which catches any Exception, not just
ConsensusFailed, specifically so a strategy that makes a real provider
call -- like this one -- fails as a normal step failure instead of
crashing the whole workflow). aircore has no idea what a "judge" or a
"prompt" is; it only knows a strategy is a callable that returns a value
or raises.

Two modes:

- `mode="synthesize"` (the default): the judge reads every answer and
  writes one merged final answer, as if answering the question directly
  -- no "Answer 2", no mention of the voting process. This is what most
  people mean by "consensus": research, coding, documentation, auditing.
- `mode="select"`: the judge picks the single best answer among the
  voters, verbatim. Useful when you want one of the actual candidates
  chosen, not a blend -- e.g. picking the best of several trading
  strategies or patches, where a merged answer wouldn't even be valid.

Two response shapes, chosen automatically:

- Plain text (the default: no `output_schema`, `confidence=False`): the
  original prompt-and-marker approach -- ask for either the answer or the
  literal text "NO CONSENSUS", nothing else.
- Structured (`output_schema` given, and/or `confidence=True`): the judge
  is asked to reply with one JSON object -- `{"consensus": bool, "answer":
  ..., ["confidence": float, "reasoning": str]}` -- parsed and validated
  by structured_output.py, the same pipeline ModelAgent(output_schema=...)
  uses. This is what lets JudgeConsensus reduce over *structured* voter
  outputs (e.g. three ModelAgents each returning a validated dict or
  Pydantic instance) into one typed merged artifact, and what makes
  `confidence`/`reasoning` typed fields (a real float, a real string)
  instead of scraped out of `===CONFIDENCE===`-style text markers.

`describe_last_call()` is an optional, duck-typed hook (see scheduler.py's
`_apply_consensus_strategy`): after a successful call, it returns a dict
describing what happened (mode, judge model, candidate count, decision,
and -- if `confidence=True` -- the judge's self-reported confidence and
reasoning). The runtime surfaces this generically in the journal without
knowing what any of it means; plain function strategies (majority,
unanimous) simply don't have this method, so nothing changes for them.

Usage:

    workflow.consensus(
        researcher, reviewer, professor,
        strategy=JudgeConsensus(provider),
    )

    # pick one candidate verbatim instead of merging
    JudgeConsensus(provider, mode="select")

    # record the judge's confidence/reasoning in the journal (structured)
    JudgeConsensus(provider, confidence=True)

    # merge three structured (dict/Pydantic) voter outputs into one typed
    # artifact instead of text
    JudgeConsensus(provider, output_schema=ReportSummary)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

from .providers import ModelProvider, ModelRequest
from .structured_output import (
    StructuredOutputError,
    extract_json,
    is_pydantic_model,
    schema_from,
    validate_against_schema,
)

SYNTHESIZE_PROMPT_TEMPLATE = """You are reconciling {n} independent answers to the \
same question. They may differ in wording but still agree in substance.

Read all {n} answers below. If they substantively agree, write ONE final \
answer that merges the best of what they say, as if you were answering the \
question directly. Do not mention "Answer 1", "Answer 2", the other \
answers, or that there were multiple answers or a voting process -- just \
give the final answer. If they substantively disagree in a way that can't \
be reconciled, reply with exactly the text: NO CONSENSUS

{numbered_outputs}
"""

SELECT_PROMPT_TEMPLATE = """You are judging {n} independent answers to the \
same question. They may differ in wording but still agree in substance.

Read all {n} answers below. If they substantively agree, reply with the \
single best answer among them, verbatim, and nothing else. If they \
substantively disagree in a way that can't be reconciled, reply with \
exactly the text: NO CONSENSUS

{numbered_outputs}
"""

_PROMPT_TEMPLATES = {
    "synthesize": SYNTHESIZE_PROMPT_TEMPLATE,
    "select": SELECT_PROMPT_TEMPLATE,
}

_STRUCTURED_MODE_INSTRUCTION = {
    "synthesize": (
        'If they substantively agree, merge them into ONE final answer for the '
        '"answer" field, as if you were answering the question directly -- do not '
        'reference "Answer 1", "Answer 2", or that there were multiple answers.'
    ),
    "select": (
        'If they substantively agree, choose the single best answer among them, '
        'verbatim, for the "answer" field.'
    ),
}

_STRUCTURED_PROMPT_TEMPLATE = """You are reconciling {n} independent answers to the \
same question. They may differ in wording but still agree in substance.

Read all {n} answers below. {mode_instruction} Set "consensus" to true if \
they substantively agree, or false if they disagree in a way that can't \
be reconciled -- if false, "answer" may be left null.{confidence_instruction}

{numbered_outputs}

Respond ONLY with valid JSON matching this schema, and nothing else (no \
explanation, no markdown code fences):

{schema_json}
"""

_CONFIDENCE_FIELD_INSTRUCTION = (
    ' Also set "confidence" to a number from 0.0 to 1.0 reflecting how strongly '
    'the source answers agreed, and "reasoning" to one or two sentences explaining it.'
)


class JudgeConsensusFailed(Exception):
    """Raised when the judge itself reports the voters didn't agree --
    either the plain-text NO_CONSENSUS marker, or (structured mode)
    "consensus": false. Distinct from a provider/network error or a
    malformed response (StructuredOutputError), which propagate as
    whatever they are -- the scheduler's broad `except Exception` in
    `_run_consensus_group` handles all of these the same way: fail this
    step, don't crash the workflow."""


def _stringify_output(output: Any) -> str:
    """Renders one voter's output for inclusion in the judge prompt.
    Plain strings pass through unchanged (the common case, and the only
    case before structured ModelAgent outputs existed); dicts/lists and
    Pydantic instances are rendered as indented JSON, so a judge reducing
    over *structured* voter outputs (e.g. three ModelAgents each built
    with output_schema=...) sees readable, well-formed data instead of a
    Python repr."""
    if isinstance(output, str):
        return output
    if hasattr(output, "model_dump"):  # pydantic v2 instance
        return json.dumps(output.model_dump(), indent=2, default=str)
    if hasattr(output, "dict") and not isinstance(output, type):  # pydantic v1 instance
        return json.dumps(output.dict(), indent=2, default=str)
    if isinstance(output, (dict, list)):
        return json.dumps(output, indent=2, default=str)
    return str(output)


class JudgeConsensus:
    """A consensus strategy that delegates the "do these agree" judgment to
    a model call instead of exact-string matching. Constructed with a
    ModelProvider so it can be used with any backend (MockProvider for
    tests, LiteLLMProvider for real judging) exactly like ModelAgent is.

    `prompt_template` (plain-text mode only) receives `n` and
    `numbered_outputs` via `.format()`, so a caller can supply a fully
    custom prompt without editing this file -- doing so opts out of
    `mode` (the given template is used as-is). It has no effect in
    structured mode (`output_schema` given, or `confidence=True`), which
    always builds its own JSON-instructing prompt.

    `no_consensus_marker` (plain-text mode only) is checked against the
    judge's response (stripped, case-insensitive) to decide whether the
    judge reported disagreement; structured mode uses a real "consensus":
    bool field instead, so this has no effect there.
    """

    def __init__(
        self,
        provider: ModelProvider,
        model: str = "mock",
        mode: str = "synthesize",
        prompt_template: Optional[str] = None,
        no_consensus_marker: str = "NO CONSENSUS",
        confidence: bool = False,
        output_schema: Optional[Any] = None,
    ) -> None:
        if mode not in _PROMPT_TEMPLATES:
            raise ValueError(f"mode must be one of {list(_PROMPT_TEMPLATES)}, got {mode!r}")
        self.provider = provider
        self.model = model
        self.mode = mode
        self.prompt_template = prompt_template or _PROMPT_TEMPLATES[mode]
        self.no_consensus_marker = no_consensus_marker
        self.confidence = confidence
        self.output_schema = output_schema
        self._answer_json_schema = schema_from(output_schema) if output_schema is not None else {"type": "string"}
        self._answer_model_cls = output_schema if is_pydantic_model(output_schema) else None
        self._structured = confidence or output_schema is not None
        self._last_call_metadata: Optional[Dict[str, Any]] = None

    def __call__(self, outputs: Sequence[Any]) -> Any:
        self._last_call_metadata = None
        if self._structured:
            return self._call_structured(outputs)
        return self._call_text(outputs)

    def _numbered_outputs(self, outputs: Sequence[Any]) -> str:
        return "\n\n".join(f"Answer {i + 1}:\n{_stringify_output(o)}" for i, o in enumerate(outputs))

    def _call_text(self, outputs: Sequence[Any]) -> Any:
        prompt = self.prompt_template.format(n=len(outputs), numbered_outputs=self._numbered_outputs(outputs))
        response = self.provider.generate(ModelRequest(prompt=prompt, model=self.model))
        answer = (response.content or "").strip()

        if answer.upper() == self.no_consensus_marker.upper():
            raise JudgeConsensusFailed(f"judge found no consensus among {len(outputs)} outputs")

        self._last_call_metadata = {
            "strategy": "JudgeConsensus",
            "mode": self.mode,
            "judge_model": self.model,
            "candidate_count": len(outputs),
            "decision": "synthesized" if self.mode == "synthesize" else "selected",
        }
        return answer

    def _wrapper_schema(self) -> Dict[str, Any]:
        properties = {"consensus": {"type": "boolean"}, "answer": self._answer_json_schema}
        required = ["consensus", "answer"]
        if self.confidence:
            properties["confidence"] = {"type": "number"}
            properties["reasoning"] = {"type": "string"}
            required += ["confidence", "reasoning"]
        return {"type": "object", "properties": properties, "required": required}

    def _call_structured(self, outputs: Sequence[Any]) -> Any:
        wrapper_schema = self._wrapper_schema()
        prompt = _STRUCTURED_PROMPT_TEMPLATE.format(
            n=len(outputs),
            mode_instruction=_STRUCTURED_MODE_INSTRUCTION[self.mode],
            confidence_instruction=_CONFIDENCE_FIELD_INSTRUCTION if self.confidence else "",
            numbered_outputs=self._numbered_outputs(outputs),
            schema_json=json.dumps(wrapper_schema, indent=2),
        )
        response = self.provider.generate(
            ModelRequest(prompt=prompt, model=self.model, response_schema=wrapper_schema)
        )
        content = response.content or ""

        try:
            data = json.loads(extract_json(content))
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(f"judge response was not valid JSON ({exc}): {content!r}") from exc
        validate_against_schema(data, wrapper_schema)

        if not data.get("consensus"):
            raise JudgeConsensusFailed(f"judge found no consensus among {len(outputs)} outputs")

        answer = self._build_answer(data.get("answer"))

        metadata: Dict[str, Any] = {
            "strategy": "JudgeConsensus",
            "mode": self.mode,
            "judge_model": self.model,
            "candidate_count": len(outputs),
            "decision": "synthesized" if self.mode == "synthesize" else "selected",
        }
        if self.confidence:
            metadata["confidence"] = data.get("confidence")
            metadata["reasoning"] = data.get("reasoning")
        self._last_call_metadata = metadata

        return answer

    def _build_answer(self, answer_data: Any) -> Any:
        if self._answer_model_cls is None:
            return answer_data
        try:
            return (
                self._answer_model_cls(**answer_data)
                if isinstance(answer_data, dict)
                else self._answer_model_cls(answer_data)
            )
        except Exception as exc:  # noqa: BLE001 -- pydantic's own ValidationError (v1/v2
            # differ), wrapped the same way structured_output.parse_structured_output does.
            raise StructuredOutputError(
                f"{self._answer_model_cls.__name__} validation failed: {exc}"
            ) from exc

    def describe_last_call(self) -> Optional[Dict[str, Any]]:
        """Optional hook the scheduler duck-types for after a successful
        strategy call (see scheduler.py's `_apply_consensus_strategy`).
        Returns None before any call has been made, or after a call that
        raised (there's nothing to describe -- the failure itself is what
        gets recorded)."""
        return self._last_call_metadata

    def __repr__(self) -> str:
        return f"<JudgeConsensus model={self.model!r} mode={self.mode!r}>"
