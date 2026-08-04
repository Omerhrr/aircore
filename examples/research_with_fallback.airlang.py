"""Bindings for research_with_fallback.airlang -- see airlang-spec-v1.md section 6
and airlang/bindings.py for the sibling-file convention this implements
(<file>.airlang.py next to <file>.airlang, exposing PROVIDERS/TOOLS/SCHEMAS/
CAPABILITIES module-level dicts).

Scripts a deliberately low-confidence judge verdict so the example's
`if confidence < 0.85 { Reviewer }` fallback actually triggers when run
via `ai run`/`ai trace` -- the point of this file is to make that
visible without needing a real provider or API key.
"""

import json

from airpy import MockProvider

_LOW_CONFIDENCE_VERDICT = json.dumps({
    "consensus": True,
    "answer": "The literature and Reddit findings broadly agree, but the sample size is small.",
    "confidence": 0.4,
    "reasoning": "Only two sources, and they hedge differently -- not confident enough to skip review.",
})


def _respond(request):
    # Only the judge's structured call sets response_schema (JudgeConsensus's
    # confidence=True switches it into structured mode) -- voter/reviewer
    # agents' plain single-shot calls never do.
    if request.response_schema:
        return _LOW_CONFIDENCE_VERDICT
    return "a plain research finding"


PROVIDERS = {"mock": MockProvider(response=_respond)}
