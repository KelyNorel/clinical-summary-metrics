"""Text-overlap metrics computed locally (BLEU, ROUGE) plus optional BERTScore and LLM-as-judge.

BERTScore downloads a RoBERTa model from Hugging Face the first time it runs.
The LLM judge calls the Anthropic API and needs ANTHROPIC_API_KEY in the environment.
"""
from __future__ import annotations

import json
import os
import re

import sacrebleu
from rouge_score import rouge_scorer

_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def bleu(candidate: str, reference: str) -> float:
    """Sentence BLEU on a 0-100 scale."""
    return sacrebleu.sentence_bleu(candidate, [reference]).score


def rouge(candidate: str, reference: str) -> dict:
    """ROUGE-1, ROUGE-2 and ROUGE-L F1 on a 0-100 scale."""
    s = _ROUGE.score(reference, candidate)
    return {k: v.fmeasure * 100 for k, v in s.items()}


def overlap_scores(candidate: str, reference: str) -> dict:
    out = {"BLEU": bleu(candidate, reference)}
    r = rouge(candidate, reference)
    out.update({"ROUGE-1": r["rouge1"], "ROUGE-2": r["rouge2"], "ROUGE-L": r["rougeL"]})
    return out


def bertscore(candidates: list[str], references: list[str], model_type: str = "roberta-large") -> list[float]:
    """BERTScore F1 (0-100), rescaled with the library's baseline so unrelated text scores near 0."""
    from bert_score import score
    _, _, f1 = score(candidates, references, model_type=model_type, lang="en",
                     rescale_with_baseline=True, verbose=False)
    return [float(x) * 100 for x in f1]


JUDGE_PROMPT = """You are a physician reviewing a clinical note section written from a doctor-patient visit.

Reference note (correct):
<reference>
{reference}
</reference>

Candidate note:
<candidate>
{candidate}
</candidate>

Rate how clinically accurate the candidate is compared with the reference, on a scale from 1 to 5:
5 = same clinical meaning (wording or order may differ)
4 = minor difference with no clinical impact
3 = missing or added detail with possible clinical impact
2 = clinically important error (wrong negation, dose, side, or invented finding)
1 = mostly wrong

Reply with JSON only: {{"score": <1-5>, "reason": "<one short sentence>"}}"""


def llm_judge(candidate: str, reference: str, model: str = "claude-sonnet-5-5",
              client=None) -> dict:
    """Score a candidate against the reference with an LLM. Returns {"score", "reason"}."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model, max_tokens=2000,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(reference=reference, candidate=candidate)}],
    )
    # Newer models may return a thinking block before the answer, so keep only text blocks.
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(m.group(0)) if m else {"score": None, "reason": text[:200]}
