"""Controlled edits to reference clinical notes.

Each function takes a note and returns an edited version, or None when the edit does not
apply (for example, a laterality swap on a note that never says left or right).

Edits come in two groups:
  - HARMLESS: the clinical meaning is unchanged (a good metric should barely move).
  - HARMFUL: a small change in words that changes the clinical meaning (a good metric
    should drop clearly, and more than for the harmless edits).
"""
from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text.strip()) if s]


# ---------------------------------------------------------------- harmless edits

def reorder_sentences(text: str) -> str | None:
    """Reverse the order of sentences. Same facts, different order."""
    sents = split_sentences(text)
    if len(sents) < 2:
        return None
    return " ".join(reversed(sents))


# Meaning-preserving rewrites common in clinical notes (pattern, replacement).
_PARAPHRASES = [
    (r"\bThe patient\b", "This patient"),
    (r"\bthe patient\b", "this patient"),
    (r"\bpresents with\b", "comes in with"),
    (r"\bpresents to\b", "comes to"),
    (r"\bdenies\b", "reports no"),
    (r"\bhistory of\b", "past history of"),
    (r"\bcomplains of\b", "reports"),
    (r"\bShe is\b", "She's"),
    (r"\bHe is\b", "He's"),
    (r"\bdoes not\b", "doesn't"),
    (r"\bis not\b", "isn't"),
    (r"\bapproximately\b", "about"),
    (r"\bmother\b", "mom"),
    (r"\bfather\b", "dad"),
    (r"\bmedications\b", "meds"),
    (r"\bstarted\b", "began"),
]


def paraphrase(text: str) -> str | None:
    """Apply every applicable meaning-preserving rewrite. None if nothing applies."""
    out = text
    for pat, rep in _PARAPHRASES:
        out = re.sub(pat, rep, out)
    return out if out != text else None


# ---------------------------------------------------------------- harmful edits

# Each pattern removes or flips one negation; only the first match is changed.
_NEGATION_FLIPS = [
    (r"\bdenies\b", "reports"),
    (r"\bDenies\b", "Reports"),
    (r"\bnegative for\b", "positive for"),
    (r"\bno\s+", ""),
    (r"\bNo\s+", ""),
    (r"\bnot\s+", ""),
    (r"\bnever\s+", ""),
]


def flip_negation(text: str) -> str | None:
    """Remove or flip the first negation ("no fever" becomes "fever")."""
    best = None
    for pat, rep in _NEGATION_FLIPS:
        m = re.search(pat, text)
        if m and (best is None or m.start() < best[0].start()):
            best = (m, rep)
    if best is None:
        return None
    m, rep = best
    out = text[:m.start()] + rep + text[m.end():]
    if rep == "" and m.start() < len(out):  # keep sentence-initial capitalization
        if m.start() == 0 or out[m.start() - 2:m.start()] == ". ":
            out = out[:m.start()] + out[m.start()].upper() + out[m.start() + 1:]
    return out


_DOSE_UNITS = r"(?:\s?(?:mg|mcg|g|mL|ml|units?|tablets?|pills?))"


_NUM = re.compile(r"\b(\d+(?:\.\d+)?)")
_DOSE = re.compile(r"\b(\d+(?:\.\d+)?)" + _DOSE_UNITS + r"\b")
_UNIT = re.compile(r"\b(\d+(?:\.\d+)?)\s?(?:pounds|lbs?|kg|cm|mm|mmHg|%|times|hours?|days?|weeks?|months?|years?)\b")


def _is_list_marker(text: str, start: int, end: int) -> bool:
    """True for enumeration markers such as '1. Hypertension.  2. Diabetes.'

    A marker is a small integer followed by '.' or ')' and a space, placed at the start of the text
    or line, or right after the end of the previous item ('Hypertension. 2. Diabetes').
    """
    if not re.match(r"[.)]\s", text[end:end + 2]) or not text[start:end].isdigit() or int(text[start:end]) > 20:
        return False
    before = text[:start].rstrip(" ")
    return before == "" or before[-1] in ".:;\n"


def change_number(text: str) -> str | None:
    """Change one clinically meaningful number.

    Preference order: a dose (multiplied by 10), then a value with a unit such as weight or
    duration (+20), then the first other number, which in these notes is usually the age (+20).
    Enumeration markers ("1. Hypertension") are skipped, since changing them changes nothing clinical.
    """
    for pattern in (_DOSE, _UNIT, _NUM):
        for m in pattern.finditer(text):
            if _is_list_marker(text, m.start(1), m.end(1)):
                continue
            value = float(m.group(1))
            new = value * 10 if pattern is _DOSE else value + 20
            new_str = str(int(new)) if new.is_integer() else f"{new:g}"
            return text[:m.start(1)] + new_str + text[m.end(1):]
    return None


# "left" or "right" as a side of the body. Excluded on purpose:
#   - handedness ("right-handed"), which rarely changes a clinical decision;
#   - "left" as a verb ("an injury, which has left him with..."), which is not a side at all.
_SIDE = re.compile(
    r"(?<!has )(?<!have )(?<!had )(?<!was )(?<!were )(?<!been )(?<!is )"
    r"\b(left|right|Left|Right)\b(?![- ]handed)"
)


def swap_laterality(text: str) -> str | None:
    """Swap every left and right that refers to a side of the body."""
    if not _SIDE.search(text):
        return None
    table = {"left": "right", "right": "left", "Left": "Right", "Right": "Left"}
    return _SIDE.sub(lambda m: table[m.group(1)], text)


def omit_sentence(text: str) -> str | None:
    """Drop the last sentence (a finding disappears from the note)."""
    sents = split_sentences(text)
    if len(sents) < 2:
        return None
    return " ".join(sents[:-1])


# Plausible but invented findings; one is appended per note, chosen by note index.
_HALLUCINATIONS = [
    "The patient was started on warfarin 5 mg daily.",
    "There is a history of myocardial infarction.",
    "The patient reports chest pain on exertion.",
    "A biopsy was positive for malignancy.",
    "The patient has a known allergy to penicillin.",
]


def add_hallucination(text: str, idx: int = 0) -> str:
    """Append one invented clinical finding."""
    return text.rstrip() + " " + _HALLUCINATIONS[idx % len(_HALLUCINATIONS)]


EDITS = {
    "reorder sentences": ("harmless", reorder_sentences),
    "paraphrase": ("harmless", paraphrase),
    "flip negation": ("harmful", flip_negation),
    "change number": ("harmful", change_number),
    "swap left/right": ("harmful", swap_laterality),
    "omit last sentence": ("harmful", omit_sentence),
    "add invented finding": ("harmful", None),  # handled with the note index
}


def apply_edit(name: str, text: str, idx: int) -> str | None:
    if name == "add invented finding":
        return add_hallucination(text, idx)
    return EDITS[name][1](text)
