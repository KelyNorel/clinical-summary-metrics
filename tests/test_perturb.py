"""Each edit must do exactly what it claims: harmless edits keep the facts, harmful ones change one thing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.perturb import (add_hallucination, change_number, flip_negation, omit_sentence,  # noqa: E402
                         paraphrase, reorder_sentences, swap_laterality)
from src.metrics import overlap_scores  # noqa: E402

NOTE = "The patient denies fever. She takes lisinopril 10 mg daily. Pain in the left knee."


def test_reorder_keeps_all_sentences():
    out = reorder_sentences(NOTE)
    assert sorted(out.split(". ")) != [] and set(out.replace(".", "").split()) == set(NOTE.replace(".", "").split())


def test_paraphrase_keeps_negation():
    out = paraphrase(NOTE)
    assert "reports no fever" in out


def test_flip_negation_changes_only_first_negation():
    assert flip_negation(NOTE).startswith("The patient reports fever.")
    assert flip_negation("No chest pain. No cough.") == "Chest pain. No cough."
    assert flip_negation("Lungs are clear.") is None


def test_change_number_scales_doses():
    assert "100 mg" in change_number(NOTE)
    assert change_number("A 55-year-old man.") == "A 75-year-old man."


def test_swap_laterality():
    assert "right knee" in swap_laterality(NOTE)
    assert swap_laterality("No laterality here.") is None


def test_swap_laterality_skips_handedness_and_verb():
    # Handedness and "left" as a verb are not a side of the body, so they are not edited.
    assert swap_laterality("A 56-year-old right-handed female.") is None
    assert swap_laterality("The injury has left him with chronic pain.") is None
    out = swap_laterality("A right-handed man with left hip pain.")
    assert out == "A right-handed man with right hip pain."


def test_omit_and_add():
    assert omit_sentence(NOTE).endswith("10 mg daily.")
    assert add_hallucination(NOTE, 0).startswith(NOTE)


def test_identical_text_scores_100():
    s = overlap_scores(NOTE, NOTE)
    assert all(abs(v - 100) < 1e-6 for v in s.values())


def test_change_number_prefers_dose_and_skips_list_markers():
    # A dose is changed before the age, and multiplied by 10.
    assert change_number("A 55-year-old on Keppra 150 mg b.i.d.") == "A 55-year-old on Keppra 1500 mg b.i.d."
    # A value with a unit (duration) is preferred over the age.
    assert change_number("A 40-year-old with knee pain for 13 years.") == "A 40-year-old with knee pain for 33 years."
    # Enumeration markers are not clinical numbers.
    assert change_number("1. Hypertension.\n2. Diabetes.") is None
    assert change_number("1. Hypertension. 2. Diabetes.") is None
    # A number that ends a sentence is not a marker.
    assert change_number("Mother died at age 58. Father is alive.") == "Mother died at age 78. Father is alive."
