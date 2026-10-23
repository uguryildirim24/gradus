import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

from matcher import leakage as L

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["For you", "Explore", "Excluded"]


def fixture(n=240, leak=None, seed=0):
    """Labels depend on the pairing: resume skill must match posting skill."""
    rng = random.Random(seed)
    skills = ["python", "welding", "nursing", "accounting", "carpentry", "biology"]
    rows = []
    for i in range(n):
        rs, ps = rng.choice(skills), rng.choice(skills)
        filler = " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "omega", "sigma"]) for _ in range(20))
        label = "For you" if rs == ps else rng.choice(["Explore", "Excluded"])
        resume = f"Experienced in {rs}. {filler} studied in Turkey, speaks Turkish."
        posting = f"We need someone with {ps} skills. {filler}"
        if leak == "posting":
            posting += " " + {"For you": "urgent", "Explore": "flexible", "Excluded": "senior"}[label]
        rows.append({"id": f"r{i}", "resume": resume, "posting": posting, "label": label,
                     "split": "train" if i < n * 0.7 else "dev"})
    return pd.DataFrame(rows)


def run(df):
    tr, dv = df[df.split == "train"], df[df.split == "dev"]
    return {r["baseline"]: r["macro_f1"] for r in L.run_baselines(tr, dv, shuffle_repeats=3)}


def test_leaky_posting_is_caught():
    s = run(fixture(leak="posting"))
    assert s["BoW: posting only"] > 0.9


def test_clean_pairing_halves_weak_full_collapses_on_shuffle():
    s = run(fixture())
    assert s["BoW: posting only"] < 0.5 and s["BoW: resume only"] < 0.5
    assert s["BoW: resume + posting"] <= 0.6  # additive BoW cannot see the interaction
    assert s["majority class"] < 0.4


def test_profile_column_gets_its_own_baselines():
    df = fixture()
    df["profile"] = ["Work arrangement: remote only" if lab == "Explore" else "Work arrangement: any"
                     for lab in df.label]
    s = run(df)
    assert {"BoW: profile only", "BoW: resume + profile", "BoW: resume + profile + posting"} <= set(s)
    assert s["BoW: profile only"] > s["BoW: resume only"]


def test_shuffle_preserves_multiset():
    df = fixture(30)
    assert sorted(L.shuffle_postings(df, 1)) == sorted(df.posting)
    assert L.shuffle_postings(df, 1) != list(df.posting)


def test_apply_swaps_single_pass():
    assert L.apply_swaps("Turkish and Hindi", [("Turkish", "Hindi"), ("Hindi", "French")]) == "Hindi and French"


def test_swap_invariance_rates():
    rows = [("studied in Turkey", "p"), ("no markers", "p"), ("Istanbul University, Turkish", "p")]
    biased = lambda r, p: "Excluded" if "Turkey" in r else "For you"
    fair = lambda r, p: "For you"
    b = L.swap_invariance(biased, rows)
    assert b["country"]["affected"] == 2 and b["country"]["flips"] == 1
    assert b["country"]["flip_rate"] == 0.5
    assert L.swap_invariance(fair, rows)["country"]["flips"] == 0
    assert L.swap_invariance(fair, [("none", "p")])["school"]["flip_rate"] is None


def test_cli_end_to_end(tmp_path):
    df = fixture(150)
    df[["id", "resume", "posting", "label"]].to_csv(tmp_path / "rows.csv", index=False)
    df[["id", "split"]].to_csv(tmp_path / "splits.csv", index=False)
    out = tmp_path / "rep.md"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/leakage_report.py"), "--rows", str(tmp_path / "rows.csv"),
                        "--splits", str(tmp_path / "splits.csv"), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "shuffled-posting control" in text and "| majority class |" in text
