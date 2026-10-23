"""Hand audit of Adaption-extracted RequirementCards against the posting text.

Usage:
  audit_extraction.py sample                 -> data/built/audit/sample_ids.txt (seed 2026, stratified)
  audit_extraction.py show START END         -> print card + posting for sample[START:END] for reading
  audit_extraction.py build VERDICTS.jsonl   -> extraction_audit.jsonl, checked_ids.txt, test_cards.jsonl,
                                              stats for the .md
                                              (hand verdicts live in data/built/audit/hand_verdicts.jsonl)

Corrected cards use the v2 card format: must_skills as any-of groups, enrolled_in,
and the will_not_sponsor_or_opt stance. Hand verdicts were written against v1, so
build turns their any-of strings into groups (ANY_OF) and applies the v2-only
fields from V2_FIELDS, each with the posting wording it rests on.

VERDICTS rows are hand-written: {"id", "wrong": {field: [corrected, reason]},
"debatable": {field: [suggested, reason]}, "trust": bool, "note": str}. Any judged
field not listed under wrong/debatable is correct.
"""
import json, random, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from matcher.schema import RequirementCard  # noqa: E402

CARDS = Path("data/built/cards/extracted_real_cards.jsonl")
POSTS = Path("data/raw/ats/postings.jsonl")
OUT = Path("data/built/audit")
SEED = 2026
FIELDS = ["degree_level", "degree_fields", "min_years", "must_skills", "license", "locations",
          "work_mode", "sponsorship", "citizens_only", "employment_type", "salary_min", "salary_max"]
# Fields label() in src/matcher/rules.py reads (salary_min is not read; salary_max is).
LABEL_FIELDS = [f for f in FIELDS if f != "salary_min"]

# Hand-corrected must-skill strings that are any-of lists but do not split cleanly
# on " or " / "Any of:". Everything else with " or " or "Any of:" splits mechanically.
ANY_OF = {
    "Digital or analog or RF testing": ["Digital testing", "Analog testing", "RF testing"],
    "Cloud Platforms (AWS, Azure, or GCP)": ["AWS", "Azure", "GCP"],
    "Data pipelines or ETL/ELT or cloud data warehouses": ["Data pipelines", "ETL", "ELT", "Cloud data warehouses"],
    "Python or equivalent scripting": ["Python", "Scripting"],
    "Circuit simulation (HSPICE or FineSim or equivalent)": ["HSPICE", "FineSim", "Circuit simulation"],
    "Python or Perl or Tcl or C/C++": ["Python", "Perl", "Tcl", "C", "C++"],
    "Python or Linux Shell scripting": ["Python", "Shell scripting"],
    "Unit or end-to-end testing": ["Unit testing", "End-to-end testing"],
    "Programming (C#/.NET or Java or React)": ["C#", ".NET", "Java", "React"],
    "Any of: C++, Linux/Windows, Git, Jira, Jenkins, Perl, Bash, Kornshell":
        ["C++", "Linux", "Windows", "Git", "Jira", "Jenkins", "Perl", "Bash", "Kornshell"],
    "Any of: circuit design, digital electronics, PCB design, signal integrity coursework":
        ["Circuit design", "Digital electronics", "PCB design", "Signal integrity"],
    "Any of: digital logic, CMOS, computer architecture, VLSI, semiconductor devices coursework":
        ["Digital logic", "CMOS", "Computer architecture", "VLSI", "Semiconductor devices"],
    # slash pairs read as either one
    "C/C++": ["C", "C++"],
    "OpenEmbedded/Yocto": ["OpenEmbedded", "Yocto"],
}


def skill_group(text: str) -> list[str]:
    """One hand-corrected must-skill string -> its list of alternatives."""
    if text in ANY_OF:
        return ANY_OF[text]
    if text.lower().startswith("any of:"):
        return [x.strip() for x in text.split(":", 1)[1].split(",") if x.strip()]
    if " or " in text.lower():
        parts = [x.strip() for x in re.split(r"(?i)\s+or\s+", text) if x.strip()]
        assert all("(" not in x and "/" not in x for x in parts), f"add {text!r} to ANY_OF"
        return parts
    return [text]


# v2-only fields the v1 hand verdicts could not express. id prefix -> (changes, why).
_ENROLLED = {
    # graduate students (the 13 degree_level rows plus Capital One's sibling wording)
    "doctoral": ("63ef67dc d6cc7924 e556f0eb ef96a3f9 bebbeff8", "current PhD students"),
    "master": ("e7067c38 ebc1e083 0231a390 949d51a9 0a38034b 60675f9d 73d1f84a 1332c5af",
               "current master's or PhD students"),
    "bachelor": ("20898b37 fe413853 e6fe2e61 bbe59360 eeda0ac4 e53bf9a1 6fe74a2d 25f578d9 3170bd08 "
                 "86615113 2b427c48 9bf4727a d3c0c779 66fcf084 46afb6b3 8028ea71 5123007b 792ba5ee "
                 "a0f4b708 c54c66c0 168be0d0 dfc6d5d9 6d7ae4ff 1e295cb0 a82e8f66 5f3d33fb 90d4be1a "
                 "2a5121b2 db1e4fb4 7bb707f9 bbd050e5 e37213e4 ca124ec8 2c773875 ce5ce805 f670187c "
                 "c2cbd149 5a07cf07 d6a60616 27282ffa 2e6978bb bae50bf6 dffa3f5f a2e915b7 e102b2db c2c6db10",
                 "current bachelor's (or higher) students"),
    # "pursuing a degree" / "enrolled in a college program" with no level, or associate named
    "associate": ("03a8913b 721548ed ac3275e6 dd3566b3 92b36766 aaeb44bf d264e688 bee2e809 e55a7700",
                  "current students in any degree program"),
}
# Second reading (2026-10-18), after v2 disagreed with the truth on 32 postings: every row where truth
# and v2 differ on enrolled_in was read again against the posting. These are the ones the first pass
# missed or set to the wrong level, each with the posting's own words. Same rules as the blueprint:
# the lowest level of study accepted; enrolled with no level named -> "associate"; "or recently
# acquired / obtained", "or equivalent experience" and "preferred" are not an enrollment requirement.
_ENROLLED_REREAD = {
    "adb1e815": ("bachelor", "'Currently pursuing a BS degree ... and enrolled full-time'"),
    "c416ad28": ("bachelor", "'exclusively open to students pursuing a university degree'; 'Currently pursuing a bachelor's'"),
    "c782a0fa": ("bachelor", "'Must be pursuing Bachelor's or Advanced degree ... actively enrolled'"),
    "23548c7d": ("bachelor", "'a 3rd or 4th year Computer Science/Engineering student'"),
    "fc66b12e": ("bachelor", "'looking for college juniors'; graduation date December 2027/Spring 2028"),
    "69430fc1": ("bachelor", "'seeking undergraduate students for internship opportunities'"),
    "f6a196e0": ("bachelor", "'Graduating between December 2027 and June 2028 with a Bachelor's Degree'"),
    "c0d0a7a8": ("bachelor", "'Scheduled to obtain a Bachelor's degree ... by Summer 2028 or Summer 2029'"),
    "c7d3b494": ("bachelor", "'Scheduled to obtain a Bachelor's degree ... by Summer 2028 or Summer 2029'"),
    "721548ed": ("bachelor", "'Bachelors or master's student with a graduation date between December 2027 - May 2028'"),
    "dd3566b3": ("bachelor", "'Must be currently enrolled in a Bachelor's or Master's program'"),
    "67c1b95e": ("associate", "'Be enrolled full-time in college in pursuant of a degree' (no level named)"),
    "fec46c2b": ("associate", "'Currently enrolled in a post-secondary program and returning to studies'"),
    "d73bb505": ("associate", "'Currently enrolled in a post-secondary program' (no level named)"),
    "afe67c2f": ("associate", "'Currently pursuing a degree in ...' (no level named; same wording as Aptiv d264e688)"),
    "c2163298": ("associate", "'Graduation date of December 2027/Spring 2028' for a summer 2027 internship (no level named)"),
    "f9edd21f": ("associate", "'Studying a quantitative degree' (no level named)"),
    "5b7bded6": ("associate", "'we can only move forward with those in their penultimate year' (no level named)"),
    # Third reading: the first pass listed Qumulo under bachelor although the note below already said
    # 'or equivalent experience' means no enrollment requirement (same wording as Mill ac1d9d95).
    "c2cbd149": ("none", "'Pursuing a BS or BA in Computer Science or Computer Engineering or equivalent experience'"),
}
# Read again and left as they were: Rundoo 21ae5c99 (co-op students, no requirement stated), Thales 8d856abc
# (apprentices study after hire), Travelers fbc330bb ('Pursuing or obtained'), Hy-Vee 0a09d266 ('working
# toward, or recently acquired'), Voloridge d429d89b ('Completion of at least three years', graduates fit),
# Risepoint 31553e6b, EMC cb8e36f2 ('junior year preferred'), Mill ac1d9d95, 10a Labs a1bc5739 (postdocs and
# staff too), Graco ac3275e6 / Aptiv d264e688 / Trimble bee2e809 / Fed aaeb44bf (associate), LSEG 60675f9d
# (master).
V2_FIELDS: dict[str, list[tuple[dict, str]]] = {}
for level, (ids, why) in _ENROLLED.items():
    for prefix in ids.split():
        if prefix not in _ENROLLED_REREAD:
            V2_FIELDS.setdefault(prefix, []).append(({"enrolled_in": level}, f"posting is for {why}"))
for prefix, (level, why) in _ENROLLED_REREAD.items():
    V2_FIELDS.setdefault(prefix, []).append(({"enrolled_in": level}, f"second reading: {why}"))
for prefix, why in [
    ("fbc330bb", "Travelers: does not support H-1B, TN or I-983/STEM OPT"),
    ("ac3275e6", "Graco: not eligible for sponsorship now or in the future (e.g. H-1B, TN, F-1 OPT)"),
    ("f6a196e0", "Ankura: not eligible for sponsorship or temporary work authorization (OPT, CPT)"),
    ("0231a390", "General Motors: no immigration support (H1-B, OPT, STEM OPT, CPT, TN, J-1)"),
]:
    V2_FIELDS.setdefault(prefix, []).append(({"sponsorship": "will_not_sponsor_or_opt"}, why))
V2_FIELDS.setdefault("73d1f84a", []).append(
    ({"citizens_only": True}, "Marsh: authorization 'on a permanent basis' means citizen or permanent resident"))
# Not marked (no enrollment requirement): ConductorAI and Altar'd State ("or recent graduate"),
# Mill and Qumulo ("or equivalent experience"; Qumulo set to none above).


def load():
    cards = [json.loads(l) for l in CARDS.open()]
    posts = {}
    for l in POSTS.open():
        d = json.loads(l)
        posts["ats_" + d["simplify_id"]] = d
    return cards, posts


def sample(cards):
    rng = random.Random(SEED)
    ids = sorted(c["id"] for c in cards)
    by = {c["id"]: c["card"] for c in cards}
    chosen, stratum = [], {}
    for name, pred, k in [
        ("sponsors", lambda c: c["sponsorship"] == "sponsors", 5),
        ("citizens_only", lambda c: c["citizens_only"], 15),
        ("will_not_sponsor", lambda c: c["sponsorship"] == "will_not_sponsor", 30),
    ]:
        pool = [i for i in ids if pred(by[i]) and i not in stratum]
        for i in rng.sample(pool, k):
            chosen.append(i); stratum[i] = name
    pool = [i for i in ids if i not in stratum]
    for i in rng.sample(pool, 100 - len(chosen)):
        chosen.append(i); stratum[i] = "random"
    return chosen, stratum


def cmd_sample():
    cards, _ = load()
    chosen, stratum = sample(cards)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample_ids.txt").write_text("".join(f"{i}\t{stratum[i]}\n" for i in chosen))
    print(Counter(stratum.values()))


def cmd_show(a, b):
    cards, posts = load()
    by = {c["id"]: c["card"] for c in cards}
    rows = [l.split("\t") for l in (OUT / "sample_ids.txt").read_text().splitlines()]
    for n, (i, s) in enumerate(rows[a:b], start=a):
        c, p = by[i], posts[i]
        print(f"\n######## [{n}] {i} ({s})")
        print("CARD:", json.dumps({k: c[k] for k in FIELDS}, ensure_ascii=False))
        print(f"POSTING: {p['title']} | {p['company']} | {p['locations']} | sim_sponsorship={p.get('sponsorship')} degrees={p.get('degrees')}")
        print(p["description"][:12000])


def cmd_build(path):
    cards, _ = load()
    by = {c["id"]: c["card"] for c in cards}
    rows = [l.split("\t") for l in (OUT / "sample_ids.txt").read_text().splitlines()]
    stratum = dict(rows)
    V = {}
    for l in open(path):
        if l.strip():
            v = json.loads(l); V[v["id"]] = v
    missing = [i for i, _ in rows if i not in V]
    assert not missing, missing
    out = []
    for i, _ in rows:
        v, card = V[i], by[i]
        verdicts, fixed = {}, dict(card)
        for f in FIELDS:
            if f in v.get("wrong", {}):
                val, why = v["wrong"][f]
                verdicts[f] = {"verdict": "wrong", "extracted": card[f], "corrected": val, "reason": why}
                fixed[f] = val
            elif f in v.get("debatable", {}):
                val, why = v["debatable"][f]
                verdicts[f] = {"verdict": "debatable", "extracted": card[f], "suggested": val, "reason": why}
                fixed[f] = val
            else:
                verdicts[f] = {"verdict": "correct"}
        # Hand verdicts list must skills as flat strings; the card's own are groups.
        fixed["must_skills"] = [skill_group(x) for g in fixed["must_skills"]
                                for x in ([g] if isinstance(g, str) else g)]
        v2 = []
        for prefix, changes in V2_FIELDS.items():
            if i.startswith("ats_" + prefix):
                for change, why in changes:
                    fixed.update(change)
                    v2.append({**change, "why": why})
        RequirementCard.from_dict(fixed)  # validates the v2 card
        label_ok = all(verdicts[f]["verdict"] == "correct" for f in LABEL_FIELDS)
        label_ok_lenient = all(verdicts[f]["verdict"] != "wrong" for f in LABEL_FIELDS)
        out.append({"id": i, "stratum": stratum[i], "fields": verdicts,
                    "label_fields_all_correct": label_ok, "label_fields_none_wrong": label_ok_lenient,
                    "trustworthy": bool(v.get("trust", True)), "note": v.get("note", ""),
                    "v2_fields": v2,
                    "corrected_card": fixed})
    with (OUT / "extraction_audit.jsonl").open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "checked_ids.txt").write_text("".join(r["id"] + "\n" for r in out if r["trustworthy"]))
    with (OUT / "test_cards.jsonl").open("w") as f:
        for r in out:
            if r["trustworthy"]:
                f.write(json.dumps({"id": r["id"], "source": "audit_corrected", "card": r["corrected_card"]},
                                   ensure_ascii=False) + "\n")
    used = {p for p in V2_FIELDS if any(r["id"].startswith("ats_" + p) for r in out)}
    assert used == set(V2_FIELDS), set(V2_FIELDS) - used
    # table
    def tab(rs):
        lines = ["| field | correct | debatable | wrong | accuracy (strict) | accuracy (debatable ok) |", "|---|---|---|---|---|---|"]
        for f in FIELDS:
            c = Counter(r["fields"][f]["verdict"] for r in rs); n = len(rs)
            lines.append(f"| {f} | {c['correct']} | {c['debatable']} | {c['wrong']} | {c['correct']/n:.0%} | {(c['correct']+c['debatable'])/n:.0%} |")
        return "\n".join(lines)
    rnd = [r for r in out if r["stratum"] == "random"]
    stats = {
        "all": (sum(r["label_fields_all_correct"] for r in out), sum(r["label_fields_none_wrong"] for r in out), len(out)),
        "random": (sum(r["label_fields_all_correct"] for r in rnd), sum(r["label_fields_none_wrong"] for r in rnd), len(rnd)),
    }
    json.dump({"stats": stats, "table_all": tab(out), "table_random": tab(rnd),
               "n_trust": sum(r["trustworthy"] for r in out)}, sys.stdout, indent=1)
    print()
    # error listing for writing the report
    for f in FIELDS:
        for r in out:
            fv = r["fields"][f]
            if fv["verdict"] != "correct":
                print(f"{f}\t{fv['verdict']}\t{r['id'][:12]}\t{fv['reason']}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sample": cmd_sample()
    elif cmd == "show": cmd_show(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "build": cmd_build(sys.argv[2])
