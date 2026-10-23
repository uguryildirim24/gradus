"""Score extracted RequirementCards field by field.

Usage:
  score_extraction.py audit [--v2 CARDS.jsonl] [--v1 CARDS.jsonl] [--diff FIELD] [--json OUT]
      Compare against the 100 hand-corrected audit cards (data/built/audit/extraction_audit.jsonl).
      Three columns:
        v1 raw        the original v1 cards against the corrected cards as written (the audit's own view)
        v1 + code     the same v1 cards after clean_card() (soft-skill stoplist, slash pairs, vague degree
                      fields, places, sponsorship cross-check), against corrected cards cleaned the same way
        v2            blueprint v2 cards (already cleaned by parse_card) against the cleaned corrected cards
      "v1 + code" is the fair baseline for the blueprint: it isolates what the new prompt adds on top of
      the code repairs both sides get. All columns use the postings v2 has. These are in-sample numbers
      (prompt and code were tuned on the audit postings). A second small table scores must_skills and
      degree_fields against the raw corrected cards, which the cleaner never touched.
  score_extraction.py fresh VERDICTS.jsonl [--checked CARDS --cards CARDS]
      Hand verdicts on postings outside the audit: one row per posting,
      {"id", "wrong": {field: [corrected, reason]}, "debatable": {field: [suggested, reason]}, "note"};
      a judged field not listed is correct. Prints (A) the tally of the verdicts themselves: the cards
      as they stood when they were read. With --checked (those cards, as saved then) and --cards (the
      cards now), also prints (B): the current cards against the hand-corrected truth, which is the
      checked card with every verdict correction applied. B shows what code changes after the reading did.

must_skills strict: the set of groups equals the corrected set. Lenient: every extracted group shares a
skill with some corrected group and every corrected group shares one with some extracted group (the
group boundaries may differ). degree_fields, locations and license have the same two readings.
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from matcher.blueprint import clean_card, is_vague_field  # noqa: E402
from matcher.schema import skill_key  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from audit_extraction import _ENROLLED_REREAD  # noqa: E402

REREAD = set(_ENROLLED_REREAD)

AUDIT = ROOT / "data/built/audit/extraction_audit.jsonl"
V1 = ROOT / "data/built/cards/extracted_real_cards.jsonl"
V2 = ROOT / "data/built/cards/extracted_real_cards_v2.jsonl"
POSTS = ROOT / "data/raw/ats/postings.jsonl"

FIELDS = ["degree_level", "enrolled_in", "degree_fields", "related_ok", "min_years", "must_skills", "license",
          "locations", "work_mode", "sponsorship", "citizens_only", "employment_type", "salary_min", "salary_max"]
# What label() in src/matcher/rules.py reads (salary_min is not read; related_ok softens the field check).
LABEL_FIELDS = [f for f in FIELDS if f not in ("salary_min", "related_ok")]  # related_ok: the audit never annotated it
LENIENT = {"must_skills", "degree_fields", "locations", "license"}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def key_set(items) -> set[str]:
    return {skill_key(str(x)) for x in items}


def groups(v) -> list[frozenset[str]]:
    return [frozenset(skill_key(x) for x in ([g] if isinstance(g, str) else g)) for g in v]


def jaccard(a: set, b: set) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def compare(field: str, ext: dict, truth: dict) -> tuple[bool, bool]:
    """(strict, lenient) for one field."""
    e, t = ext.get(field), truth.get(field)
    if field == "must_skills":
        eg, tg = groups(e or []), groups(t or [])
        strict = set(eg) == set(tg)
        lenient = strict or (all(any(g & h for h in tg) for g in eg) and all(any(g & h for h in eg) for g in tg))
        return strict, lenient
    if field == "degree_fields":  # "related field" is related_ok's entry, scored there
        e, t = [x for x in e or [] if not is_vague_field(x)], [x for x in t or [] if not is_vague_field(x)]
    if field in ("degree_fields", "locations"):
        a, b = key_set(e or []), key_set(t or [])
        return a == b, a == b or jaccard(a, b) >= 0.5
    if field == "license":
        a, b = skill_key(e) if e else None, skill_key(t) if t else None
        strict = a == b
        return strict, strict or bool(a and b and (a in b or b in a))
    strict = e == t
    return strict, strict


def posting_index() -> dict[str, dict]:
    out = {}
    for l in POSTS.open():
        d = json.loads(l)
        out["ats_" + d["simplify_id"]] = d
    return out


def fix(card: dict, post: dict | None = None) -> dict:
    """clean_card with the posting text when we have it (v1 + code)."""
    if post is None:
        return clean_card(card)
    text = f"Title: {post['title']}\nCompany: {post['company']}\nLocations: {'; '.join(post.get('locations') or [])}\n\n{post['description']}"
    return clean_card(card, text, post.get("locations") or [])


def tally(pairs: list[tuple[str, dict, dict]]) -> dict:
    """pairs: (id, extracted card, truth card) -> per-field counts and all-label-fields-right."""
    per = {f: Counter() for f in FIELDS}
    allright = allright_len = noskills = 0
    tp = fp = fn = spurious = 0
    wrong_ids: dict[str, list[str]] = {f: [] for f in FIELDS}
    for i, e, t in pairs:
        res = {f: compare(f, e, t) for f in FIELDS}
        for f, (s, l) in res.items():
            per[f]["strict"] += s
            per[f]["lenient"] += l
            if not s:
                wrong_ids[f].append(i)
        e_sk = {skill_key(a) for g in e.get("must_skills") or [] for a in ([g] if isinstance(g, str) else g)}
        t_sk = {skill_key(a) for g in t.get("must_skills") or [] for a in ([g] if isinstance(g, str) else g)}
        tp, fp, fn, spurious = tp + len(e_sk & t_sk), fp + len(e_sk - t_sk), fn + len(t_sk - e_sk), spurious + bool(e_sk - t_sk)
        allright += all(res[f][0] for f in LABEL_FIELDS)
        allright_len += all(res[f][1] for f in LABEL_FIELDS)
        noskills += all(res[f][0] for f in LABEL_FIELDS if f != "must_skills")
    return {"n": len(pairs), "per": per, "all_label_strict": allright, "all_label_lenient": allright_len,
            "all_but_skills_strict": noskills, "wrong_ids": wrong_ids,
            "skill_precision": tp / (tp + fp) if tp + fp else 0.0, "skill_recall": tp / (tp + fn) if tp + fn else 0.0,
            "spurious_cards": spurious}


def pct(a: int, n: int) -> str:
    return f"{a / n:.0%}" if n else "-"


def table(cols: dict[str, dict], recall: bool = True, skip: tuple[str, ...] = ()) -> str:
    """recall=False leaves out the recall row, for truth built from the scored cards themselves.
    skip: fields left out (related_ok on the audit truth, which never annotated it)."""
    names = list(cols)
    head = "| field | " + " | ".join(f"{n} strict" if True else n for n in names) + " | " + \
           " | ".join(f"{n} lenient" for n in names) + " |"
    lines = [head, "|---|" + "---|" * (2 * len(names))]
    for f in FIELDS:
        if f in skip:
            continue
        s = [pct(cols[n]["per"][f]["strict"], cols[n]["n"]) for n in names]
        l = [pct(cols[n]["per"][f]["lenient"], cols[n]["n"]) if f in LENIENT else "" for n in names]
        lines.append(f"| {f} | " + " | ".join(s) + " | " + " | ".join(l) + " |")
    for label, key in [("all label fields right (strict)", "all_label_strict"),
                       ("all label fields right (lenient)", "all_label_lenient"),
                       ("all but must_skills right (strict)", "all_but_skills_strict")]:
        s = [f"{cols[n][key]}/{cols[n]['n']} = {pct(cols[n][key], cols[n]['n'])}" for n in names]
        lines.append(f"| **{label}** | " + " | ".join(s) + " | " + " | ".join("" for _ in names) + " |")
    for label, fn_ in [("must skills: share of named skills that are real (precision)", lambda c: f"{c['skill_precision']:.0%}"),
                       *([("must skills: share of real skills named (recall)", lambda c: f"{c['skill_recall']:.0%}")]
                         if recall else []),
                       ("cards with a made-up required skill", lambda c: f"{c['spurious_cards']}/{c['n']}")]:
        lines.append(f"| *{label}* | " + " | ".join(fn_(cols[n]) for n in names) + " | " + " | ".join("" for _ in names) + " |")
    return "\n".join(lines)


def cmd_audit(a) -> None:
    posts = posting_index()
    rows = read_jsonl(AUDIT)
    truth_raw = {r["id"]: r["corrected_card"] for r in rows}
    stratum = {r["id"]: r["stratum"] for r in rows}
    trust = {r["id"]: r["trustworthy"] for r in rows}
    v1 = {r["id"]: r["card"] for r in read_jsonl(Path(a.v1))}
    v2 = {r["id"]: r["card"] for r in read_jsonl(Path(a.v2))} if Path(a.v2).exists() else {}
    truth_clean = {i: fix(t) for i, t in truth_raw.items()}
    sets = {
        "all 100": list(truth_raw),
        "random 50": [i for i in truth_raw if stratum[i] == "random"],
        "trusted 87": [i for i in truth_raw if trust[i]],
    }
    out = {}
    for sname, ids in sets.items():
        if v2:  # every column on the same postings: the ones v2 has
            have = [i for i in ids if i in v2]
            if len(have) != len(ids):
                print(f"note: v2 has {len(have)} of {len(ids)} {sname} postings; every column uses those {len(have)}")
            ids = have
        ids = [i for i in ids if i in v1]
        cols = {"v1 raw": tally([(i, v1[i], truth_raw[i]) for i in ids]),
                "v1+code": tally([(i, fix(v1[i], posts[i]), truth_clean[i]) for i in ids])}
        if v2:
            cols["v2"] = tally([(i, v2[i], truth_clean[i]) for i in ids])
        print(f"\n### audit set: {sname} (n={len(ids)}), IN SAMPLE: the v2 prompt and the code were tuned on these postings\n")
        print(table(cols, skip=("related_ok",)))
        print("\nThe audit truth is the v1 cards corrected by hand, so must_skills naming, grouping and recall lean "
              "toward v1 (v1 raw and v1+code). Do not decide v1 vs v2 on must_skills exact match: two copies of the "
              "same posting sent to the model disagree on must_skills about 1 time in 6, and the truth carries soft "
              "and junk entries. related_ok is left out: the truth never annotated it.")
        # enrolled_in: the truth was read a second time where v2 disagreed (audit_extraction._ENROLLED_REREAD),
        # so the full-set number is partly fitted to v2. Score it again without those postings.
        before = [i for i in ids if i[4:12] not in REREAD]
        er = {n: sum(compare("enrolled_in", src, truth_raw[i] if n == "v1 raw" else truth_clean[i])[0]
                     for i, src in ((i, v1[i] if n == "v1 raw" else fix(v1[i], posts[i]) if n == "v1+code" else v2[i])
                                    for i in before))
              for n in cols}
        print(f"\nenrolled_in without the {len(ids) - len(before)} postings whose truth was re-read after seeing v2 "
              f"(n={len(before)}): " + ", ".join(f"{n} {pct(v, len(before))}" for n, v in er.items()) +
              ". v1 cards never carried enrolled_in, so v1's number only counts postings that need none.")
        # The cleaned truth above went through the same clean_card being scored, so a cleaner bug
        # cannot show there. Against the corrected cards as written:
        raw_cols = {"v1+code": tally([(i, fix(v1[i], posts[i]), truth_raw[i]) for i in ids])}
        if v2:
            raw_cols["v2"] = tally([(i, v2[i], truth_raw[i]) for i in ids])
        print(f"\nagainst the raw (uncleaned) corrected cards, {sname}:\n")
        print("| field | " + " | ".join(f"{n} strict | {n} lenient" for n in raw_cols) + " |")
        print("|---|" + "---|---|" * len(raw_cols))
        for f in ("must_skills", "degree_fields"):
            print(f"| {f} | " + " | ".join(f"{pct(c['per'][f]['strict'], c['n'])} | {pct(c['per'][f]['lenient'], c['n'])}"
                                         for c in raw_cols.values()) + " |")
        out[sname] = {n: {"n": c["n"], "per": {f: dict(v) for f, v in c["per"].items()},
                          "all_label_strict": c["all_label_strict"], "all_label_lenient": c["all_label_lenient"],
                          "all_but_skills_strict": c["all_but_skills_strict"], "skill_precision": c["skill_precision"],
                          "skill_recall": c["skill_recall"], "spurious_cards": c["spurious_cards"]} for n, c in cols.items()}
        out[sname]["enrolled_in_not_reread"] = {"n": len(before), **er}
        if a.diff and sname == "all 100":
            col = cols["v2"] if v2 else cols["v1+code"]
            src = v2 if v2 else v1
            for i in col["wrong_ids"][a.diff]:
                print(f"\n{a.diff} {i[:12]}\n  got  {fix(src[i], posts[i]) [a.diff] if not v2 else src[i][a.diff]}\n  want {truth_clean[i][a.diff]}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


def hand_truth(verdicts: list[dict], checked: dict[str, dict]) -> dict[str, dict]:
    """Checked card + every correction (wrong) and suggestion (debatable) from the verdicts."""
    out = {}
    for r in verdicts:
        card = dict(checked[r["id"]])
        for f, (val, _) in {**r.get("debatable", {}), **r.get("wrong", {})}.items():
            card[f] = val
        out[r["id"]] = card
    return out


def cmd_fresh(a) -> None:
    rows = read_jsonl(Path(a.verdicts))
    if a.checked and a.cards:
        checked = {r["id"]: r["card"] for r in read_jsonl(Path(a.checked))}
        now = {r["id"]: r["card"] for r in read_jsonl(Path(a.cards))}
        truth = hand_truth(rows, checked)
        t = tally([(i, now[i], truth[i]) for i in truth if i in now])
        print(f"\n### (B) current cards vs hand-corrected truth, fresh postings (n={t['n']})\n")
        print(table({"v2 now": t}, recall=False))
        print("\nNo recall row: this truth was built from these v2 cards plus the omissions noticed while reading, "
              "so it is not independent of them.")
        if a.diff:
            for i in t["wrong_ids"][a.diff]:
                print(f"\n{a.diff} {i[:12]}\n  got  {now[i][a.diff]}\n  want {truth[i][a.diff]}")
        print("\n### (A) verdicts as given when the cards were read")
    n = len(rows)
    per = {f: Counter() for f in FIELDS}
    allr = allr_len = 0
    for r in rows:
        bad = set(r.get("wrong", {}))
        deb = set(r.get("debatable", {}))
        for f in FIELDS:
            per[f]["strict"] += f not in bad and f not in deb
            per[f]["lenient"] += f not in bad
        allr += not any(f in bad or f in deb for f in LABEL_FIELDS)
        allr_len += not any(f in bad for f in LABEL_FIELDS)
    cols = {"v2 fresh": {"n": n, "per": per, "all_label_strict": allr, "all_label_lenient": allr_len,
                         "all_but_skills_strict": sum(not any(f in set(r.get("wrong", {})) | set(r.get("debatable", {}))
                                                              for f in LABEL_FIELDS if f != "must_skills") for r in rows)}}
    print(f"\n### fresh postings, hand-checked (n={n})\n")
    cols["v2 fresh"]["per"] = per
    # lenient column for every field here: debatable is allowed
    names = list(cols)
    lines = ["| field | strict | lenient (debatable ok) |", "|---|---|---|"]
    for f in FIELDS:
        lines.append(f"| {f} | {pct(per[f]['strict'], n)} | {pct(per[f]['lenient'], n)} |")
    lines.append(f"| **all label fields right** | {allr}/{n} = {pct(allr, n)} | {allr_len}/{n} = {pct(allr_len, n)} |")
    lines.append(f"| **all but must_skills right** | {cols['v2 fresh']['all_but_skills_strict']}/{n} = "
                 f"{pct(cols['v2 fresh']['all_but_skills_strict'], n)} | |")
    print("\n".join(lines))
    if a.json:
        Path(a.json).write_text(json.dumps({"n": n, "per": {f: dict(v) for f, v in per.items()},
                                            "all_label_strict": allr, "all_label_lenient": allr_len}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("audit")
    s.add_argument("--v1", default=str(V1))
    s.add_argument("--v2", default=str(V2))
    s.add_argument("--diff")
    s.add_argument("--json")
    f = sub.add_parser("fresh")
    f.add_argument("verdicts")
    f.add_argument("--checked")
    f.add_argument("--cards")
    f.add_argument("--diff")
    f.add_argument("--json")
    a = ap.parse_args()
    (cmd_audit if a.cmd == "audit" else cmd_fresh)(a)


if __name__ == "__main__":
    main()
