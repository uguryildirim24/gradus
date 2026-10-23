from matcher.pairs import build_prompt, build_rows, Req
from matcher.rules import format_completion, label
from matcher.schema import CandidateCard, RequirementCard
from matcher.splits import HELDOUT, family_of
import random

FIELDS = ["nursing", "software engineering", "accounting", "welding", "biology lab", "teaching"]


def make_reqs(n=60):
    reqs = []
    for i in range(n):
        f = FIELDS[i % len(FIELDS)]
        card = RequirementCard(
            source_posting_id=str(i), title=f"{f} role {i}", field=f,
            onet_soc=f"{10 + (i % len(FIELDS)) * 3}-1011.00",
            degree_level="bachelor", degree_fields=(f,), min_years=i % 4,
            must_skills=((f"{f} a",), (f"{f} b", f"{f} d")), nice_skills=(f"{f} c",),
            locations=("Boston",), work_mode="onsite", industry="health",
            employment_type="internship" if i % 2 else "full_time", salary_max=90000,
            sponsorship="unstated",
        )
        reqs.append(Req(f"rq_{i:04d}", card, "onet", family_of(card.field, card.onet_soc)))
    return reqs


def make_cands(n=300):
    rng = random.Random(1)
    out = []
    for i in range(n):
        f = rng.choice(FIELDS)
        out.append((f"cd_{i:04d}", CandidateCard(
            field=f, degree_level=rng.choice(["bachelor", "master"]), degree_field=f,
            years=rng.randint(0, 8), skills=(f"{f} a", f"{f} b"), current_location="Boston",
            work_auth=rng.choice(["citizen_or_pr", "f1_opt"]))))
    return out


RES = build_rows(make_reqs(), make_cands(), per_card=7)
ROWS = RES["rows"]
REQ = {r.rq_id: r for r in make_reqs()}
CAND = dict(make_cands())


def test_labels_equal_rules():
    vc = {v["id"]: CandidateCard.from_dict(v["card"]) for v in RES["variant_candidates"]}
    from matcher.schema import RequirementCard as RC
    vr = {v["id"]: RC.from_dict(v["card"]) for v in RES["variant_requirements"]}
    assert ROWS
    for r in ROWS:
        cand = CAND.get(r["cd_id"]) or vc[r["cd_id"]]
        card = vr.get(r["rq_id"]) or REQ[r["rq_id"]].card
        b, reasons, flags = label(cand, card)
        assert r["bucket"] == b and r["reasons"] == reasons and r["flags"] == flags
        assert r["completion"] == format_completion(b, reasons)


def test_no_leakage():
    for key in ("base_rq_id", "cd_id"):
        seen = {}
        for r in ROWS:
            if key == "cd_id" and r["cd_id"].startswith("cd_v_"):
                continue
            assert seen.setdefault(r[key], r["split"]) == r["split"], (key, r[key])
    ids = [r["pair_id"] for r in ROWS]
    assert len(ids) == len(set(ids))
    # variant candidates are never shared across splits either
    vs = {}
    for r in ROWS:
        if r["cd_id"].startswith("cd_v_"):
            assert vs.setdefault(r["cd_id"], r["split"]) == r["split"]


def test_heldout_families_never_in_train():
    held = set(RES["heldout_families"])
    assert len(held) == 2
    for r in ROWS:
        if r["family"] in held:
            assert r["split"] == HELDOUT
        else:
            assert r["split"] != HELDOUT
    assert any(r["split"] == HELDOUT for r in ROWS)


def test_bucket_balance_and_size():
    from collections import Counter
    c = Counter(r["bucket"] for r in ROWS)
    n = len(ROWS)
    assert n >= 60 * 6
    for b, t in (("Excluded", .35), ("Explore", .35), ("For you", .30)):
        assert abs(c[b] / n - t) < 0.12, c


def test_build_prompt_truncates():
    p = build_prompt("Candidate\nBoston, MA", "profile", "word " * 10000, max_posting_tokens=100)
    assert p.endswith("Answer:") and "[truncated]" in p and len(p) < 1600
    short = build_prompt("r", "Lives in: Boston", "short posting")
    assert "short posting" in short and "[truncated]" not in short


def test_build_prompt_keeps_work_auth_sentence_past_the_budget():
    filler = "\n\n".join(f"We offer a perk number {i} to every employee on the team." for i in range(400))
    posting = ("Title: Intern\nCompany: Acme\nLocations: SF\n\n" + filler +
               "\n\nWe are unable to sponsor visas for this role.\n\nQualifications\n- Pursuing a bachelor's degree")
    p = build_prompt("Candidate\nBoston, MA", "profile", posting, max_posting_tokens=300)
    assert "unable to sponsor visas" in p and "Pursuing a bachelor" in p and "Title: Intern" in p
    assert len(p.split("JOB POSTING:")[1]) < 1500


def test_build_prompt_cleans_resume_and_posting():
    resume = "Ana Lopez\nana@x.com | 617-555-0100\n\n**SUMMARY**\n*   Ran assays"
    p = build_prompt(resume, "profile", "**Title:** Tech\n\n*   Lift 20 lb")
    assert "ana@x.com" not in p and "Ana Lopez" not in p and "617-555" not in p and "**" not in p
    assert "RESUME:\nCandidate" in p and "- Ran assays" in p and "- Lift 20 lb" in p


def test_build_prompt_order():
    p = build_prompt("Candidate\n\nSUMMARY\nTHE RESUME", "THE PROFILE", "THE POSTING")
    assert p.index("SUMMARY\nTHE RESUME") < p.index("CANDIDATE PROFILE:\nTHE PROFILE") < p.index("JOB POSTING:\nTHE POSTING")


def test_occupations_never_span_splits():
    socs = {}
    for r in ROWS:
        soc = REQ[r["base_rq_id"]].card.onet_soc
        assert socs.setdefault(soc, r["split"]) == r["split"], soc


def test_rows_come_from_all_three_candidate_kinds():
    from collections import Counter
    kinds = Counter(r["candidate_kind"] for r in ROWS)
    assert set(kinds) <= {"real", "repaired", "copy"} and kinds["repaired"] and kinds["copy"]
    for r in ROWS:
        if r["candidate_kind"] == "repaired":
            assert r["repaired_from"] in CAND
        if r["candidate_kind"] != "real":
            assert r["cd_id"].startswith("cd_v_")


def test_no_candidate_has_a_contradictory_status():
    from matcher.rules import status_problems
    from matcher.variants import prefs_fit, unique_skills
    vc = {v["id"]: CandidateCard.from_dict(v["card"]) for v in RES["variant_candidates"]}
    for r in ROWS:
        c = CAND.get(r["cd_id"]) or vc[r["cd_id"]]
        assert not status_problems(c)
        assert prefs_fit(c) and unique_skills(c.skills) == c.skills


def test_fingerprint_tally_flags_a_planted_shortcut():
    from matcher.fingerprints import Tally, worst
    t = Tally()
    buckets = ("For you", "Explore", "Excluded")
    for i in range(300):
        t.add({"work_auth": "citizen_or_pr", "x": str(i % 2)}, buckets[i % 3])
    for i in range(100):
        t.add({"work_auth": "other_visa", "x": str(i % 2)}, "Excluded" if i < 90 else "Explore")
    bad = worst(t)
    assert [(x["feature"], x["value"]) for x in bad] == [("work_auth", "other_visa")]


def test_fit_features_are_reported_not_steered_or_gated():
    from matcher.fingerprints import FIT_FEATURES, Tally, worst
    t = Tally()
    for i in range(100):  # years short is never Keep: the rule, not a shortcut
        t.add({"years_minus_min": "<0", "work_auth": "citizen_or_pr"}, "Excluded" if i < 90 else "Explore")
        t.add({"years_minus_min": "2", "work_auth": "citizen_or_pr"}, "For you")
    assert "years_minus_min" in FIT_FEATURES
    assert worst(t) == [] and any(x["feature"] == "years_minus_min" and x["top_share"] > 0.8 for x in t.table())
    # steering sees only the side feature: the fit feature adds nothing to the score
    assert t.share_after({"years_minus_min": "<0"}, "Excluded") == 0.0


def test_stance_cell_names_only_statuses_the_stance_does_not_decide():
    from matcher.fingerprints import STANCE_DECIDES, stance_cell
    from matcher.schema import RequirementCard
    sponsors = RequirementCard(source_posting_id="x", title="Analyst", sponsorship="sponsors")
    citizens = RequirementCard(source_posting_id="x", title="Analyst", citizens_only=True,
                               sponsorship="will_not_sponsor")
    assert stance_cell(CandidateCard(work_auth="citizen_or_pr"), sponsors) == "citizen / sponsors"
    assert stance_cell(CandidateCard(work_auth="citizen_or_pr"), citizens) == "citizen / citizens only"
    assert stance_cell(CandidateCard(work_auth="f1_cpt"), sponsors) == "F-1 CPT / sponsors"
    assert stance_cell(CandidateCard(work_auth="f1_cpt"), citizens) == STANCE_DECIDES
    assert stance_cell(CandidateCard(work_auth="needs_h1b_now"), sponsors) == STANCE_DECIDES


def test_prompt_parts_returns_the_texts_as_the_prompt_holds_them():
    from matcher.pairs import build_prompt, prompt_parts
    from matcher.textclean import fit_posting, redact, strip_markdown

    resume = "Jane Doe\njane@x.com\n\n**Skills:** Python, SQL"
    profile = "Education: bachelor's degree\nWork authorization: U.S. citizen\n"
    posting = "Title: Analyst\nCompany: Acme\n\n## Requirements\n- SQL\n" + "Duties line. " * 900
    prompt = build_prompt(resume, profile, posting)
    r, p, q = prompt_parts(prompt)
    assert r == redact(strip_markdown(resume)) and p == profile.strip()
    assert q == fit_posting(strip_markdown(posting), 6000) and len(q) <= 6000
    assert build_prompt(r, p, q) == prompt
