"""assemble_training's re-paired dev set."""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import assemble_training as at  # noqa: E402

from matcher.rules import status_problems  # noqa: E402
from matcher.schema import CandidateCard, RequirementCard  # noqa: E402

CARD = RequirementCard(source_posting_id="p1", title="Data Analyst", field="Data analysis", degree_level="bachelor",
                       must_skills=(("Python",),), locations=("Boston, MA",), work_mode="onsite",
                       employment_type="full_time")
GRAD = CandidateCard(field="Data analysis", degree_level="bachelor", degree_field="Statistics", graduation_year=2022,
                     years=3, skills=("Python",), current_location="Boston, MA")


def test_swap_profile_leaves_out_a_status_the_resume_contradicts():
    cpt = replace(GRAD, work_auth="f1_cpt", graduation_year=2028, years=0)  # a student's profile on a graduate's resume
    opt = replace(GRAD, work_auth="f1_opt", graduation_year=2026, years=0)
    citizen = replace(GRAD, work_auth="citizen_or_pr", remote_preference="remote_only")
    opts = at.profile_swaps(GRAD, CARD, [cpt, opt, citizen])
    mixed = [m for ms in opts.values() for m in ms]
    assert [m.work_auth for m in mixed] == ["citizen_or_pr"]
    assert all(not status_problems(m) for m in mixed) and mixed[0].remote_preference == "remote_only"


def test_pick_refuses_a_skewed_set_of_rendered_rows():
    """Texts the gates drop can fall unevenly (r4: no will_not_sponsor_or_opt check posting kept a text): among picked
    train rows with both texts, no stance or variant kind may fall under 80% of its share of the picked rows."""
    cards = {"rq_a": {"sponsorship": "sponsors"}, "rq_b": {"sponsorship": "will_not_sponsor_or_opt"},
             "rq_c": {"sponsorship": "unstated", "citizens_only": True}}
    train = ([{"rq_id": "rq_a", "cd_id": f"a{i}", "variant_kind": "full_match"} for i in range(50)]
             + [{"rq_id": "rq_b", "cd_id": f"b{i}", "variant_kind": "years_short"} for i in range(30)]
             + [{"rq_id": "rq_c", "cd_id": f"c{i}", "variant_kind": "full_match"} for i in range(20)])
    picked = train
    have_p = {"rq_a", "rq_b", "rq_c"}
    shares, why = at.skew_problems(picked, train, {r["cd_id"] for r in train}, have_p, cards)
    assert why == [] and shares["picked_with_text"] == 100
    # half the no-OPT rows lost their resume text: 15 of 85 is 17.6%, under 80% of 30%
    have_r = {r["cd_id"] for r in train} - {f"b{i}" for i in range(15)}
    shares, why = at.skew_problems(picked, train, have_r, have_p, cards)
    assert any("will_not_sponsor_or_opt" in x for x in why) and any("years_short" in x for x in why)
    assert shares["work-authorization stance"]["will_not_sponsor_or_opt"] == {"train_split": 0.3, "picked": 0.3,
                                                                             "picked_with_text": 0.1765}
    # a mild drop (25 of 95 = 26%, 88% of 30%) passes
    have_r = {r["cd_id"] for r in train} - {f"b{i}" for i in range(5)}
    assert at.skew_problems(picked, train, have_r, have_p, cards)[1] == []
    # nothing rendered yet: nothing to judge
    assert at.skew_problems(picked, train, set(), set(), cards) == ({"picked_with_text": 0, "picked": 100}, [])
    # citizens-only is a stance of its own
    assert at.stance_of(cards["rq_c"]) == "citizens_only" and at.stance_of({}) == "unstated"


def test_pick_balanced_spreads_rows_over_the_bases():
    """Equal buckets, round-robin: when Explore fills early the last bases still give their other buckets, no base
    gives more than per_base rows, and n = 0 takes as many as the smallest bucket allows."""
    import random
    from collections import Counter
    rows = [{"pair_id": f"{b}_{k}_{i}", "base_rq_id": b, "bucket": k}
            for b in (f"rq_{j}" for j in range(10)) for k, m in (("For you", 2), ("Explore", 4), ("Excluded", 3))
            for i in range(m)]
    sel = at.pick_balanced(rows, 0, 9, random.Random(0))
    assert Counter(r["bucket"] for r in sel) == {"For you": 20, "Explore": 20, "Excluded": 20}
    assert len({r["base_rq_id"] for r in sel if r["bucket"] == "Explore"}) == 10
    capped = at.pick_balanced(rows, 0, 3, random.Random(0))  # 3 a base: 30 rows, 10 a bucket
    assert max(Counter(r["base_rq_id"] for r in capped).values()) <= 3
    assert Counter(r["bucket"] for r in capped) == {"For you": 10, "Explore": 10, "Excluded": 10}
    assert len(at.pick_balanced(rows, 12, 9, random.Random(0))) == 12


def test_enrollment_repairs_swap_skip_rows_for_graduates_on_student_roles():
    """Re-pairs: a graduated candidate of the split on a student-only posting of the split (and an enrolled one on a
    role for a higher level), label from rules.label, in place of picked Skip rows, buckets unchanged, no stance, kind
    or Skip reason under 80% of its count, at most per_base rows a base."""
    import random
    from collections import Counter

    from synthetic_text import posting_text, resume_text

    from matcher import full_match
    from matcher.rules import label

    stud = RequirementCard(source_posting_id="s1", title="Research Intern", field="Lab and life sciences",
                           degree_level="none", enrolled_in="bachelor", must_skills=(("Python",),), locations=("Boston, MA",),
                           work_mode="onsite", employment_type="internship")
    grad_only = replace(stud, source_posting_id="s2", enrolled_in="master")
    q_cards = {"rq_s1": stud, "rq_s2": grad_only, **{f"rq_f{i}": replace(CARD, source_posting_id=f"f{i}") for i in range(45)}}
    student = full_match(stud)
    cands = {f"cd_g{i}": replace(student, graduation_year=2022 - i % 3) for i in range(6)}  # graduated, want internships
    cands |= {f"cd_b{i}": replace(student, graduation_year=2028) for i in range(4)}  # bachelor's students
    ft = replace(GRAD, employment_types_wanted=("full_time",))  # never a valid re-pair: wants no internship
    picked, kinds = {}, {}
    for i in range(45):  # 15 a bucket on 45 bases: For you, Explore (role avoided), Skip (10 must_skill, 5 location)
        k = ("For you", "Explore", "Excluded")[i % 3]
        skill = i < 30
        picked[f"cd_p{i}"] = {"For you": ft, "Explore": replace(ft, roles_avoided=("analyst",)),
                              "Excluded": replace(ft, skills=()) if skill else replace(ft, current_location="Denver, CO")}[k]
        kinds[f"cd_p{i}"] = {"For you": "full_match", "Explore": "pref_role_avoided"}.get(k) or ("must_skill" if skill else "location")
    cands |= picked
    sel = []
    for i, (cd, c) in enumerate(picked.items()):
        b, reasons, _ = label(c, q_cards[f"rq_f{i}"])
        sel.append({"pair_id": f"p{i}", "rq_id": f"rq_f{i}", "cd_id": cd, "base_rq_id": f"rq_f{i}", "bucket": b,
                    "reasons": reasons, "variant_kind": kinds[cd], "candidate_kind": "copy", "split": "train"})
    assert Counter(r["bucket"] for r in sel) == {"For you": 15, "Explore": 15, "Excluded": 15}
    split_rows = sel + [{"pair_id": f"s{i}", "rq_id": rq, "cd_id": cd, "base_rq_id": rq, "split": "train"}
                        for i, (rq, cd) in enumerate((("rq_s1", "cd_g0"), ("rq_s2", "cd_b0"), *(("rq_s1", c) for c in cands)))]
    cd_cards = {i: c.to_dict() for i, c in cands.items()}
    rq_cards = {i: q.to_dict() for i, q in q_cards.items()}
    resumes = {i: resume_text(c) for i, c in cands.items()}
    postings = {i: posting_text(q) for i, q in q_cards.items()}
    rows, rep = at.enrollment_repairs(sel, split_rows, "train", cd_cards, rq_cards, resumes, postings, 7, random.Random(0), 1.0)
    new = [r for r in rows if r["candidate_kind"] == "enrollment_repair"]
    kept = {r["pair_id"] for r in rows}  # each re-pair sits where the Skip row it replaced sat
    assert [i for i, r in enumerate(rows) if r["candidate_kind"] == "enrollment_repair"] == \
        [i for i, r in enumerate(sel) if r["pair_id"] not in kept]
    # the floor: 10 must_skill Skips keep 8 and 5 location ones keep 4, so three swaps, two graduates to one student
    assert len(new) == 3 and rep["added"] == {"graduated": 2, "level": 1}, rep
    assert Counter(r["bucket"] for r in rows) == Counter(r["bucket"] for r in sel)
    assert Counter(r["variant_kind"] for r in rows if r["bucket"] == "Excluded") == {
        "must_skill": 8, "location": 4, "repair_enrollment_graduated": 2, "repair_enrollment_level": 1}
    for r in new:
        c, q = CandidateCard.from_dict(cd_cards[r["cd_id"]]), RequirementCard.from_dict(rq_cards[r["rq_id"]])
        assert (r["bucket"], r["reasons"]) == label(c, q)[:2] and r["split"] == "train" and r["base_rq_id"] == r["rq_id"]
        assert (r["cd_id"], r["rq_id"]) not in {(x["cd_id"], x["rq_id"]) for x in split_rows}
        assert r["completion"].startswith("Skip — for students currently pursuing")
    by = {r["variant_kind"]: r for r in new}
    assert "you have graduated" in by["repair_enrollment_graduated"]["completion"]
    assert by["repair_enrollment_level"]["rq_id"] == "rq_s2" and "you are pursuing a bachelor's degree" in \
        by["repair_enrollment_level"]["completion"]
    # a share cap: none above max_share, counting the enrollment Skips already there
    assert at.enrollment_repairs(sel, split_rows, "train", cd_cards, rq_cards, resumes, postings, 7, random.Random(0),
                                 0.0)[1]["added"] == {}
    # a full base takes its swap from its own Skip rows: none there, no re-pair
    full = sel + [{"pair_id": f"x{i}", "rq_id": "rq_s1", "cd_id": "cd_p0", "base_rq_id": "rq_s1", "bucket": "For you",
                   "reasons": [], "variant_kind": "full_match", "candidate_kind": "copy", "split": "train"} for i in range(7)]
    got = at.enrollment_repairs(full, split_rows, "train", cd_cards, rq_cards, resumes, postings, 7, random.Random(0), 1.0)[0]
    assert sum(r["base_rq_id"] == "rq_s1" for r in got) == 7
