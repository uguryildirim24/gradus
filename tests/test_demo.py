"""Tests for the non-model parts of the demo: resume text in, profile, run B's prompt, the decider, the H'' rules,
buckets, server, scorer API."""
import io
import json
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_pool as bp  # noqa: E402
import demo_common as dc  # noqa: E402
import demo_lib as lib  # noqa: E402
import server as srv  # noqa: E402

from matcher.decider import T_HI, T_LO, keep_score  # noqa: E402
from matcher.prompt_b import prompt_b  # noqa: E402
from matcher.profile import render_profile  # noqa: E402

FORM = {"work_auth": "f1_cpt", "current_location": "Boston, MA", "relocate": "anywhere_us", "remote_preference": "any",
        "employment_types": ["internship"], "salary_floor": "25000", "industries_wanted": "", "industries_avoided": "",
        "roles_avoided": "", "degree_level": "bachelor", "degree_field": "Computer Science",
        "graduation_year": "2028", "years": "1"}


def posting(pid="p1", **card):
    base = {"source_posting_id": pid, "title": "Software Intern", "employment_type": "internship", "work_mode": "onsite",
            "locations": ["Boston, MA"], "must_skills": [["Rust"], ["Kubernetes"]], "degree_level": "bachelor",
            "min_years": 5, "industry": "Software", "seniority": "entry"}
    base.update(card)
    return {"id": pid, "company": "Acme", "title": "Software Intern", "locations": base["locations"], "url": "http://x",
            "text": "We build things. " * 40, "card": base}


# ---- resume text in

def test_redact_removes_contact_details_and_name():
    text = ("Alex Morgan\nalex.morgan@example.com | (617) 555-0142 | linkedin.com/in/alexmorgan | Boston, MA\n"
            "12 Elm Street, Apt 4\nSkills: Python 2019 - 2023")
    out = lib.redact(text)
    assert "Alex Morgan" not in out and "@" not in out and "555" not in out and "linkedin" not in out
    assert "Elm Street" not in out
    assert "Boston, MA" in out and "Python 2019 - 2023" in out
    assert not out.startswith("|")


def test_redact_keeps_a_heading_first_line():
    assert lib.redact_real("EDUCATION\nB.S. Biology").startswith("EDUCATION")
    # the prompt's resume opens with "Candidate" as every training resume does (textclean.redact)
    assert lib.redact("EDUCATION\nB.S. Biology") == "Candidate\nEDUCATION\nB.S. Biology"


@pytest.mark.parametrize("text,name", [
    ("Şebnem Çelik\nBoston, MA\nSkills: Python", "Şebnem Çelik"),
    ("José Álvarez\nSkills: Python", "José Álvarez"),
    ("Resume\nJordan Lee\nSkills: SQL", "Jordan Lee"),
    ("Maria de la Cruz\nSkills: R", "Maria de la Cruz"),
    ("Jordan Lee, B.S.\nSkills: R", "Jordan Lee"),
    ("Dr. Ana María López-García\nResearch", "López"),
    ("JORDAN LEE\nSkills", "JORDAN LEE"),
])
def test_redact_names_in_any_script_and_shape(text, name):
    out = lib.redact(text)
    assert name not in out and out.endswith(text.split("\n")[-1])
    assert "Boston, MA" in out if "Boston" in text else True


def test_redact_name_elsewhere_and_phone_shapes():
    out = lib.redact("Jordan Lee\n6175550142\n617.555.0142 or 1-617-555-0142 x12\n+1 617 555 0142\n"
                     "Phone: 617-555-0142 | Email: j@x.com\nSkills\nReferences: Jordan Lee")
    assert "Jordan" not in out and "555" not in out and "0142" not in out and "x12" not in out
    assert " or" not in out and "Phone" not in out and "Email" not in out and "Skills" in out


@pytest.mark.parametrize("text,left", [
    ("Jordan Lee | jl@x.com | (617) 555-0142 | Boston, MA\nSkills: Python", "Boston, MA\nSkills: Python"),
    ("Jordan Lee — Boston, MA\nSkills", "Boston, MA\nSkills"),
    ("Jordan Lee – Boston, MA\nSkills", "Boston, MA\nSkills"),
    ("Jordan Lee · jl@x.com\nSkills", "Skills"),
    ("Jordan Lee, Boston, MA\nSkills", "Boston, MA\nSkills"),
    ("Lee, Jordan\nSkills\nReferences: Jordan Lee", "Skills\nReferences:"),
    ("JORDAN LEE (he/him)\nSkills", "Skills"),
    ("Jordan Lee | Software Engineering Intern\nSkills", "Software Engineering Intern\nSkills"),
    ("Şebnem Çelik | sebnem@x.com | Somerville, MA\nSkills\nReferences: Şebnem Çelik", "Somerville, MA\nSkills\nReferences:"),
    ("Resume\nJordan Lee | (617) 555-0142\nSkills", "Skills"),
])
def test_redact_name_sharing_a_line_with_contact_details(text, left):
    # the name line becomes "Candidate", as in training resumes (textclean.redact drops a "Resume" title)
    assert lib.redact(text) == "Candidate\n" + left


@pytest.mark.parametrize("text", ["Boston, Massachusetts\nSkills", "New York, NY\nSkills", "Skills\nJava, Python\nMore",
                                  "Python, SQL\nSkills", "Harvard University | Cambridge, MA",
                                  "Skills\nMachine Learning | Computer Vision"])
def test_redact_keeps_places_and_skill_lists(text):
    # no name found: everything stays under the "Candidate" header (a first line "New York, NY" or "Harvard
    # University" is not read as a name)
    assert lib.redact_real(text) == text and lib.redact(text) == "Candidate\n" + text


@pytest.mark.parametrize("text", ["Harvard University\nEDUCATION\nB.S.", "Lehigh University\nB.S. Chemistry",
                                  "Software Engineering Intern\nAcme", "Dean's List 2024\nGPA 3.9"])
def test_redact_keeps_institution_and_title_lines(text):
    assert lib.redact_real(text) == text and lib.redact(text) == "Candidate\n" + text


def _docx(paragraphs):
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f'<w:p><w:r><w:t>{p}</w:t></w:r></w:p>' for p in paragraphs)
    xml = f'<?xml version="1.0"?><w:document xmlns:w="{ns}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def _docx_xml(body):
    ns = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
          'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
          'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
          'xmlns:v="urn:schemas-microsoft-com:vml"')
    xml = f'<?xml version="1.0"?><w:document {ns}><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def test_docx_text_box_is_read_once():
    box = '<w:txbxContent><w:p><w:r><w:t>BOXTEXT</w:t></w:r></w:p></w:txbxContent>'
    body = ('<w:p><w:r><w:t>Before</w:t></w:r><w:r><mc:AlternateContent>'
            f'<mc:Choice Requires="wps"><w:drawing><wps:wsp><wps:txbx>{box}</wps:txbx></wps:wsp></w:drawing></mc:Choice>'
            f'<mc:Fallback><w:pict><v:shape><v:textbox>{box}</v:textbox></v:shape></w:pict></mc:Fallback>'
            '</mc:AlternateContent></w:r></w:p><w:p><w:r><w:t>After</w:t></w:r></w:p>')
    assert lib.extract_text("cv.docx", _docx_xml(body)) == "Before\nBOXTEXT\nAfter"


def test_corrupt_docx_is_a_plain_value_error():
    for name, data in (("bad.docx", b"just some text, not a zip"), ("nodoc.docx", _zip_without_document())):
        with pytest.raises(ValueError, match="could not read"):
            lib.extract_text(name, data)


def test_corrupt_pdf_is_a_plain_value_error():
    pytest.importorskip("pypdf")  # pypdf is in the optional `demo` group
    for data in (b"%PDF-1.4 broken", b"not a pdf at all"):
        with pytest.raises(ValueError, match="could not read"):
            lib.extract_text("bad.pdf", data)


def _zip_without_document():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("other.xml", "<x/>")
    return buf.getvalue()


def test_docx_and_text_extraction():
    assert lib.extract_text("cv.docx", _docx(["Line one", "Line two"])) == "Line one\nLine two"
    assert lib.extract_text("cv.txt", b"hello world resume") == "hello world resume"
    with pytest.raises(ValueError):
        lib.extract_text("cv.exe", b"x")
    with pytest.raises(ValueError):
        lib.extract_text("cv.txt", b"   ")


# ---- profile form

def test_profile_comes_from_render_profile():
    card = lib.candidate_from_form(FORM)
    assert lib.profile_text(FORM) == render_profile(card)
    assert "Willing to relocate to: anywhere in the US" in lib.profile_text(FORM)
    assert "Minimum salary: $25,000 a year" in lib.profile_text(FORM)


def test_relocation_must_be_explicit():
    with pytest.raises(ValueError):
        lib.candidate_from_form({**FORM, "relocate": "list", "willing_locations": ""})
    c = lib.candidate_from_form({**FORM, "relocate": "list", "willing_locations": "New York, NY; Seattle, WA"})
    assert c.willing_locations == ("New York, NY", "Seattle, WA")
    assert lib.candidate_from_form({**FORM, "relocate": "nowhere"}).willing_locations == ()
    c = lib.candidate_from_form({**FORM, "relocate": "list", "willing_locations": "Boston, MA, Denver, CO"})
    assert c.willing_locations == ("Boston, MA", "Denver, CO")


def test_form_validation_and_degree_fields():
    with pytest.raises(ValueError):
        lib.candidate_from_form({**FORM, "work_auth": "tourist"})
    c = lib.candidate_from_form({**FORM, "salary_floor": ""})
    assert c.salary_floor is None and c.graduation_year == 2028 and c.years == 1
    assert c.degree_field == "Computer Science" and c.employment_types_wanted == ("internship",)


# ---- H'' before the model: the rules on the posting's extracted card (matcher.hybrid, as fit_select_cycle3 scores it)

def hp_skip(form, row):
    v = lib.elig_verdict(lib.candidate_from_form(form), row)
    return lib.elig_skip(v, lib.elig_cats(form))


def test_demo_uses_the_shipped_h_double_prime():
    import fit_select_cycle3 as fsc
    import hybrid_rules as hr
    from matcher import hybrid
    assert hr.rule_verdict is hybrid.rule_verdict and hr.hybrid is hybrid.hybrid is fsc.hybrid
    assert lib.ELIG_CATS == fsc.VARIANTS["H''_without_work_mode"] and lib.ELIG_SALARY_CAP is True
    assert sorted(lib.ELIG_CATS) == ["degree_field", "degree_level", "enrollment", "location", "work_auth"]
    sel = json.loads((ROOT / "eval/selection_cycle3.json").read_text())
    assert sel["pick"]["system"] == "B+H''_without_work_mode|decider"
    assert sel["definitions"]["H''_without_work_mode"] == sorted(lib.ELIG_CATS) + ["salary cap"]


def test_decider_is_the_picks_readout():
    import numpy as np

    from matcher import evalstats
    sel = json.loads((ROOT / "eval/selection_cycle3.json").read_text())["pick"]["cut_offs"]
    assert (T_LO, T_HI) == (sel["t_lo"], sel["t_hi"])  # exact, not rounded
    rng = np.random.default_rng(0)
    for row in rng.normal(20, 4, size=(2000, 3)).astype(np.float32).astype(np.float64):
        p = evalstats.softmax(row, 1.0)
        assert keep_score(row) == p[0] + 0.5 * p[1]
    assert dc.decider_label({"Keep": 30.0, "Explore": 20.0, "Skip": 20.0}) == "Keep"
    assert dc.decider_label({"Keep": 20.0, "Explore": 30.0, "Skip": 20.0}) == "Explore"  # s = 0.5
    assert dc.decider_label({"Keep": 20.0, "Explore": 20.0, "Skip": 30.0}) == "Skip"


def test_elig_skip_on_citizens_only_depends_on_work_auth():
    row = posting(citizens_only=True)
    assert "citizens" in hp_skip(FORM, row)
    assert hp_skip({**FORM, "work_auth": "citizen_or_pr"}, row) is None


def test_elig_skips_on_degree_level_and_field_but_never_on_skills_or_years():
    assert hp_skip(FORM, posting()) is None  # Rust/Kubernetes missing and 5 years asked: the model's call
    assert "master's" in hp_skip(FORM, posting(degree_level="master"))
    assert "degree field" in hp_skip(FORM, posting(degree_fields=["Mechanical Engineering"]))
    # the shown reason is the H'' one only, never the skill or years misses beside it
    assert hp_skip(FORM, posting(degree_level="master")) == "needs a master's degree or higher; you are pursuing a bachelor's degree"


def test_elig_skips_on_location_and_enrollment_never_on_job_type():
    assert hp_skip(FORM, posting(employment_type="full_time")) is None  # H'' leaves the job type to the model
    far = posting(locations=["Austin, TX"])
    assert hp_skip({**FORM, "relocate": "nowhere"}, far)
    assert hp_skip(FORM, far) is None  # anywhere in the US
    assert hp_skip({**FORM, "relocate": "list", "willing_locations": "Austin, TX"}, far) is None
    grad = posting(enrolled_in="bachelor")
    assert hp_skip(FORM, grad) is None  # FORM is a bachelor's student graduating 2028
    assert "graduated" in hp_skip({**FORM, "graduation_year": "2024"}, grad)


def test_an_empty_form_field_never_fires_an_override():
    # a blank graduation year reads "not a student" to the rules, a blank field of study matches no field,
    # a blank "Lives in" matches no place: none of them may Skip a posting
    assert hp_skip(FORM, posting(enrolled_in="bachelor")) is None
    assert "not a student" in lib.elig_verdict(lib.candidate_from_form({**FORM, "graduation_year": ""}),
                                                  posting(enrolled_in="bachelor"))["reasons"][0]
    assert hp_skip({**FORM, "graduation_year": ""}, posting(enrolled_in="bachelor")) is None
    assert hp_skip({**FORM, "degree_field": ""}, posting(degree_fields=["Mechanical Engineering"])) is None
    away = posting(locations=["Austin, TX"])
    assert hp_skip({**FORM, "current_location": "", "relocate": "nowhere"}, away) is None
    assert hp_skip({**FORM, "current_location": "", "relocate": "list", "willing_locations": "Denver, CO"}, away) is None
    no_level = {k: v for k, v in FORM.items() if k != "degree_level"}
    assert hp_skip(no_level, posting(degree_level="master")) is None
    # a candidate with no degree is not enrolled whatever the graduation year says: that Skip still fires
    assert "not a student" in hp_skip({**FORM, "degree_level": "none", "graduation_year": "", "degree_field": "",
                                       "work_auth": "citizen_or_pr"}, posting(enrolled_in="bachelor"))


def test_elig_never_skips_on_work_mode_thin_or_not():
    remote = {**FORM, "remote_preference": "remote_only"}
    v = lib.elig_verdict(lib.candidate_from_form(remote), posting())
    assert v["rule"] == "Skip" and "work_mode" in v["hard_cats"]  # the rules Skip it ...
    assert hp_skip(remote, posting()) is None  # ... and H'' without work mode leaves it to the model
    head = "Title: Intern\nCompany: Acme\nLocations: Boston, MA\n\n"
    teaser = posting(must_skills=[], degree_level="none", min_years=0)
    assert hp_skip(remote, {**teaser, "text": head + "Join our team this summer."}) is None
    assert lib.posting_body({"text": head + "Body here."}) == "Body here."


def test_elig_work_auth_skips_include_sponsorship_and_opt():
    row = posting(sponsorship="will_not_sponsor_or_opt")
    assert "OPT" in hp_skip({**FORM, "work_auth": "f1_opt", "graduation_year": "2026"}, row)
    assert hp_skip(FORM, row) is None  # CPT on an internship stays eligible


def test_salary_cap_runs_after_the_model():
    row = posting(salary_max=20000, min_years=0)  # Rust and Kubernetes still unmet on the card
    cand, cats = lib.candidate_from_form(FORM), lib.elig_cats(FORM)  # floor 25,000
    v = lib.elig_verdict(cand, row)
    assert lib.elig_skip(v, cats) is None and v["salary_soft"]  # skills left out: the form cannot carry them
    p = {"Keep": 0.8, "Explore": 0.15, "Skip": 0.05}
    c = lib.scored_card(row, label_word="Keep", p=p, reason="fits", source="model", scores=None, verdict=v, cats=cats)
    assert c["label"] == "Explore" and c["bucket"] == "Explore" and c["reason"].startswith("pay tops out at $20,000")
    assert c["rule"]["kind"] == "salary_cap" and c["rule"]["model_label"] == "Keep" and c["rule"]["mark"] == lib.RULE_MARK
    for word in ("Explore", "Skip"):  # a cap, never a raise
        c = lib.scored_card(row, label_word=word, p=p, reason="r", source="model", scores=None, verdict=v, cats=cats)
        assert c["label"] == word and "rule" not in c


@pytest.mark.parametrize("text", [
    "Students on an F1 visa (including those with CPT or OPT authorization) are not eligible for this role.",
    "Individuals who need sponsorship (for example, following F-1 OPT/CPT, or J-1 status) are not eligible.",
    "We're unable to sponsor or take over sponsorship for any type of employment visa, including F-1 (OPT/CPT).",
    "Candidates must have permanent U.S. work authorization; we do not sponsor or accept student visas (e.g., F-1).",
    "F1 Visa on CPT accepted only for masters-level students.",
    "GM does not provide immigration-related sponsorship. This includes any work authorization requiring "
    "immigration support from the company (e.g., H1-B, OPT, STEM OPT, CPT, TN, J-1, etc.)",
    "This position requires access to information subject to ITAR.",
    "Must have the ability to work in the United States for an unlimited amount of time without sponsorship.",
    "You must have legal authorization to work in the U.S. on a permanent basis.",
    "Minimum Clearance Required to Start: Public Trust",
])
def test_pool_leaves_out_hidden_citizens_only_postings(text):
    assert bp.refuses_f1({"citizens_only": False}, {"description": "We build things. " * 20 + text})
    assert not bp.refuses_f1({"citizens_only": True}, {"description": text})  # the H'' rules handle these


@pytest.mark.parametrize("text", [
    "F-1 students are eligible if they are using Curricular Practical Training (CPT).",
    "CPT & Pre-Opt candidates invited to apply.\nCompeer is unable to sponsor an employment visa at this time.",
    "Visa sponsorship is not available for this position.",
    "Must be authorized to work in the United States without employer sponsorship now or in the future.",
])
def test_pool_keeps_postings_open_to_cpt(text):
    assert not bp.refuses_f1({"citizens_only": False}, {"description": text})


def test_cpt_welcome_never_overrides_itar():
    text = "F-1 students are eligible if they are using CPT. This role requires access to ITAR-controlled data."
    assert bp.refuses_f1({"citizens_only": False}, {"description": text})


@pytest.mark.parametrize("text,event", [
    ("This opportunity is reserved for registered 2026 AfroTech attendees as part of our recruiting experience.", True),
    ("Must be attending, or have attended, the Society of Women Engineers Conference in Boston, MA.", True),
    ("Note: This option is reserved for applicants needing assistance/reasonable accommodation related to a disability.",
     False),
    ("A designation reserved for a select group of organizations.", False),
])
def test_pool_leaves_out_event_only_postings(text, event):
    assert bp.event_only({"description": "We build things. " * 60 + text}) is event


# ---- prompts and buckets

def test_split_prompt_rebuilds_prompt_b():
    prefix, suffix = lib.split_prompt("My resume text", "Work authorization: x", "The posting body.")
    assert prefix + suffix == prompt_b("My resume text", "Work authorization: x", "The posting body.")
    assert prefix.startswith("You are labeling job-match rows") and "\n\nRESUME:\n" in prefix
    assert prefix.endswith("JOB POSTING:\n") and suffix.endswith("\n\nAnswer:")
    long = "word " * 3000  # B reads the whole posting: nothing is cut
    p2, s2 = lib.split_prompt("r", "p", long)
    assert p2 + s2 == prompt_b("r", "p", long) and s2.count("word") == 3000
    assert p2 == prefix.replace("My resume text", "r").replace("Work authorization: x", "p")
    # a finished prompt (an eval row) splits at the same place, so the demo scores it with the same prefix cache
    assert lib.split_row(prefix + suffix) == (prefix, suffix)


def test_buckets_are_sorted_by_p_keep():
    cards = [
        lib.make_card(posting("a"), label_word="Keep", p={"Keep": 0.6, "Explore": 0.3, "Skip": 0.1}, reason="r", source="model"),
        lib.make_card(posting("b"), label_word="Keep", p={"Keep": 0.9, "Explore": 0.05, "Skip": 0.05}, reason="r", source="model"),
        lib.make_card(posting("c"), label_word="Explore", p={"Keep": 0.2, "Explore": 0.6, "Skip": 0.2}, reason="r", source="model"),
        lib.rule_card(posting("d"), "limited to US citizens or permanent residents",
                      {"reason_cats": {"limited to US citizens or permanent residents": "work_auth"}}, lib.ELIG_CATS),
    ]
    b = lib.to_buckets(cards)
    assert [c["id"] for c in b["For you"]] == ["b", "a"]
    assert [c["id"] for c in b["Explore"]] == ["c"]
    assert [c["id"] for c in b["Excluded"]] == ["d"] and b["Excluded"][0]["source"] == "rules"
    assert b["Excluded"][0]["rule"] == {"hybrid": "H''", "kind": "skip", "categories": ["work_auth"], "mark": lib.RULE_MARK}


# ---- shared scorer pieces

class CharTok:
    """One token per character; the chat frame is '<U>' + prompt + '<A>'."""

    def apply_chat_template(self, msgs, add_generation_prompt, tokenize, enable_thinking):
        assert not enable_thinking and not tokenize
        return "<U>" + msgs[0]["content"] + "<A>"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_plan_prefix_shares_all_but_last_prefix_token():
    prefix = "RESUME abc\nJOB POSTING:\n"
    items = [{"id": "1", "suffix": "alpha\n\nAnswer:"}, {"id": "2", "suffix": "beta\n\nAnswer:"}]
    pre, ids = dc.plan_prefix(CharTok(), prefix, items)
    assert len(pre) == len("<U>" + prefix) - 1
    assert all(i[: len(pre)] == pre for i in ids)
    assert ids[0] == [ord(c) for c in "<U>" + prefix + "alpha\n\nAnswer:<A>"]


def test_plan_prefix_single_item_and_empty():
    pre, ids = dc.plan_prefix(CharTok(), "P:", [{"id": "1", "suffix": "x"}])
    assert pre == ids[0][: len("<U>P:") - 1]
    assert dc.plan_prefix(CharTok(), "P:", []) == ([], [])


def test_label_maths_and_reason_cleanup():
    p = dc.label_probs({"Keep": -0.1, "Explore": -2.5, "Skip": -3.0})
    assert abs(sum(p.values()) - 1) < 1e-3 and p["Keep"] > 0.8
    assert dc.label_probs({"Keep": 20.0, "Explore": 17.6, "Skip": 17.1}) == p  # scores or logprobs: same softmax
    assert dc.decider_label({"Keep": -3, "Explore": -1, "Skip": -2}) == "Explore"
    # the decider, not the argmax: Keep the largest score yet s under T_HI is Explore; Skip the largest yet s at
    # or above T_LO is Explore too
    assert dc.decider_label({"Keep": 1.0, "Explore": 0.9, "Skip": 0.8}) == "Explore"
    assert dc.decider_label({"Keep": 0.0, "Explore": 1.0, "Skip": 1.2}) == "Explore"
    assert dc.decider_label({"Keep": 2.0, "Explore": 2.0, "Skip": 2.0}) == "Explore"  # s = 0.5
    assert dc.clean_reason("Keep — meets the requirements; location works.") == "meets the requirements; location works"
    assert dc.clean_reason("Skip: a; a; b; c; d; e") == "a; b; c; d"
    assert dc.clean_reason("Explore — a; b; and the cut of", complete=False) == "a; b"
    assert dc.clean_reason("Keep") == ""


class FakeScorer:
    name = "fake"
    calls: list = []

    def health(self):
        return {"backend": "fake", "model": "fake/model"}

    def score(self, prefix, items, reasons, max_reason_tokens):
        FakeScorer.calls.append((prefix, items, reasons))
        res = []
        for k, it in enumerate(items):
            lab = {"Keep": -0.2, "Explore": -2.0, "Skip": -3.0} if k % 2 == 0 else {"Keep": -3.0, "Explore": -0.3, "Skip": -2.0}
            word = dc.decider_label(lab)
            res.append({"id": it["id"], "scores": lab, "p": dc.label_probs(lab), "label": word,
                        "reason": "fake reason" if reasons else None, "prompt_tokens": 1, "cached_tokens": 0, "seconds": 0.0})
        return {"backend": "fake", "model": "fake/model", "results": res, "stats": {"items": len(items)}}


@pytest.fixture()
def stack(tmp_path):
    """A fake scorer behind dc.serve, and the demo server in front of it, on free ports."""
    FakeScorer.calls = []
    servers = []

    def free_port():
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    sp = free_port()
    threading.Thread(target=dc.serve, args=(FakeScorer(), "127.0.0.1", sp), daemon=True).start()
    pool = tmp_path / "pool.jsonl"
    pool.write_text("\n".join(json.dumps(r) for r in [
        posting("a"), posting("b"), posting("c"), posting("d"), posting("cit", citizens_only=True)]))
    personas = tmp_path / "personas"
    personas.mkdir()
    (personas / "alex.json").write_text(json.dumps({"name": "alex", "resume": "r" * 100, "profile": FORM}))
    results = tmp_path / "results"
    results.mkdir()
    demo = srv.Demo(f"http://127.0.0.1:{sp}", pool, results, personas)
    dp = free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", dp), srv.make_handler(demo, ROOT / "demo" / "index.html"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    import time
    for _ in range(50):  # wait for the scorer port
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{sp}/health", timeout=1)
            break
        except OSError:
            time.sleep(0.05)
    yield f"http://127.0.0.1:{dp}", demo, results
    httpd.shutdown()


def post(url, body, headers=None, raw=False):
    data = body if raw else json.dumps(body).encode()
    req = urllib.request.Request(url, data, headers or {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        return {"_code": e.code, **json.load(e)}


RESUME = "Alex Morgan\nalex@example.com\nB.S. Computer Science, Python, SQL, Java. Intern at Harbor Analytics, summer 2026."


def test_page_and_health(stack):
    base, _, _ = stack
    page = urllib.request.urlopen(base + "/").read().decode()
    assert "Excluded" in page and "For you" in page and "Explore" in page and 'id="resume"' in page
    h = json.load(urllib.request.urlopen(base + "/api/health"))
    assert h["pool"] == 5 and h["scorer"]["backend"] == "fake"
    assert json.load(urllib.request.urlopen(base + "/api/personas"))["personas"] == ["alex"]
    assert 'property="og:image" content="https://rolfyildirim--gradus.modal.run/og.jpg"' in page
    card = urllib.request.urlopen(base + "/og.jpg")
    assert card.headers["Content-Type"] == "image/jpeg" and card.read(2) == b"\xff\xd8"


def test_score_returns_three_buckets_and_calls_scorer_once(stack):
    base, _, _ = stack
    r = post(base + "/score", {"resume": RESUME, "profile": FORM, "live": 20})
    assert set(r["buckets"]) == {"For you", "Explore", "Excluded"}
    assert [c["id"] for c in r["buckets"]["Excluded"]] == ["cit"]  # H'' rule on citizens_only (F-1), before the model
    assert r["buckets"]["Excluded"][0]["source"] == "rules" and r["buckets"]["Excluded"][0]["rule"]["kind"] == "skip"
    assert len(r["buckets"]["For you"]) == 2 and len(r["buckets"]["Explore"]) == 2
    assert r["stats"]["rule_skipped"] == 1 and r["stats"]["scored_live"] == 4 and r["stats"]["not_scored"] == 0
    assert len(FakeScorer.calls) == 1
    prefix, items, reasons = FakeScorer.calls[0]
    assert "alex@example.com" not in prefix and "Alex Morgan" not in prefix  # redacted
    assert "CANDIDATE PROFILE:\n" in prefix and "Work authorization:" in prefix and reasons is True
    assert prefix + items[0]["suffix"] == prompt_b(lib.redact(RESUME), lib.profile_text(FORM), posting("a")["text"])


def test_live_cap_and_errors(stack):
    base, _, _ = stack
    r = post(base + "/score", {"resume": RESUME, "profile": FORM, "live": 2})
    assert r["stats"]["scored_live"] == 2 and r["stats"]["not_scored"] == 2
    assert post(base + "/score", {"resume": "short", "profile": FORM})["_code"] == 400
    assert post(base + "/score", {"resume": RESUME, "profile": {**FORM, "work_auth": "nope"}})["_code"] == 400


def test_precomputed_results_are_used_only_for_the_same_prompt(stack):
    base, demo, results = stack
    p = demo.prepare({"resume": RESUME, "profile": FORM})
    hit = {"label": "Keep", "p": {"Keep": 0.97, "Explore": 0.02, "Skip": 0.01},
           "scores": {"Keep": 25.0, "Explore": 21.0, "Skip": 20.5}, "reason": "from file",
           "text_sha": lib.text_sha(posting("a")["text"])}
    (results / "x.json").write_text(json.dumps({"prefix_sha": lib.prefix_sha(p["prefix"]), "results": {"a": hit, "b": hit}}))
    r = post(base + "/score", {"resume": RESUME, "profile": FORM, "precomputed": "x", "live": 20})
    assert r["stats"]["precomputed"] == 2 and r["stats"]["scored_live"] == 2
    assert {c["source"] for c in r["buckets"]["For you"]} >= {"precomputed"}
    other = post(base + "/score", {"resume": RESUME + " more", "profile": FORM, "precomputed": "x", "live": 20})
    assert other["stats"]["precomputed"] == 0 and other["stats"]["scored_live"] == 4
    assert "different resume or profile" in other["stats"]["warning"]
    assert r["stats"]["warning"] == ""
    demo.scorer_url = "http://127.0.0.1:9"  # scorer down: the error says why the file did not cover it
    down = post(base + "/score", {"resume": RESUME + " more", "profile": FORM, "precomputed": "x", "live": 20})
    assert down["_code"] == 502 and "different resume or profile" in down["error"] and "cannot reach" in down["error"]
    assert post(base + "/score", {"resume": RESUME, "profile": FORM, "precomputed": "missing"})["_code"] == 400


def test_extract_endpoint_and_profile_endpoint(stack):
    base, _, _ = stack
    r = post(base + "/api/extract", _docx(["Hello docx resume"]), {"X-Filename": "cv.docx"}, raw=True)
    assert r["text"] == "Hello docx resume"
    assert post(base + "/api/extract", b"x", {"X-Filename": "cv.zip"}, raw=True)["_code"] == 400
    # the page sends encodeURIComponent(file.name): a Turkish file name must work
    name = urllib.parse.quote("Şebnem Çelik CV.docx")
    assert post(base + "/api/extract", _docx(["Merhaba"]), {"X-Filename": name}, raw=True)["text"] == "Merhaba"
    bad = post(base + "/api/extract", b"plain text, not a zip", {"X-Filename": "bad.docx"}, raw=True)
    assert bad["_code"] == 400 and "could not read" in bad["error"]
    assert post(base + "/api/profile", FORM)["profile"] == lib.profile_text(FORM)


def test_work_auth_flip_adds_citizens_only_and_keeps_the_others(stack):
    base, demo, _ = stack
    demo.pool = [posting(f"n{i}") for i in range(6)] + [posting(f"c{i}", citizens_only=True) for i in range(3)]
    demo.by_id = {r["id"]: r for r in demo.pool}
    f1 = post(base + "/score", {"resume": RESUME, "profile": FORM, "live": 4})
    cit = post(base + "/score", {"resume": RESUME, "profile": {**FORM, "work_auth": "citizen_or_pr"}, "live": 4})
    model_ids = lambda r: {c["id"] for b in r["buckets"].values() for c in b if c["source"] == "model"}  # noqa: E731
    assert f1["stats"]["rule_skipped"] == 3 and model_ids(f1) == {"n0", "n1", "n2", "n3"}
    assert model_ids(cit) == {"n0", "n1", "n2", "n3", "c0", "c1", "c2"}
    assert cit["stats"]["citizens_only_extra"] == 3 and cit["stats"]["scored_live"] == 7
    assert [c["id"] for c in cit["not_scored"]] == ["n4", "n5"] and cit["stats"]["not_scored"] == 2
    assert "Alex Morgan" not in cit["resume"] and "Harbor Analytics" in cit["resume"]


def test_salary_cap_applies_to_live_and_precomputed_keeps(stack):
    base, demo, results = stack
    demo.pool = [posting("lo", salary_max=20000, min_years=0), posting("ok")]  # FORM's floor is 25,000
    demo.by_id = {r["id"]: r for r in demo.pool}
    r = post(base + "/score", {"resume": RESUME, "profile": FORM, "live": 20})  # the fake scorer: Keep, then Explore
    assert r["buckets"]["For you"] == [] and r["stats"]["rule_capped"] == 1 and r["stats"]["rule_skipped"] == 0
    lo = next(c for c in r["buckets"]["Explore"] if c["id"] == "lo")
    assert lo["source"] == "model" and lo["rule"]["model_label"] == "Keep" and "under your $25,000 floor" in lo["reason"]
    p = demo.prepare({"resume": RESUME, "profile": FORM})
    hit = {"label": "Keep", "p": {"Keep": 0.97, "Explore": 0.02, "Skip": 0.01},
           "scores": {"Keep": 25.0, "Explore": 21.0, "Skip": 20.5}, "reason": "from file"}
    (results / "cap.json").write_text(json.dumps({"prefix_sha": lib.prefix_sha(p["prefix"]), "results": {
        i: {**hit, "text_sha": lib.text_sha(demo.by_id[i]["text"])} for i in ("lo", "ok")}}))
    r = post(base + "/score", {"resume": RESUME, "profile": FORM, "precomputed": "cap", "live": 20})
    assert [c["id"] for c in r["buckets"]["For you"]] == ["ok"] and r["stats"]["precomputed"] == 2
    assert [(c["id"], c["source"], c["rule"]["kind"]) for c in r["buckets"]["Explore"]] == [("lo", "precomputed", "salary_cap")]


def test_precompute_gives_reasons_per_label_and_drops_stale_text(stack, tmp_path):
    base, demo, results = stack
    persona = tmp_path / "p.json"
    persona.write_text(json.dumps({"name": "alex", "resume": RESUME, "profile": FORM}))
    out = results / "alex.json"
    demo.precompute(persona, out, reasons_per_label=1)
    data = json.loads(out.read_text())
    assert set(data["results"]) == {"a", "b", "c", "d"} and data["model"] == "fake/model"
    with_reason = {i: r["label"] for i, r in data["results"].items() if r["reason"]}
    assert sorted(with_reason.values()) == ["Explore", "Keep"]  # one per label the fake scorer gives
    assert all(r["text_sha"] == lib.text_sha(posting(i)["text"]) for i, r in data["results"].items())
    demo.by_id["a"] = {**demo.by_id["a"], "text": "rewritten posting " * 30}  # a rebuilt pool reuses an id
    r = post(base + "/score", {"resume": RESUME, "profile": FORM, "precomputed": "alex", "live": 20})
    assert r["stats"]["precomputed"] == 3 and r["stats"]["scored_live"] == 1
    assert r["stats"]["precomputed_model"] == "fake/model" and r["stats"]["warning"] == ""
    files = json.load(urllib.request.urlopen(base + "/api/results"))["files"]
    assert files[0]["model"] == "fake/model" and "3 of 4" in files[0]["warning"]
    cit = tmp_path / "cit.json"
    demo.precompute(persona, cit, reasons_per_label=1, work_auth="citizen_or_pr")
    assert json.loads(cit.read_text())["work_auth"] == "citizen_or_pr" and "cit" in json.loads(cit.read_text())["results"]


def test_scorer_down_is_a_clear_502(stack):
    base, demo, _ = stack
    demo.scorer_url = "http://127.0.0.1:9"  # nothing listens
    r = post(base + "/score", {"resume": RESUME, "profile": FORM})
    assert r["_code"] == 502 and "cannot reach the scorer" in r["error"]
    assert json.load(urllib.request.urlopen(base + "/api/health"))["scorer"]["ok"] is False


def test_scorer_api_rejects_bad_requests():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    threading.Thread(target=dc.serve, args=(FakeScorer(), "127.0.0.1", port), daemon=True).start()
    import time
    time.sleep(0.3)
    assert post(f"http://127.0.0.1:{port}/score", {"items": []})["_code"] == 400


# ---- the server never pulls in model code

def test_server_import_graph_has_no_model_libraries():
    code = ("import sys; sys.path.insert(0, 'demo'); sys.path.insert(0, 'src'); import server; "
            "bad = [m for m in ('torch', 'mlx', 'mlx_lm', 'transformers', 'safetensors', 'numpy') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def _mini_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)


def test_pdf_text_with_pypdf():
    pytest.importorskip("pypdf")
    assert "Hello pdf resume" in lib.extract_text("cv.pdf", _mini_pdf("Hello pdf resume"))
