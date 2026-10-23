import json
import re
from pathlib import Path

import pytest

from matcher import textclean as tc

ROOT = Path(__file__).resolve().parents[1]

MD = "**Title:** Intern\n\n## About\n*   Run assays\n*   Write [docs](https://x.io/a)\n---\n__bold__ and *it*"


def test_strip_markdown_leaves_plain_text():
    out = tc.strip_markdown(MD)
    assert not tc.has_markdown(out)
    assert "- Run assays" in out and "Write docs" in out and "bold and it" in out
    assert "---" not in out and "##" not in out


def test_strip_markdown_keeps_real_text():
    t = "Python, C++ and 3*4 matrix; snake_case_name stays. 2 * 3"
    assert tc.strip_markdown(t) == t


def test_redact_contact_block_becomes_candidate():
    r = ("Maria Gonzalez\n123 Main Street, Apt 4B, Boston, MA 02115\nmaria.g@gmail.com | (617) 555-0134 | linkedin.com/in/mariag\n\n"
         "SUMMARY\nChemist. Call 617-555-0134 or write maria.g@gmail.com.\nEDUCATION\nB.S. Chemistry, 2025")
    out = tc.redact(r)
    assert out.startswith("Candidate\nBoston, MA")
    assert not tc.has_contact(out) and "Maria" not in out and "Main Street" not in out
    assert "B.S. Chemistry, 2025" in out


def test_redact_keeps_resume_without_contact_block():
    r = "Candidate\nHouston, TX\n\nPROFESSIONAL SUMMARY\nDoctoral-level psychologist."
    assert tc.redact(r) == r
    assert tc.redact("SUMMARY\nA\nEDUCATION\nB").startswith("Candidate\nSUMMARY")


def test_redact_labelled_contact_lines():
    out = tc.redact("Priya Nair, MSc\nEmail: priya@uni.ac.in\nPhone: +1 (857) 555 0101\nAddress: 45 Oak Ave, Somerville, MA 02143\nWebsite: https://priya.dev\n\nEXPERIENCE\nLab tech")
    assert not tc.has_contact(out) and "Priya" not in out and out.startswith("Candidate")
    assert "EXPERIENCE\nLab tech" in out


def test_redact_keeps_body_content_that_looks_like_a_street():
    body = ("EXPERIENCE\nIT Assistant\n- Migrated 3 Google Drive accounts to the new domain\n"
            "- Supervised 8 Lab Technicians at 5 Hospital Place\n- Opened 2 Oak Street clinics for the county")
    out = tc.redact("Maria Gonzalez\nBoston, MA\n\n" + body)
    assert "Migrated 3 Google Drive accounts" in out and "at 5 Hospital Place" in out and "2 Oak Street clinics" in out
    # a real address in the body still goes
    assert "Main Street" not in tc.redact("Candidate\n\nEXPERIENCE\nOffice at 123 Main Street, Boston, MA 02115")


def test_redact_keeps_headline_city_and_summary_lines():
    out = tc.redact("Maria Gonzalez\nPharmacy Technician\nNew York\nmaria@x.com | (617) 555-0134\n\nSKILLS\nPharmacy")
    assert out.startswith("Candidate\nPharmacy Technician\nNew York") and "Maria" not in out and "@" not in out
    out = tc.redact("Wei Chen\nData Analyst\n\nEXPERIENCE\nAnalyst")
    assert out.startswith("Candidate\nData Analyst")
    summary = "Lab scientist – 10 years of assay work – seeks a research role"
    assert summary in tc.redact("Wei Chen\n" + summary + "\n\nEDUCATION\nB.S.")
    # a "Resume" title line: the name is the next line
    assert "Wei" not in tc.redact("Resume\nWei Chen\nAustin, TX\n\nSKILLS\nSQL")


def test_redact_drops_zip_without_comma_and_bare_domains():
    out = tc.redact("Priya Nair\nSomerville MA 02143\n\nPUBLICATIONS\nDeposited at ncbi.nlm.nih.gov/geo/acc and ASP.NET apps")
    assert "02143" not in out and "Somerville MA" in out
    assert "ncbi.nlm.nih.gov" not in out and "ASP.NET" in out
    assert not tc.has_contact(out)


WA = "We are unable to sponsor employment visas for this role."
POSTING = ("Title: Intern\nCompany: Acme\nLocations: NYC\n\n" + "\n\n".join(f"Perk {i}: we love our team and our snacks." for i in range(300))
           + "\n\nQualifications\n- Pursuing a bachelor's degree\n- 2 years of Python\n\n" + WA
           + "\n\nAcme is an equal opportunity employer. We provide reasonable accommodation to applicants with disabilities.")


def test_fit_posting_keeps_header_requirements_and_work_auth():
    out = tc.fit_posting(POSTING, 1500)
    assert len(out) <= 1500
    assert out.startswith("Title: Intern\nCompany: Acme\nLocations: NYC")
    assert WA in out and "Pursuing a bachelor" in out and "2 years of Python" in out
    assert "equal opportunity" not in out  # boilerplate goes first
    assert tc.cut_posting(POSTING, 1500).find(WA) < 0


def test_fit_posting_short_text_untouched_and_hard_cut_marked():
    assert tc.fit_posting("Title: A\n\nshort", 500) == "Title: A\n\nshort"
    assert tc.fit_posting("word " * 5000, 400).endswith("[truncated]")


FILLER = "\n\n".join(f"Our team builds great products and our culture values curiosity, paragraph {i} of many words here." for i in range(40))


def test_fit_posting_keeps_short_label_value_pay_mode_and_enrollment_lines():
    post = ("Title: Analyst Intern\nCompany: Acme\nLocations: NYC\n\nAbout Us\n" + FILLER +
            "\n\nBase Pay Range: $24.00 - $32.00 Per Hour\n\nWorkplace Category: Hybrid\n\nRelevant Experience (1 Year)\n\n"
            "Remote:\nNo\n\nNeeds to be officially enrolled in an undergraduate bachelor program in Finance.\n\n"
            "Required Qualifications\n\nEducation:\nHigh School Diploma or GED (Required)\n\nPreferred Qualifications\n\n"
            "Skills:\n- Python\n")
    out = tc.fit_posting(post, 1800)
    assert len(out) <= 1800
    for line in ("$24.00 - $32.00 Per Hour", "Workplace Category: Hybrid", "Relevant Experience (1 Year)", "Remote:\nNo",
                 "officially enrolled", "High School Diploma or GED (Required)"):
        assert line in out, line
    # a heading whose section holds only sub-headings is kept, so preferred skills stay preferred
    assert "Required Qualifications" in out and "Preferred Qualifications" in out
    assert out.index("Preferred Qualifications") < out.index("- Python")


def test_short_value_lines_are_not_headings():
    for s in ("Base Pay Range: $24.00 - $32.00 Per Hour", "Workplace Category: Hybrid", "Relevant Experience (1 Year)",
              "Summary Pay Range: $44,000 - $89,000", "Compensation: $40 per hour"):
        assert not tc._is_heading(s), s
    assert tc._is_heading("Minimum Qualifications") and tc._is_heading("Compensation:")


def test_strip_markdown_drops_zero_width_characters():
    assert tc.strip_markdown("Required\u200b Qualifications\ufeff") == "Required Qualifications"


def _real(name: str) -> list[dict]:
    p = ROOT / name
    if not p.exists():
        pytest.skip(f"no {name} in this checkout")
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_fit_posting_keeps_the_card_facts_of_the_real_postings():
    # independent of fit_posting's word lists: the facts the extraction found in each posting
    rows = _real("data/raw/ats/postings.jsonl")
    test = tc.card_fact_survival(rows, _real("data/built/audit/test_cards.jsonl"), 6000)
    assert test["must_group_kept"] == 1.0 and test["postings_losing_a_must_group"] == 0, test
    assert test["mode_sentences_kept"] == 1.0 and test["pay_kept"] in (1.0, None), test
    g = tc.card_fact_survival(rows, _real("data/built/cards/extracted_real_cards.jsonl"), 6000)
    assert g["pay_kept"] == 1.0 and g["license_kept"] == 1.0 and g["min_years_kept"] == 1.0, g
    assert g["must_group_kept"] >= 0.965 and g["mode_sentences_kept"] >= 0.998, g
    assert tc.survival(rows, 6000)["all_survive"] == 1.0


def _long(body: str, filler_lines: int = 60) -> str:
    intro = "\n".join(f"Our team ships product number {i} to customers in many markets across the globe." for i in range(filler_lines))
    return "Title: Software Engineer Intern\nCompany: Acme\nLocations: NYC\n\nAbout the Role\n" + intro + "\n\n" + body


def test_fit_posting_keeps_requirement_lines_under_an_unknown_heading():
    # d3c0c779: "You have ..." under "We're excited about you because" lost to a long "About the Role" section
    post = _long("We're excited about you because\n- You have a solid understanding of algorithms and data structures\n"
                 "- You enjoy working with people")
    out = tc.fit_posting(post, 2500)
    assert len(out) <= 2500 and "understanding of algorithms and data structures" in out


def test_fit_posting_sheds_soft_requirements_before_strong_ones():
    # bb47dc93: the shed loop dropped the last unit first, so the closing "(required)" lines went before the
    # earlier nice-to-have ones
    soft = "\n\n".join(f"Preferred {i}:\n\n- Nice to have exposure to framework {i} in a team" for i in range(40))
    post = ("Title: Analyst Intern\nCompany: Acme\nLocations: NYC\n\n" + soft +
            "\n\nQualifications:\n\n- Proficient with Excel (required)\n\n- Proficient with SQL and Python (required)")
    for budget in range(1200, 2000, 100):
        out = tc.fit_posting(post, budget)
        assert len(out) <= budget
        assert "Proficient with Excel (required)" in out and "Proficient with SQL and Python (required)" in out, budget


def test_fit_posting_keeps_office_days_sentences():
    # e7067c38: an office-days sentence without hybrid / remote / on-site was 'other' text and was shed
    reqs = "\n".join(f"- Skilled with tool number {i} for daily analysis tasks" for i in range(40))
    post = ("Title: Analyst\nCompany: Acme\nLocations: NYC\n\nQualifications\n" + reqs + "\n\nHow we work\n"
            "You will work at least 4 days in the office each week, with the flexibility to work from home 1 day a week.")
    assert "4 days in the office" in tc.fit_posting(post, 1500)


def test_is_work_auth():
    assert tc.is_work_auth("Applicants must be authorized to work in the U.S. without sponsorship.")
    assert tc.is_work_auth("This role requires U.S. citizenship and an active security clearance.")
    assert not tc.is_work_auth("We hire without regard to race, religion or citizenship status.")
    assert not tc.is_work_auth("Pay with Visa card at the cafeteria.")


def test_split_synthetic_and_compose():
    title, company, body = tc.split_synthetic("**Data Intern**\nNorthwind Labs\n\nWe build things.\n*   Python")
    assert (title, company) == ("Data Intern", "Northwind Labs") and "- Python" in body
    t, c, b = tc.split_synthetic("Data Intern\nWe are a fast growing company that builds many things for many people.\n\nBody")
    assert c == ""
    # a company suffix may end in '.' (54 training postings: 'Willowmere Inc.', 'Foxhollow Harvest Co.'); a short sentence
    # on line 2 is still no company name
    for name in ("Willowmere Inc.", "Foxhollow Harvest Co.", "Orrin Building Co."):
        assert tc.split_synthetic(f"Nurse\n{name}\n\nWe care.")[1:] == (name, "We care.")
    for line in ("We build things.", "Apply today!", "Join us?"):
        assert tc.split_synthetic(f"Nurse\n{line}\n\nBody")[1] == "", line
    out = tc.compose_posting(title, company, "NYC", body)
    assert out.startswith("Title: Data Intern\nCompany: Northwind Labs\nLocations: NYC\n\n")


def test_ats_style_locations_in_some_rows():
    outs = {tc.format_locations(["New York, NY"], "onsite", f"k{i}") for i in range(60)}
    assert "NYC" in outs and "New York, NY" in outs
    assert tc.format_locations([], "remote", "k") == "Remote in USA"


PARAS = [
    "Acme is an equal opportunity employer and considers all qualified applicants without regard to race, color, religion or sex.",
    "We offer comprehensive benefits including medical, dental and vision coverage, a 401(k) plan and paid time off for all staff.",
    "Applicants must be authorized to work in the United States and we do not offer visa sponsorship for this opening at this time.",
    "Candidates must have a bachelor's degree and 3 years of experience; the minimum qualifications are listed above for you.",
    "If you need a reasonable accommodation during the application process, please let us know and we will gladly help you out.",
    "Compensation range for this role is $20 to $30 per hour, based on 2026 location, and a reasonable accommodation is available.",
]


def test_harvest_keeps_only_eeo_accommodation_and_privacy_paragraphs():
    rows = [{"company": "Zed", "description": "Intro text here.\n\n" + "\n\n".join(PARAS)}]
    bank = tc.harvest_boilerplate(rows)
    assert PARAS[0] not in bank  # names a brand (Acme)
    assert PARAS[4] in bank
    assert PARAS[1] not in bank and PARAS[2] not in bank and PARAS[3] not in bank and PARAS[5] not in bank


# What the reviewer found in the old bank: each of these must be refused.
BAD_BANK = [
    "We are an equal opportunity employer. This role is based out of our Seattle or Boston office and follows a hybrid schedule.",
    "Employees work in-person 5 days a week out of our office in Midtown Manhattan; we are an equal opportunity employer.",
    "Even if you don't meet every single requirement, we encourage you to apply; we are an equal opportunity employer for all.",
    "This position is overtime eligible and we are an equal opportunity employer who considers all applicants without regard to race.",
    "Interns are not eligible for employee benefits. We are an equal opportunity employer and welcome individuals with disabilities.",
    "AppLovin is an equal opportunity employer and considers all qualified applicants without regard to race, color or religion.",
    "If you need a reasonable accommodation, call 555-123-4567 or visit https://example.com/accommodation for help with the process.",
    "Reasonable accommodation is available; a background check and drug screening are required of all equal opportunity hires.",
    "Your salary will be determined by academic level; we are an equal opportunity employer and welcome individuals with disabilities.",
    # pointers to a page or link the synthetic posting does not have
    "Please refer to the privacy notice at the bottom of this page for submitting any data access, deletion, or other data subject rights requests.",
    "The Employee and Applicant Privacy Statement describes how we collect and safeguard applicant information. Please see the privacy statement on our website here.",
]
# An independent list of things no bank paragraph may contain.
BANNED_IN_BANK = re.compile(
    r"https?://|www\.|@|\d|\$|\b(?:salary|pay|bonus|stipend|overtime|intern\w*|co-?op|full[- ]time|part[- ]time|hybrid|remote|"
    r"in[- ]person|on-?site|office|background check|drug|references?|sponsor\w*|visa|citizen\w*|e-?verify|require\w*|"
    r"degree|years?|students?|meet every|travel|relocat\w*|this page|website|click|link)\b", re.I)


def test_bank_rule_refuses_the_reviewed_bad_paragraphs():
    for p in BAD_BANK:
        assert not tc.bank_paragraph_ok(p), p
    assert tc.bank_paragraph_ok(PARAS[4])


def test_every_paragraph_of_the_real_bank_passes_the_banned_list():
    raw = ROOT / "data/raw/ats/postings.jsonl"
    if not raw.exists():
        pytest.skip("no real postings in this checkout")
    rows = [json.loads(x) for x in raw.read_text().splitlines() if x.strip()]
    bank = tc.harvest_boilerplate(rows)
    assert len(bank) >= 15
    for p in bank:
        assert not BANNED_IN_BANK.search(p), p
        # no capitalised brand or place: every capitalised word is in the generic list
        assert tc._bank_caps_ok(p), p


def test_load_bank_rebuilds_a_stale_or_old_format_file(tmp_path):
    post = tmp_path / "postings.jsonl"
    post.write_text(json.dumps({"company": "Zed", "description": PARAS[4]}) + "\n")
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(["Stale paragraph with a hybrid schedule out of our Boston office, equal opportunity."]))
    assert tc.load_bank(bank_path, post) == [PARAS[4]]
    data = json.loads(bank_path.read_text())
    assert data["fingerprint"] == tc.bank_fingerprint(post) and data["paragraphs"] == [PARAS[4]]
    assert tc.load_bank(bank_path, post, exclude_companies={"zed"}) == []  # exclusion set changed: rebuilt


def test_boilerplate_augmentation_is_deterministic_and_60_percent():
    bank = ([f"We are an equal opportunity employer, statement number {i} for everyone." for i in range(5)]
            + [f"If you need a reasonable accommodation, paragraph {i}, please let us know." for i in range(5)]
            + [f"Read our applicant privacy notice, paragraph {i}, to see how we handle data." for i in range(5)])
    counts = [tc.boilerplate_count(f"rq_{i}") for i in range(2000)]
    assert 0.55 < sum(c > 0 for c in counts) / 2000 < 0.65 and max(counts) == 3
    key = next(f"rq_{i}" for i in range(100) if tc.boilerplate_count(f"rq_{i}") == 3)
    a = tc.compose_posting("T", "C", "SF", "Body.", boiler=bank, key=key)
    assert a == tc.compose_posting("T", "C", "SF", "Body.", boiler=bank, key=key)
    assert a.count("equal opportunity") == 1 and a.count("accommodation") == 1 and a.count("privacy") == 1
    # the body already has an EEO line: no second one
    b = tc.compose_posting("T", "C", "SF", "Body. We are an equal opportunity employer.", boiler=bank, key=key)
    assert b.count("equal opportunity") == 1


def test_training_posting_header_from_card_and_rendered_text():
    card = {"title": "Lab Intern", "locations": ["New York, NY"], "work_mode": "onsite"}
    out = tc.training_posting(card, "Lab Intern\nNorthwind Labs\n\nWe test things.\n*   Pipette", "rq_1")
    assert out.startswith("Title: Lab Intern\nCompany: Northwind Labs\nLocations: ") and "- Pipette" in out and "*" not in out
    assert out == tc.training_posting(card, "Lab Intern\nNorthwind Labs\n\nWe test things.\n*   Pipette", "rq_1")


def test_street_regex_does_not_eat_education_lines():
    assert not tc.STREET.search("May 2026\n\nBachelor of Arts\nSt. Cloud State University")
    assert tc.STREET.search("373 Hill Court, St. Louis, MO")


def test_training_bank_is_fresh_and_clean():
    if not (ROOT / "data/raw/ats/postings.jsonl").exists() or not (ROOT / "data/built/audit/test_cards.jsonl").exists():
        pytest.skip("no real postings in this checkout")
    bank = tc.training_bank(ROOT)
    assert bank and all(tc.bank_paragraph_ok(p) for p in bank)
    card = {"title": "Lab Intern", "locations": ["Boston, MA"], "work_mode": "onsite"}
    out = tc.training_posting(card, "Lab Intern\nVarrow Labs\n\nWe run assays.", "rq_00003", bank)
    assert out.startswith("Title: Lab Intern\nCompany: Varrow Labs\nLocations: ")


def test_redact_one_line_headers_drop_the_name():
    cases = {
        "Jane Doe | Boston, MA | 617.555.0182 | jane.doe@umass.edu\n\nEDUCATION\nUMass Boston":
            "Candidate\nBoston, MA\n\nEDUCATION\nUMass Boston",
        "Jane Doe – Boston, MA – jane@x.com": "Candidate\nBoston, MA",
        "Jane Doe (617) 555-0182 jane@x.com\nBoston, MA": "Candidate\nBoston, MA",
        "Jane Doe, RN | jane@x.com | Boston, MA": "Candidate, RN\nBoston, MA",
        "Jane Doe | Boston, MA": "Candidate\nBoston, MA",
        "JANE DOE | New York, NY | jane@x.com": "Candidate\nNew York, NY",
        "jane doe\nBoston, MA": "Candidate\nBoston, MA",
        "Boston, MA | jane@x.com": "Candidate\nBoston, MA",
    }
    for raw, want in cases.items():
        assert tc.redact(raw) == want, raw


def test_redact_keeps_a_headline_when_there_is_no_name_line():
    assert tc.redact("Data Analyst\nNew York\nSUMMARY\nAnalyst") == "Candidate\nData Analyst\nNew York\nSUMMARY\nAnalyst"
    assert tc.redact("Registered Nurse, BSN\nBoston, MA\n\nSKILLS\nTriage").startswith("Candidate\nRegistered Nurse, BSN\nBoston")
    # a name line still goes; a headline after the comma stays
    for raw in ("Juan de la Cruz\nBoston, MA", "Maria Gonzalez\nBoston, MA"):
        assert tc.redact(raw) == "Candidate\nBoston, MA", raw
    assert tc.redact("Jane Doe, Data Analyst\nBoston, MA") == "Candidate\nData Analyst\nBoston, MA"


def test_has_contact_needs_a_place_after_a_header_street():
    assert not tc.has_contact("Candidate\nBoston, MA\nSUMMARY\nRan the clinic at 5 Hospital Place.")
    assert tc.has_contact("Candidate\n22 Lake Boulevard, Springfield\n\nSKILLS\nx")
    assert tc.has_contact("Candidate\n45 Oak Ave, Somerville, MA 02143")


# Real work-authorization sentences that the old splitter cut at 'U.S.' or 'Inc.' (SpaceX 9bf4727a, Northrop
# 83a9740c, Boeing 6aef6415, Varda 892a9689): each must reach the fitted text whole.
REAL_WA = (
    "- To conform to U.S. Government export regulations, applicant must be a (i) U.S. citizen or national, (ii) U.S. lawful, "
    "permanent resident (aka green card holder), (iii) Refugee under 8 U.S.C. § 1157, or (iv) Asylee under 8 U.S.C. § 1158, or "
    "be eligible to obtain the required authorizations from the U.S. Department of State.",
    "U.S. Citizenship is required for all positions with a government clearance and certain other restricted positions.",
    "To meet U.S. export control compliance requirements, a “U.S. Person” as defined by 22 C.F.R. §120.62 is required. "
    "“U.S. Person” includes U.S. Citizen, U.S. National, lawful permanent resident, refugee, or asylee.",
    "Varda Space Industries, Inc. participates in the U.S. Department of Homeland Security E-Verify program.",
)


def test_split_sentences_does_not_split_at_abbreviations():
    assert tc.split_sentences(REAL_WA[0]) == [REAL_WA[0]]
    assert tc.split_sentences(REAL_WA[3] + " We build spacecraft.") == [REAL_WA[3], "We build spacecraft."]
    assert tc.split_sentences("Ask Dr. Smith, e.g. on Monday. Work with John F. Kennedy. Done.") == [
        "Ask Dr. Smith, e.g. on Monday.", "Work with John F. Kennedy.", "Done."]
    assert len(tc.split_sentences(REAL_WA[2])) == 2


@pytest.mark.parametrize("wa", REAL_WA)
def test_fit_posting_keeps_real_work_auth_sentences_whole(wa):
    # the sentence sits in a long line (fit_posting splits only lines over 260 chars) between filler sentences
    line = ("Our team builds rockets and our culture values curiosity in everything we do every day. " * 3 + wa +
            " We also host many events for the team throughout the year in our offices.")
    post = _long(line, filler_lines=80)
    out = tc.fit_posting(post, 2000)
    assert len(out) <= 2000
    assert " ".join(wa.split()) in " ".join(out.split())


def test_survival_gate_is_not_fit_postings_own_splitter(monkeypatch):
    # with the old splitter fit_posting loses the head of the SpaceX sentence; the gate must see it
    row = {"title": "Engineer", "company": "Acme", "description": _long(
        "Our team builds rockets and our culture values curiosity in everything we do every day. " * 3 + REAL_WA[0], 80)}
    assert tc.survival([row], 2000)["all_survive"] == 1.0
    old = lambda line: [p for p in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", line.strip()) if p]
    monkeypatch.setattr(tc, "split_sentences", old)
    assert tc.survival([row], 2000)["all_survive"] == 0.0


def test_is_work_auth_recall_and_acronyms():
    for s in ("Must have authorization to work in the United States.", "Candidates must have employment authorization.",
              "Applicants must hold current work eligibility in the U.S.", "Must be a US national.",
              "We cannot hire international students.", "This role requires US work status.", "We support OPT students.",
              "We accept TN visa holders.", "Applications generated from outside the United States will not be considered.",
              "This role supports a UK Eyes Only programme, so nationality restrictions apply.",
              "Candidates must reside in the United States."):
        assert tc.is_work_auth(s), s
    for s in ("You can opt out at any time.", "Our ear protection is provided.", "TN is the state where we are based.",
              "We support U.S. national security missions.", "You may travel outside the United States twice a year."):
        assert not tc.is_work_auth(s), s


def test_redact_keeps_headlines_that_are_not_names():
    for head in ("Biology Major", "Aspiring Biochemist", "Bilingual (English/Spanish) Customer Service Representative"):
        assert tc.redact(f"{head}\nTampa, FL\n\nSKILLS\nPCR").startswith(f"Candidate\n{head}\nTampa, FL"), head


def test_redact_keeps_credentials_after_the_name():
    assert tc.redact("Maria Gonzalez, RN, BSN\nTampa, FL\n\nSKILLS\nx").startswith("Candidate, RN, BSN\nTampa, FL")
    assert tc.redact("John Smith, Pharm.D.\nTampa, FL\n\nSKILLS\nx").startswith("Candidate, Pharm.D.\nTampa, FL")
    for raw in ("Dr. Wei Chen\nAustin, TX", "Mehmet Öztürk\nAustin, TX", "Candidate | Austin, TX"):
        assert tc.redact(raw) == "Candidate\nAustin, TX", raw


def test_redact_keeps_skill_and_project_names_that_end_like_a_domain():
    r = "Candidate\nTampa, FL\n\nSKILLS\nExcel, Python, Socket.io, Next.js\nPROJECTS\n- Built ledger.io, a budgeting app"
    assert tc.redact(r) == r and not tc.has_contact(r)
    assert tc.has_contact("Candidate\njanedoe.io | Boston, MA\n\nSKILLS\nx")
    assert "janedoe" not in tc.redact("Jane Doe\njanedoe.io | Boston, MA\n\nSKILLS\nx")


def test_redact_international_phone_and_body_street_comma():
    out = tc.redact("Ayse Yilmaz | Istanbul, Turkey | +90 532 123 45 67 | ayse@x.com\n\nSKILLS\nx")
    assert out == "Candidate\nIstanbul, Turkey\n\nSKILLS\nx"
    assert tc.has_contact("Call me at +90 532 123 45 67") and tc.has_contact("Phone: +44 20 7946 0958")
    out = tc.redact("Candidate\nBoston, MA\n\nEXPERIENCE\n- Worked at 4 Park Place, Boston, MA on the front desk")
    assert "- Worked at Boston, MA on the front desk" in out


def test_header_shows_every_location_and_keeps_washington_dc():
    locs = [f"City{i}, TX" for i in range(8)] + ["Washington, DC"]
    for i in range(40):
        out = tc.format_locations(locs, "onsite", f"k{i}")
        assert all(x in out for x in locs[:8]) and "Washington, DC" in out, out


def test_bank_drops_a_paragraph_that_points_at_a_poster():
    p = ("We are an equal opportunity employer and consider all applicants without regard to race, color or religion. "
         "Please view the Know Your Rights poster for more about your rights under the law.")
    assert not tc.bank_paragraph_ok(p) and tc.bank_paragraph_ok(p.split(" Please")[0])


def test_redact_pronouns_repeated_city_and_url_punctuation():
    assert tc.redact("Jane Doe (she/her)\nBoston, MA\n\nSummary\nAnalyst.").split("\n")[:2] == ["Candidate", "Boston, MA"]
    # a fake street line leaves its city and the writer's own location line repeats it: one city line stays
    assert tc.redact("Maria Gonzalez\n12 Oak Avenue, Boston, MA 02115\nBoston, MA\n\nSummary\nNurse.").split("\n")[:3] == \
        ["Candidate", "Boston, MA", ""]
    body = tc.redact("Candidate\n\nExperience\n- Shared code on github.com/jane/repo; analysis in R.\n- See https://x.org/a/b, then more.")
    assert "- Shared code on; analysis in R." in body and "- See, then more." in body and not tc.has_contact(body)


def test_redact_does_not_read_a_city_or_a_school_as_a_name():
    # a real resume whose name line the demo already removed can open with its city or school
    assert tc.redact("New York, NY\nSkills") == "Candidate\nNew York, NY\nSkills"
    assert tc.redact("Los Angeles, California\nSkills") == "Candidate\nLos Angeles, California\nSkills"
    assert tc.redact("Harvard University | Cambridge, MA\nSkills") == "Candidate\nHarvard University | Cambridge, MA\nSkills"
    assert tc.redact("Lehigh University\nB.S. Chemistry") == "Candidate\nLehigh University\nB.S. Chemistry"
    # a state code that is also a credential stays a credential unless the place table knows the city
    assert tc.redact("John Smith, MD\nBoston, MA").startswith("Candidate, MD\nBoston, MA")
    assert tc.redact("Baltimore, MD\nSkills") == "Candidate\nBaltimore, MD\nSkills"
