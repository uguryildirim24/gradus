"""The real-test postings that carry a requirement the RequirementCard cannot hold, under e589370's rule.

Commit e589370 (clean_real_checked.py dev-v2) took out of the checked real dev set every posting with a restriction
the card cannot hold: one school only (school_only), class standing (class_standing), a graduation window
(grad_window), a degree level the card gets wrong (degree_level), must be local or onsite during the school year
(must_be_local); and, only when a non-Skip row's candidate is past the limit, a degree ceiling (degree_ceiling) or an
experience cap (exp_cap). On such a posting the builder's gold label is wrong by construction. The real test
(data/built/real_test/test.jsonl) was built the same way and never screened. This file applies that same rule to its
86 postings, as a fixed, pre-defined filter for a disclosed secondary number (every system scored with and without the
flagged postings); nothing is re-selected on it.

How the table was made: every one of the 86 postings was read in full (build_real_test.posting_text, the fuller text
where the prompt's 6,000-character fit cuts it) next to the card the builder used (data/built/audit/test_cards.jsonl),
on 2026-10-20, before any model prediction, eval output, results or diagnosis file under data/models/ was opened.
The decision is blind to which rows any model gets wrong. Same kinds, same conditional rule and same evidence
standard as clean_real_checked.RESTRICTIONS: each entry quotes words that appear verbatim in the posting (and all of
them also appear in the posting as the prompt shows it). Where a posting has several restrictions, the entry names
one that removes it outright (REMOVE_KINDS) before a conditional one. Source is "read" for every entry (named by this
read; the test postings had no checker flags of these kinds to draw on).

No conditional entry removes its posting: every degree_ceiling posting has an enrolled card, and the builder makes a
student of a higher degree a Keep (variants.degree_enrolled_higher) only on a card with no enrollment, so no non-Skip
row there is past the ceiling; the entries stay in the table so the rule is applied as written.

Applying it: test_decisions() runs clean_real_checked.dev_v2_decisions on the test rows with this table, the builder's
candidate cards (data/built/real_test/rows.jsonl) deciding the conditional kinds, so the rule is the committed code,
not a copy.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_real_test as brt  # noqa: E402
import clean_real_checked as ccr  # noqa: E402

TEST_FILE = ROOT / "data/built/real_test/test.jsonl"
BUILDER_ROWS = ROOT / "data/built/real_test/rows.jsonl"
REMOVE_KINDS = ccr.REMOVE_KINDS
CHECK_KINDS = ccr.CHECK_KINDS

TEST_RESTRICTIONS: dict[str, tuple[str, str, str, object]] = {  # posting id: (kind, source, posting quote, limit)
    # Apple Bank | IT Data Intern - Analytics, & Automation
    'ats_1e295cb0-851c-4444-95da-daa70198fa7d': ('class_standing', "read", 'currently enrolled in a related Bachelor’s degree program as a rising Senior, with an expected graduation date December 2027 - May 2028', None),
    # RTX | Software Engineer Intern
    'ats_27282ffa-070f-4079-bf3e-b159fa097600': ('class_standing', "read", 'Must have a minimum of 60 hours of credit by the start of the internship', None),
    # RTX | Electrical Production Support Engineer Intern - Summer 2027
    'ats_3170bd08-be46-4758-93fa-a6d1157147c3': ('class_standing', "read", 'will be at a minimum, entering their Sophomore by the start of summer 2027', None),
    # ConocoPhillips | Trading Analytics Intern
    'ats_5123007b-1b8e-4d00-babb-8c1c4ec245cc': ('class_standing', "read", 'Current level in college: Junior, Senior, or Graduate Student', None),
    # Allegion | Hardware Engineer Intern - Product Lifecycle Engineering
    'ats_792ba5ee-a91f-4674-a32b-415aec46bf35': ('class_standing', "read", 'Sophomore year must be complete by the start of the summer internship program for undergraduate candidates', None),
    # Altar'd State | IT Analytics Intern
    'ats_7b0bc5f5-b7eb-4ee8-905a-aff3cc7c2c4f': ('class_standing', "read", "Currently enrolled in a Bachelor's degree as a Junior or Senior student, or recent graduate", None),
    # CACI | Software Engineer Intern
    'ats_a82e8f66-23ec-4c7e-a091-3b34f4635e0f': ('class_standing', "read", 'Must be a Rising Junior/Senior in a bachelor’s program or a graduate student', None),
    # Analytical Mechanics Associates | Oceanographic Lidar Remote Sensing and Field Campaign Intern
    'ats_bae50bf6-94fe-4d18-bcb5-84cbab255649': ('class_standing', "read", 'Currently at the junior or senior undergraduate level.', None),
    # RTX | Digital Design Electrical Engineer Intern - Summer 2027
    'ats_c782a0fa-e20c-43ce-a32c-51fd89919b77': ('class_standing', "read", 'Must have a minimum of 60 hours of credit by the start of the internship / co-op session', None),
    # Voloridge | Quantitative Research Intern
    'ats_d429d89b-a684-47e8-a3c8-32639ab0d9c8': ('class_standing', "read", 'Completion of at least three years of an undergraduate or graduate degree program', None),
    # Capital One | Applied Research Intern
    'ats_d6cc7924-5545-46c3-8847-89f485a17231': ('class_standing', "read", '1st year of PhD coursework must be completed by June 2027', None),
    # General Motors | Machine Learning for Automation Researcher Intern - AI and Machine Learning for Automation Researcher
    'ats_0231a390-9b59-4ef8-a1f2-f4be35a26151': ('grad_window', "read", 'Must be graduating between August 2026 - June 2027', None),
    # General Dynamics Information Technology | Generative AI Software Development Intern
    'ats_03a8913b-a007-4657-8f59-bc29a722beda': ('grad_window', "read", 'Applicants should have a minimum of one semester remaining in their studies after their internship concludes', None),
    # Klaviyo | AI Engineer Intern
    'ats_0a38034b-5b0a-49b9-98c7-c844defb5ae4': ('grad_window', "read", 'Graduating in December 2027 or May/June 2028', None),
    # Verkada | Frontend Software Engineer Intern
    'ats_20898b37-1e46-4759-b318-1a51ae4d1200': ('grad_window', "read", "Actively pursuing a Bachelor's or Master's degree in Computer Science or similar technical field of study and graduating by June 2028", None),
    # Micron Technology | ASIC Digital Design Intern
    'ats_2e6978bb-ece5-4ba5-9ed6-b6b11bed2627': ('grad_window', "read", 'Must be enrolled in a degree program and not graduate before August 2027', None),
    # Merck | Associate Scientist Postdoctoral Fellow - AI/ML and Agentic AI for Drug Discovery
    'ats_61dda24d-33f5-4c50-9d90-8d7de6ea82a6': ('grad_window', "read", 'Receive a Ph.D. no later than spring 2027', None),
    # DoorDash | Software Engineer Intern
    'ats_66fcf084-7f7c-43ed-8947-83179031963c': ('grad_window', "read", 'are graduating between Fall 2027 to Summer 2028', None),
    # DriveTime | Data Science Intern - Summer 2027
    'ats_721548ed-6f3b-4fc1-ae29-76461872f99a': ('grad_window', "read", 'Bachelors or master’s student with a graduation date between December 2027 – May 2028', None),
    # Marsh | Finance Innovation & Technology Intern - Master's Finance Innovation & Technology
    'ats_73d1f84a-69e1-4693-b9a8-03e6ae82b0b1': ('grad_window', "read", 'with an expected graduation date on or before Spring 2028', None),
    # Anduril | Firmware Engineer Intern
    'ats_7bb707f9-e94b-4f2d-8921-ee6d523213bc': ('grad_window', "read", 'Must be returning to school at the end of the internship to continue academic studies for at least one quarter or semester', None),
    # Hadrian | Data Science/Data Engineer Intern
    'ats_8028ea71-392c-4173-9d2a-1dc31c5a4571': ('grad_window', "read", 'with an expected graduation date between December 2027 and June 2028', None),
    # Rocket Lab USA | Electrical Engineering Intern
    'ats_86615113-8398-487e-bc33-95253076af51': ('grad_window', "read", 'have at least one semester of school remaining post internship', None),
    # CoverMyMeds | CMM Technology Intern
    'ats_90d4be1a-888d-4b8c-a5e7-f07c01f135bb': ('grad_window', "read", 'scheduled to graduate in the Spring of 2028 or sooner', None),
    # The Federal Reserve System | Data Science Intern
    'ats_aaeb44bf-c9ec-46e3-989c-e8594be3ac17': ('grad_window', "read", 'Must be a current, matriculated student in an accredited program graduating December 2027 or later', None),
    # Booz Allen | Systems Engineer Intern - Summer Games
    'ats_c0d0a7a8-afbc-4283-886f-e4d7e36f52f0': ('grad_window', "read", 'Scheduled to obtain a Bachelor’s degree in a Technology, Engineering, or Mathematics field by Summer 2028 or Summer 2029', None),
    # Belvedere Trading | Quantitative Trading Intern - Summer 2027
    'ats_c2163298-bd36-4efb-8554-64b9997a0b1b': ('grad_window', "read", 'Graduation date of December 2027/Spring 2028', None),
    # Booz Allen | Software Developer Intern - Summer Games
    'ats_c7d3b494-6183-4201-ae92-c7ddbae1ae27': ('grad_window', "read", 'Scheduled to obtain a Bachelor’s degree in a Technology, Engineering, or Mathematics field by Summer 2028 or Summer 2029', None),
    # Chicago Trading Company | Quantitative Trading Intern
    'ats_ca124ec8-21c1-40d5-8aba-dea45cf0fe0d': ('grad_window', "read", 'You will graduate between December 2027-June 2028', None),
    # DoorDash | Software Engineer Intern - Summer 2027
    'ats_d3c0c779-480d-4810-9393-88d215aa5e7e': ('grad_window', "read", 'are graduating between Fall 2027 to Summer 2028', None),
    # DriveTime | Software Engineer Intern - Summer 2027
    'ats_db1e4fb4-a7e7-4c57-ae1c-7ffdab053208': ('grad_window', "read", "You are currently pursuing a bachelor's or master’s with a graduation date of December 2027 – May 2028", None),
    # Cboe | Software Engineer Intern - C++
    'ats_dd3566b3-ae9b-41bd-8cdf-29ccbfe6d891': ('grad_window', "read", 'should not be scheduled to graduate before December of the year in which the internship takes place', None),
    # Motorola | Software Engineer Intern - Summer 2027
    'ats_dffa3f5f-d496-43e5-90e0-105b940da03c': ('grad_window', "read", "Candidate's Graduation date must occur on or after December 2027.", None),
    # The Nuclear Company | Data Science Intern
    'ats_e53bf9a1-b4a2-4708-ab00-bcc4ac9a8901': ('grad_window', "read", 'Available for a full six-month term and returning to school afterward', None),
    # BlackRock | Quantitative Master’s Intern - Investments - Quantitative Investing
    'ats_e7067c38-ffd5-4e91-b625-6732557cb064': ('grad_window', "read", 'graduating between September 2027 and July 2028', None),
    # Ankura Consulting Group | Forensic Data & Analytics Intern
    'ats_f6a196e0-04eb-49bd-b649-5db7a14421e5': ('grad_window', "read", 'Graduating between December 2027 and June 2028', None),
    # Belvedere Trading | Software Engineer Intern - Summer 2027
    'ats_fc66b12e-8d7f-4a0e-b4f6-8f54189941d9': ('grad_window', "read", 'Graduation date of December 2027/Spring 2028', None),
    # Northrop Grumman | Software Engineer Intern
    'ats_fe413853-4493-4bf0-a4af-5723d1b9fdbc': ('grad_window', "read", 'will be enrolled full time in Fall 2027', None),
    # Travelers | Business Insights & Analytics Intern - BI&A LDP
    'ats_fbc330bb-a6c5-48a8-b0d9-537f8bf68269': ('degree_level', "read", 'Pursuing or obtained a Bachelor’s or Master’s degree.', None),
    # Pennsylvania State University | Guidance and Control Research & Development Intern
    'ats_69430fc1-dbac-4fba-8c07-697012305db3': ('must_be_local', "read", 'The successful candidate will work up to 20 hours/week during the fall and spring semesters and 40 hours/week over the summer', None),
    # Life Fitness | Android Software Engineer Intern
    'ats_e37213e4-b577-4fe8-8489-091b1ea2837b': ('must_be_local', "read", 'Must be local to the Chicagoland area and available to participate in the hybrid internship program', None),
    # Baird | IT Data & Analytics Intern
    'ats_e55a7700-5c93-43ed-a30e-6398ef65a328': ('must_be_local', "read", 'We are seeking a local student who can start in June 2027', None),
    # Zoox | Digital Engineer Student Worker
    'ats_bbd050e5-7cc1-4a9e-8592-b7b17d47105b': (
        'must_be_local', "read", 'contract student worker program ... Able to commit a minimum of 20 hours per week ... '
                                 'Able to work on-site at one of our office locations', None),
    # Astranis | Electrical Engineer Intern
    'ats_25f578d9-1330-4ae5-9608-a7740ea38a81': ('degree_ceiling', "read", 'Currently pursuing a B.S. in electrical or computer engineering', 'bachelor'),
    # Meridian Partners | Embedded Software Engineer Co-op
    'ats_2b427c48-848f-4f28-b666-9136b44ba757': ('degree_ceiling', "read", 'Working towards a BS or MS degree in Software Engineering, Computer Science, Electrical Engineering, or related field', 'master'),
    # Republic Airways | Strategic Operations Intern
    'ats_46afb6b3-6be4-457a-ab81-e5a109555aff': ('degree_ceiling', "read", 'Pursuing a bachelor’s degree (B.A. / B.S.)', 'bachelor'),
    # K2 Space | Electrical Engineer Intern - Summer 2027
    'ats_6fe74a2d-23cb-4b80-977c-92f082c5fed2': ('degree_ceiling', "read", 'Must be currently enrolled in a Bachelor’s or Master’s degree in electrical or computer engineering', 'master'),
    # ABB | Product Management Intern
    'ats_a2e915b7-cd1b-4137-9c87-a2707b3de604': ('degree_ceiling', "read", "Currently enrolled in a bachelor's degree program in the United States", 'bachelor'),
    # Buildertrend | Software Engineer Intern
    'ats_c2c6db10-5836-4c07-a30a-b82ff757db89': ('degree_ceiling', "read", 'Currently pursuing a Bachelor’s degree in computer science, information systems, or a related field', 'bachelor'),
    # Qumulo | Software Development Engineer Intern
    'ats_c2cbd149-9096-42ca-b785-b8ab7d96e3cf': ('degree_ceiling', "read", 'Pursuing a BS or BA in Computer Science or Computer Engineering or equivalent experience', 'bachelor'),
    # Philips | Service Sales Intern - IB Focus
    'ats_c54c66c0-44fb-477d-906d-1544410c81ce': ('degree_ceiling', "read", 'We welcome individuals are currently pursuing an undergraduate (BS) degree', 'bachelor'),
    # Meridian Partners | Flight Software Engineer Co-op
    'ats_ce5ce805-2b7c-460d-a010-28c5e413b7f6': ('degree_ceiling', "read", 'Working towards BS or MS in Computer Engineering, Computer Science, Electrical Engineering, or related field', 'master'),
    # Hewlett Packard Enterprise | ASIC Engineer Intern
    'ats_d6a60616-c4a4-45d6-8fb3-68aead7bf769': ('degree_ceiling', "read", "Currently pursuing a Bachelor's degree in Electrical Engineering, Computer Engineering, or a related technical field", 'bachelor'),
    # DraftKings | Software Engineer Intern
    'ats_e102b2db-3bfc-42d6-b363-84cbbe6172e5': ('degree_ceiling', "read", 'Currently enrolled in a Bachelor’s degree program in Computer Science, Computer Engineering, or a related technical field', 'bachelor'),
    # Astranis | Flight Software Intern
    'ats_e6fe2e61-3900-434b-b4ba-4dd51b2556b8': ('degree_ceiling', "read", 'Currently pursuing a B.S. in Computer Science, Electrical or Computer Engineering, or equivalent technical degree', 'bachelor'),
    # Meridian Partners | Python Software Engineer Co-op
    'ats_eeda0ac4-569c-4d27-927d-56e668f95ffa': ('degree_ceiling', "read", 'Working towards a BS or BA degree in Computer Science, Engineering. or related subject', 'bachelor'),
    # Freeform | Electrical Engineer Intern
    'ats_f670187c-507f-4b2c-8c12-8bb6c3800465': ('degree_ceiling', "read", 'Currently enrolled in a bachelor’s degree in electrical engineering or computer engineering', 'bachelor'),
}
# Read and nothing found the card cannot hold (a GPA, program dates, hours, a "3+ months" floor or a preferred line are
# not among the kinds, as on the dev side).
TEST_READ_CLEAR = frozenset({
    'ats_0a09d266-9cc5-488d-a0f1-a5950423f368', 'ats_1332c5af-a188-4adc-b53a-332d33165496',
    'ats_168be0d0-eff5-4b46-acaf-353e65e445b0', 'ats_21ae5c99-91ac-4040-8486-fdad81449835',
    'ats_2a5121b2-8df8-49ee-aef9-5b192735c687', 'ats_2c773875-aa62-411f-a839-511d6f823d7d',
    'ats_31553e6b-5e69-46b9-8944-9d7097c8df02', 'ats_36e1c66a-1cae-4b36-b414-d74b38a44a7f',
    'ats_5a07cf07-d505-4525-bec1-6e0fa482b1d9', 'ats_67c1b95e-7d8e-429c-8252-51543316efbd',
    'ats_6d7ae4ff-faad-43e5-8a8c-e621e27b715a', 'ats_92b36766-febe-474d-941e-0bf16bc2ecc0',
    'ats_949d51a9-718b-4bfe-8a44-2735fc2a1a8a', 'ats_9bf4727a-804d-4ee8-8c44-0eeec2960987',
    'ats_a0f4b708-8312-4d5d-b203-5379abf73c59', 'ats_a1902097-6f8c-4bf1-8924-0b50d519c98e',
    'ats_a1bc5739-2d47-4f6c-8627-5b11e873e160', 'ats_aa53868c-7316-4c11-b2d1-387065247bb1',
    'ats_ac1d9d95-dc9f-44d9-ab63-b5fbd5785de5', 'ats_ac3275e6-e472-46bb-87d5-72c63714a25a',
    'ats_bbe59360-d6e5-4189-9126-da097b9c93c6', 'ats_bebbeff8-81af-4e7a-b6f6-7658307b38b6',
    'ats_bee2e809-5a33-41d9-8c9b-0dc0c44f7100', 'ats_cb8e36f2-8593-4793-bcf8-e10743512481',
    'ats_d05c32eb-6bfa-4b00-a9ec-162781cb5a2d', 'ats_d264e688-c953-468e-a492-487e9683cb63',
    'ats_dfc6d5d9-fa48-468c-999b-1116299c47c2', 'ats_e556f0eb-1ee9-4cd0-b6f2-502f5ba3d435',
    'ats_ef96a3f9-16e2-4616-994f-10af59a086aa',
})
# The judgment calls behind an entry or a clear posting, for the record.
TEST_NOTES: dict[str, str] = {
    'ats_1332c5af-a188-4adc-b53a-332d33165496': '"Expected graduation date of Fall 2025 or later." every currently-enrolled candidate meets; card enrolled_in master holds "Currently pursuing M.S. or Ph.D."',
    'ats_168be0d0-eff5-4b46-acaf-353e65e445b0': 'spring internship, relocation offered; not must_be_local',
    'ats_21ae5c99-91ac-4040-8486-fdad81449835': 'co-op term; no kind',
    'ats_2a5121b2-8df8-49ee-aef9-5b192735c687': "Bachelor's or advanced degree, no ceiling",
    'ats_2c773875-aa62-411f-a839-511d6f823d7d': '"Working towards a Bachelor\'s Degree" but next line "Enrolled in a full time matriculated and nationally accredited baccalaureate or graduate program" admits graduate students; GPA under stand-out',
    'ats_46afb6b3-6be4-457a-ab81-e5a109555aff': "'minimum of 20 hours for spring and fall interns' is per term, not must_be_local",
    'ats_5123007b-1b8e-4d00-babb-8c1c4ec245cc': 'also grad window Fall 2027 through Spring 2029',
    'ats_5a07cf07-d505-4525-bec1-6e0fa482b1d9': "Bachelor's, Master's, or Ph.D.",
    'ats_66fcf084-7f7c-43ed-8947-83179031963c': 'also exp cap 2, BS/MS',
    'ats_67c1b95e-7d8e-429c-8252-51543316efbd': '"live within commuting distance" only for the summer dates; card holds location',
    'ats_949d51a9-718b-4bfe-8a44-2735fc2a1a8a': "Master's or PhD; card enrolled_in master",
    'ats_9bf4727a-804d-4ee8-8c44-0eeec2960987': "bachelor's or graduate; spring term",
    'ats_a1bc5739-2d47-4f6c-8627-5b11e873e160': "10a Labs Engineering Fellow: 'up to 20 hours per week' is a flexible remote "
                                                 "schedule and the pay tiers by student level sit under Benefits; no "
                                                 "school-year or student-status line, so neither must_be_local nor class_standing",
    'ats_bae50bf6-94fe-4d18-bcb5-84cbab255649': 'also 40 weeks at 10hr/week, onsite',
    'ats_bbd050e5-7cc1-4a9e-8592-b7b17d47105b': "an enrolled B.S. or M.S. student at 20+ hours a week on site in Foster City is working "
                                                 "alongside school, as in dev's Zoox student worker ats_c57ba1d3 (must_be_local); "
                                                 "the posting never says 'school year', so this is a reading, not a quote",
    'ats_0a09d266-9cc5-488d-a0f1-a5950423f368': "'working toward, or recently acquired, a degree' names no level, so the card's "
                                                 "'none' holds it (Travelers names levels and is degree_level)",
    'ats_d05c32eb-6bfa-4b00-a9ec-162781cb5a2d': "'Currently pursuing or recently completed a degree' names no level (as Hy-Vee)",
    'ats_aa53868c-7316-4c11-b2d1-387065247bb1': "'currently going through a formal computer science program, or have recently "
                                                 "completed one' names no level (as Hy-Vee)",
    'ats_cb8e36f2-8593-4793-bcf8-e10743512481': '"Completion of junior year preferred" is preferred; GPA not a kind',
    'ats_e37213e4-b577-4fe8-8489-091b1ea2837b': 'also grad 2028 or 2029',
    'ats_e556f0eb-1ee9-4cd0-b6f2-502f5ba3d435': 'PhD; card enrolled_in doctoral',
    'ats_e55a7700-5c93-43ed-a30e-6398ef65a328': 'part time 15-20 h during semesters; also grad May 2028 or later',
    'ats_ef96a3f9-16e2-4616-994f-10af59a086aa': 'PhD; card enrolled_in doctoral',
    'ats_f670187c-507f-4b2c-8c12-8bb6c3800465': 'nice-to-have advanced degree and Masters/PhD pay tiers',
    'ats_fbc330bb-a6c5-48a8-b0d9-537f8bf68269': "the must-have takes a bachelor's student or any graduate: neither degree_level nor enrolled_in alone holds it, as dev's 'Currently pursuing or recently completed a Master's, or Ph.D.'",
    'ats_fc66b12e-8d7f-4a0e-b4f6-8f54189941d9': 'also "college juniors"',
}


def quote_problems(raw: dict[str, dict], shown: bool = False) -> list[str]:
    """TEST_RESTRICTIONS entries whose quote is not in the posting text (whitespace folded; each part of a quote
    joined by " ... " must be there). shown: check the posting as the prompt shows it (build_real_test.shown_posting)."""
    norm = lambda t: re.sub(r"\s+", " ", t)  # noqa: E731
    out = []
    for i, (_, _, q, _) in TEST_RESTRICTIONS.items():
        if i not in raw:
            out.append(i)
            continue
        text = brt.posting_text(raw[i])
        text = norm(brt.shown_posting(text) if shown else text)
        if not all(norm(part) in text for part in q.split(" ... ")):
            out.append(i)
    return out


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def test_decisions(rows: list[dict] | None = None, cands: dict[str, dict] | None = None) -> dict[str, dict]:
    """posting id -> {kind, source, quote, limit, remove, past_limit_rows} for the real test, by
    clean_real_checked.dev_v2_decisions (a posting neither restricted nor read-clear stops it)."""
    rows = read_jsonl(TEST_FILE) if rows is None else rows
    if cands is None:
        cands = {r["pair_id"]: r["candidate"] for r in read_jsonl(BUILDER_ROWS)}
    return ccr.dev_v2_decisions({"test": rows}, {"test": cands}, TEST_RESTRICTIONS, TEST_READ_CLEAR)


test_decisions.__test__ = False  # a function of this module, not a pytest test
