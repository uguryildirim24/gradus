"""Run B's prompt (cycle 3, eval/protocol_amendment_5.json runs.B): the rubric at the top of the single user turn, then
h's layout (matcher.pairs.build_prompt) without h's instruction, with the whole posting (no token budget).

    SYSTEM_B + "\\n\\n" + "RESUME:\\n...\\n\\nCANDIDATE PROFILE:\\n...\\n\\nJOB POSTING:\\n<whole posting>\\n\\nAnswer:"

SYSTEM_B is the labelers' rubric as training saw it: scripts/build_fit_train3.system_b() reads it from
eval/protocol_amendment_5.json (labeler_prompt_test3 through the end of the three rubric bullets). It is written out here
so the demo carries it without eval/ (tests/test_demo_b_fresh.py checks it equals system_b() and its sha256 equals
eval/cycle3_eval_inputs.json system_b_sha256). Standard library and matcher.pairs only: no model code.
"""
from __future__ import annotations

from .pairs import INSTRUCTION, build_prompt

SYSTEM_B = (
    "You are labeling job-match rows for a test set. Each row has a job posting, a candidate's "
    "resume, and the candidate's profile (what they filled in on a form). Decide what a careful "
    'career advisor would tell this candidate about applying. Today is October 2026.\n'
    '\n'
    '- Skip: something the posting states as a hard requirement rules them out. That covers work '
    'authorization (US citizens or permanent residents only; no visa sponsorship when the candidate '
    'will need it; no OPT or CPT when they are on F-1), student status (a posting for current '
    'students when the candidate has graduated, or one for graduates when they are still a student; '
    'compare the graduation year with today), a required degree level, or a required degree field '
    'they lack unless the posting accepts "related" or "equivalent", a required license they don\'t '
    'hold, an employment type, location or work mode their profile rules out, stated pay below their '
    'salary floor, or a skill or amount of experience the posting clearly calls required (must have, '
    "required, minimum qualifications) that the resume doesn't show.\n"
    "- Keep: nothing rules them out and they meet the posting's stated requirements. Skills given as "
    'examples ("such as", "e.g.", "or similar", "one of", "experience with a language like") are met '
    'by any one of them or a close equivalent. Missing preferred, bonus or nice-to-have items do not '
    'block Keep unless most of them are missing. Duties describe the work; they are not requirements.'
    '\n'
    '- Explore: worth a look but not a clean fit. Nothing rules them out, but they are about a year '
    'short on required experience, they miss most of the preferred items, a requirement is genuinely '
    'ambiguous, or the posting says too little to judge (a short teaser with no real requirements).'
)
SYSTEM_B_SHA256 = "1d12b0d0ec70e3dcab5e7523a3477c04fc75714564b5e33b94c31f44e7cb2e0d"
WHOLE = 10 ** 12  # build_fit_train3.WHOLE: a posting budget no posting reaches, so fit_posting keeps it whole


def prompt_b(resume: str, profile: str, posting: str) -> str:
    """build_fit_train3.prompt_b(resume, profile, posting, SYSTEM_B): the rubric, a blank line, then build_prompt's text
    after INSTRUCTION with the whole posting."""
    p = build_prompt(resume, profile, posting, max_posting_tokens=WHOLE)
    head = INSTRUCTION + "\n"
    if not p.startswith(head + "RESUME:\n"):
        raise ValueError("build_prompt no longer opens with INSTRUCTION then RESUME:")
    return SYSTEM_B + "\n\n" + p[len(head):]
