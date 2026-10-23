# Cycle-3 diagnostic: the author's row read (test 3)

Written 2026-10-21 under eval/protocol_amendment_4.json (406135c). Test 3 is a
practice set; nothing here changes its recorded result.

Sample: seed 20261021, one row per posting. 14 rows where raw h says Skip and gold
says Explore (pool 25), 6 rows where gold says Keep and raw h does not (pool 17).
Raw h, not h + H'. Pool sizes show raw h calls only 25 gold-Explore rows Skip; most
of the 55 Explore-read-as-Skip errors in h + H' come from the H' layer.

## Verdicts

| # | row | posting | gold | h | my read |
|---|---|---|---|---|---|
| 1 | f3_2c05148d71 | Recursion talent network | Explore | Skip | gold fine (no real opening) |
| 2 | f3_0f1aa9d2c4 | Watershed SWE | Explore | Skip | gold fine |
| 3 | f3_f8ba3471cb | Amylyx VP Clinical Compliance | Explore | Skip | gold fine |
| 4 | f3_2422913fbb | Chai Research Scientist | Explore | Skip | debatable: wet-lab BA for frontier ML research, contract-only; Skip defensible |
| 5 | f3_5e2574fbcd | Fresenius supply chain | Explore | Skip | gold fine |
| 6 | f3_8f809cb1a7 | Xaira Sr SWE | Explore | Skip | gold fine |
| 7 | f3_ae3f9595fa | Adams Clinical Sr Mgr | Explore | Skip | gold fine |
| 8 | f3_f6f7527175 | Watershed Perf Marketing | Explore | Skip | debatable: "Required Qualifications: 3+ years ... Salesforce, AdWords, Webflow" not shown; rubric text says Skip |
| 9 | f3_7963168fdf | Adaptive Sr Mgr SWE | Explore | Skip | gold fine |
| 10 | f3_7cadeab32e | Sartorius PhD intern | Explore (adj.) | Skip | debatable: hands-on TFF asked for, not shown; A said Skip |
| 11 | f3_1c392c2d67 | Sartorius Bangalore | Explore (adj.) | Skip | debatable: mech-eng degree vs "related scientific field", no GMP/CAPA; A said Skip |
| 12 | f3_ed91e5b89b | eGenesis QS Specialist | Explore | Skip | gold fine |
| 13 | f3_3494fe48cc | Jade IR teaser | Explore | Skip | gold fine (teaser) |
| 14 | f3_cd7bf319d6 | Simons reporting intern | Explore | Skip | debatable: three journalism clips required, none shown |
| 15 | f3_228211536b | Orbital FDE, London | Keep | Explore | gold doubtful: US citizen with no UK right to work stated, wants hybrid/remote only |
| 16 | f3_73dd23ac40 | Watershed SWE | Keep | Skip | gold fine, but inconsistent with row 2 (same posting, same missing Pandas/Matplotlib, Explore there) |
| 17 | f3_8eb3f48b1f | Adaptive Staff Cloud | Keep | Skip | gold fine |
| 18 | f3_5570719cb6 | Fresenius supply chain | Keep | Skip | gold fine |
| 19 | f3_7cbc6cf55d | Twist marketing intern | Keep | Explore | gold plausible |
| 20 | f3_fe7002720f | Adams Clinical Sr Mgr | Keep | Skip | gold fine |

## What it says

1. h is too strict, for real. On 13-14 of 20 rows the gold is the better call. h
   turns "a listed qualification isn't named on the resume" into Skip, which is the
   rules' behaviour it was trained on.
2. The gold is softer than the rubric's own words. On rows 8, 10, 11, 14 the
   posting lists a required qualification the resume lacks; the rubric text says
   Skip, the labelers said Explore. Both labelers are LLMs and share that lean, so
   their 89% agreement partly measures a shared habit. About 6 of 20 gold labels
   are debatable.
3. Gold is inconsistent on twin rows (2 vs 16): the same missing skill counts as a
   blocker in one row and not in the other, depending on the rest of the profile.
4. Non-US postings: the rubric only covers US work authorization, so a US citizen
   "relocating to London" reads as a clean Keep (row 15). That is a rubric gap.
5. Test-data artifact: many synthetic resumes carry the posting's exact title as the
   candidate's current job (rows 3, 6, 7, 13, 15, 17, 20) and list the posting's
   keywords as skills. Labelers lean on that ("already holds the same title"). Real
   resumes rarely look like this, so these tests overstate how easy Keep is.

## Consequences for the teacher decision

- A rubric-prompted LLM teacher will match this gold far better than the rules,
  partly because it shares the labelers' lean. That is fine for scoring against
  this gold, but it is the same family judging itself. A blind human-read sheet (100 rows,
  test 1) is the one outside check on that, and test 4's report should give scores
  on the A/B-agree subset as well as the full gold.
- Before relabeling 10k rows, settle the one boundary the rubric leaves open: a
  listed qualification that is not marked "required", or is marked required but is
  a soft skill or application material. Write the decision into the labeling prompt
  as a clarification of the rubric, the same for training labels and test-4 gold.
- Add a line for non-US postings (needs the right to work in that country).
- The resume generator for cycle-3 training and test 4 should stop copying the
  posting's title and keyword list into the resume.
