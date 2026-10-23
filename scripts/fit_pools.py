"""Fresh-test, dev and train pools of venator's posting store for the v4 fit layer, and the frozen eval protocol.

  fit_pools.py snapshot   every stored venator Posting as plain text (needs venator's code: run it with
      venator's Python, $VENATOR_REPO/.venv/bin/python -B scripts/fit_pools.py snapshot)
      -> data/built/fit/venator_postings.jsonl
  fit_pools.py split      dedupe against everything this project has used, split by employer into test / dev / train
      -> data/built/fit/split.json, data/built/fit/pools/{test,dev,train}.jsonl, eval/fit_freeze.json
  fit_pools.py protocol   write eval/protocol.json (the decision rule, frozen before any fit data is built)

Builders read a pool with `load_pool("test")` (a list of posting dicts: key, source, board, employer, title,
locations, url, posted_at, workplace_type, opportunity_type, description_kind, text, chars, field, length).

Used anywhere in this project (a posting matching one by id, URL or near-copy text is dropped from every pool):
the 2,655 fetched ATS postings and the 46 failed fetches (data/raw/ats; the 86 real-test postings and every
real_train posting are among them, checked below), every posting id under data/built/real_test and
data/built/real_train, and the 701 postings in data/built/fit/venator_picks.json (postings labelled by hand earlier,
outside this project; a frozen one-time input that this repo does not rebuild).
Near-copy: matcher.nearcopy rules (same company and title, or description 6-word-window Jaccard >= 0.8), found
through an inverted index of rare windows so 97k postings run against the used set in seconds.

Employers are pools' unit: every posting of a test employer is in the test pool and nowhere else; dev employers are
distinct from both. An employer is a cluster of names (Employers: "Veeva" and "Veeva Systems", "Dyne Tx" and "Dyne
Therapeutics", "BMC" and "Boston Medical Center" are one), and a test or dev cluster holds no name from the used set.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from matcher.nearcopy import NEAR, company_key, norm_text, shingles  # noqa: E402

FIT = ROOT / "data/built/fit"
POOLS = FIT / "pools"
SNAP = FIT / "venator_postings.jsonl"
PICKS = FIT / "venator_picks.json"
SPLIT = FIT / "split.json"
FREEZE = ROOT / "eval/fit_freeze.json"
PROTOCOL = ROOT / "eval/protocol.json"
VENATOR = Path(os.environ.get("VENATOR_REPO", "venator"))  # venator's code checkout
INSTALL = Path(os.environ.get("VENATOR_HOME", "venator-data"))  # venator's data folder (its posting store)
PROFILE = os.environ.get("VENATOR_PROFILE", "default")

SEED = 20261020
TEST_TARGET, TEST_MIN_EMPLOYERS, TEST_CAP = 250, 25, 10   # postings, employers, postings per employer
DEV_TARGET, DEV_CAP = 300, 15
MAX_EMPLOYER = 800       # test and dev employers have at most this many postings after dedupe (CVS-sized boards stay in train)
SHORT_SHARE, EMPTY_SHARE = 0.16, 0.05   # test and dev: at least 16% teasers (1-699 chars), at most 5% empty bodies
SHORT = 700              # "teaser" below this many characters
LENGTHS = ((0, 1, "empty"), (1, 700, "teaser 1-699"), (700, 1300, "700-1300"), (1300, 3000, "1300-3000"), (3000, 10**9, "3000+"))
FIELDS = ("internship", "health_nursing", "lab_bio", "data", "software", "sales_ops", "other")


# ---------------------------------------------------------------- snapshot (venator's code)

def snapshot(_: argparse.Namespace) -> None:
    sys.path.insert(0, str(VENATOR / "src"))
    from venator.discover.store import load_postings
    from venator.profile.loader import load_profile
    from venator.qualify.posting import canonical_posting

    profile = load_profile(INSTALL / "profiles" / PROFILE)
    posts = load_postings(INSTALL / "data/postings")
    FIT.mkdir(parents=True, exist_ok=True)
    by_key = {}
    with open(SNAP, "w") as f:
        for p in posts:
            text = canonical_posting(p, {}, include_spans=False).description_text
            row = {"key": p["key"], "source": p.get("source", ""), "board": str(p.get("board", "")),
                   "employer": posting_company(p, profile) or "", "title": str(p.get("title") or ""),
                   "locations": posting_locations(p), "url": str(p.get("url") or ""),
                   "apply_url": str(p.get("apply_url") or ""), "posted_at": str(p.get("posted_at") or ""),
                   "discovered_at": str(p.get("discovered_at") or ""),
                   "workplace_type": str(p.get("workplace_type") or ""),
                   "opportunity_type": str(p.get("opportunity_type") or ""),
                   "description_kind": str(p.get("description_kind") or ""), "text": text}
            by_key[p["key"]] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(by_key)} postings -> {SNAP}")


def posting_company(p: dict, profile) -> str:
    """The company venator shows for a stored Posting (venator.score.selection.inputs_for_passes)."""
    return p.get("company") or profile.sources.names.get(str(p.get("board", "")), "")


def posting_locations(p: dict) -> str:
    locs = p.get("locations")
    names = [str(x.get("name")) for x in locs if isinstance(x, dict) and x.get("name")] if isinstance(locs, list) else []
    return "; ".join(dict.fromkeys(names)) or str(p.get("location") or "")


# ---------------------------------------------------------------- shared helpers

def jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm_url(u: str) -> str:
    u = (u or "").strip().casefold().split("#")[0].split("?")[0].rstrip("/")
    u = re.sub(r"^https?://(www\.)?", "", u)
    return re.sub(r"/(application|apply)$", "", u)


def url_ids(source: str, url: str) -> set[tuple[str, str, str]]:
    """(source, tenant, id) tokens of an ATS URL: greenhouse numeric ids, lever/ashby uuids, workday requisition."""
    u = norm_url(url)
    out = set()
    if source == "workday":
        tenant = u.split(".")[0]
        last = u.rsplit("/", 1)[-1]
        if "_" in last:
            out.add(("workday", tenant, last.rsplit("_", 1)[-1]))
    else:
        for m in re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|\d{6,}", u):
            out.add((source, "", m))
    return out


def venator_ids(p: dict) -> set[tuple[str, str, str]]:
    ext = p["key"].split(":", 2)[-1] if p["key"].count(":") >= 2 else ""
    if p["source"] == "workday":
        tenant = p["board"].split(".")[0].casefold()
        return {("workday", tenant, ext.casefold())} | url_ids("workday", p["url"])
    if p["source"] in ("greenhouse", "lever", "ashby"):
        return {(p["source"], "", ext.casefold())}
    return set()


_GENERIC = {"the", "of", "and", "a", "us", "usa", "america", "american", "national", "new", "united", "global", "general",
            "first", "medical", "university", "health", "healthcare", "group", "st", "saint", "center", "research",
            "services", "solutions", "systems", "technologies", "technology", "sciences", "science", "bio", "pharma",
            "pharmaceuticals", "therapeutics", "labs", "lab", "clinical", "international", "partners", "consulting"}
PLACEHOLDER = re.compile(r"\b(confidential|stealth|not listed|undisclosed|anonymous)\b")


def employer_tokens(name: str) -> set[str]:
    """Tokens two names of one employer share: the despaced company key ("scholarrock"), its first word when that is
    specific, 3+ letters (else its first two words: "veeva", "dyne", "thermofisher"), and the initials of a 3+ word name ("bmc")."""
    ck = company_key(name)
    w = ck.split()
    if not w:
        return set()
    out = {ck.replace(" ", ""), "".join(w[:2])}
    out.add(w[0] if w[0] not in _GENERIC and len(w[0]) >= 3 else "".join(w[:2]))
    if len(w) >= 3:
        out.add("".join(x[0] for x in w))
    return {t for t in out if len(t) >= 3}


class Employers:
    """Employer clusters: union-find over company keys linked by a shared employer token. Conservative on purpose:
    over-merging only keeps two employers in one pool; under-merging would let one employer sit in two pools."""

    def __init__(self, names):
        self.parent: dict[str, str] = {}
        owner: dict[str, str] = {}
        for n in sorted(set(names)):
            ck = company_key(n)
            if not ck:
                continue
            self.parent.setdefault(ck, ck)
            for t in employer_tokens(n):
                if t in owner:
                    self._union(owner[t], ck)
                else:
                    owner[t] = ck

    def _find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def key(self, name: str) -> str:
        ck = company_key(name)
        return self._find(ck) if ck in self.parent else ""


_FIELD_RX = [  # first match wins, on the normalized title
    ("internship", r"\b(intern|interns|internship|internships|co op|coop|summer associate|student|students|apprentice|"
                   r"apprenticeship|fellowship|werkstudent|trainee|graduate program)\b"),
    ("health_nursing", r"\b(nurse|nurses|nursing|rn|lpn|lvn|cna|np|aprn|pharmacist|pharmacy|physician|therapist|"
                       r"therapy|dental|dentist|dialysis|medical assistant|phlebotomist|phlebotomy|sonographer|"
                       r"radiology|radiologic|respiratory|surgical|surgeon|caregiver|patient care|home health|"
                       r"social worker|dietitian|dietician|paramedic|emt|optometrist|veterinary|psychiatric|"
                       r"psychologist|behavioral health|hospitalist|anesthesia|midwife|care coordinator|case manager)\b"),
    ("lab_bio", r"\b(scientist|scientists|research associate|laboratory|lab|chemist|chemistry|assay|bioprocess|"
                r"cell|molecular|microbiology|microbiologist|biologist|biology|process development|formulation|"
                r"toxicology|pharmacology|qc|immunology|genomics|sequencing|histotechnologist|technologist|"
                r"specimen|antibody|protein|purification|manufacturing associate|biochemist|in vivo|in vitro)\b"),
    ("data", r"\b(data|analytics|analyst|machine learning|ml|ai|statistician|statistics|statistical|biostatistician|"
             r"biostatistics|bioinformatics|bioinformatician|computational|business intelligence|bi|quantitative)\b"),
    ("software", r"\b(software|developer|devops|sre|cloud|cybersecurity|security engineer|it|helpdesk|help desk|"
                 r"systems administrator|network|programmer|full stack|fullstack|frontend|front end|backend|back end|"
                 r"web|platform engineer|infrastructure|forward deployed)\b"),
    ("sales_ops", r"\b(sales|account executive|account manager|business development|marketing|operations|"
                  r"customer|retail|store|representative|coordinator|logistics|supply chain|warehouse|procurement|"
                  r"buyer|recruiter|recruiting|talent acquisition|hr|human resources|finance|financial|accountant|"
                  r"accounting|administrative|admin|receptionist|assistant|driver|delivery|merchandiser|cashier|"
                  r"customer service|office)\b"),
]


def field_of(title: str) -> str:
    t = norm_text(title)
    for name, rx in _FIELD_RX:
        if re.search(rx, t):
            return name
    return "other"


def length_of(chars: int) -> str:
    return next(name for lo, hi, name in LENGTHS if lo <= chars < hi)


class RareIndex:
    """matcher.nearcopy.NearIndex semantics over many postings: same (company key, title), or description Jaccard
    >= NEAR. Candidates come from 6-word windows shared by few indexed postings (boilerplate windows are skipped
    for lookup but still count in the exact Jaccard)."""

    def __init__(self, posts: dict[str, dict], rare: int = 50):
        self.keys = {}
        for i in sorted(posts):
            self.keys.setdefault((company_key(posts[i].get("company", "")), norm_text(posts[i].get("title", ""))), i)
        self.sh = {i: shingles(posts[i].get("description", "")) for i in sorted(posts) if posts[i].get("description", "").strip()}
        inv = collections.defaultdict(list)
        for i, s in self.sh.items():
            for w in s:
                inv[w].append(i)
        self.inv = {w: v for w, v in inv.items() if len(v) <= rare}

    def near(self, post: dict, title_rule: bool = True) -> tuple[str, str] | None:
        if title_rule:
            k = (company_key(post.get("company", "")), norm_text(post.get("title", "")))
            if k in self.keys:
                return self.keys[k], "same company and title"
        if not post.get("description", "").strip():
            return None
        s = shingles(post["description"])
        cand = collections.Counter()
        for w in s:
            for i in self.inv.get(w, ()):
                cand[i] += 1
        best = None
        for i, _ in cand.most_common(20):
            t = self.sh[i]
            if min(len(s), len(t)) < NEAR * max(len(s), len(t)):
                continue
            j = len(s & t) / len(s | t)
            if j >= NEAR and (best is None or j > best[0]):
                best = (j, i)
        return (best[1], f"description {best[0]:.2f} alike") if best else None


def as_near(p: dict) -> dict:
    return {"company": p["employer"], "title": p["title"], "description": p["text"]}


# ---------------------------------------------------------------- split

def used_set() -> tuple[dict[str, dict], set[str], set[tuple], set[str], dict]:
    """(used postings for near-copy by id, used URLs, used id tokens, used employer keys, notes)."""
    ats = {"ats_" + d["simplify_id"]: d for d in jsonl(ROOT / "data/raw/ats/postings.jsonl")}
    fails = list(jsonl(ROOT / "data/raw/ats/failures.jsonl"))
    built_ids = set()
    for sub in ("real_test", "real_train"):
        for f in (ROOT / "data/built" / sub).rglob("*.jsonl"):
            for d in jsonl(f):
                for k in ("rq_id", "base_rq_id", "posting_id"):
                    if isinstance(d.get(k), str):
                        built_ids.add(d[k])
                if isinstance(d.get("id"), str) and d["id"].startswith("ats_"):
                    built_ids.add(d["id"])
    real_test = {d["base_rq_id"] for d in jsonl(ROOT / "data/built/real_test/test.jsonl")}
    outside = sorted(i for i in built_ids if i not in ats)
    picks = json.loads(PICKS.read_text())["picks"]
    snap = {p["key"]: p for p in jsonl(SNAP)}
    used = {i: {"company": d.get("company", ""), "title": d.get("title", ""), "description": d.get("description", "")}
            for i, d in ats.items()}
    urls = {norm_url(d.get("url", "")) for d in list(ats.values()) + fails} - {""}
    toks = set()
    for d in list(ats.values()) + fails:
        toks |= url_ids(d.get("ats", ""), d.get("url", ""))
    emp = {d.get("company", "") for d in ats.values()}
    for x in picks:
        p = snap.get(x["key"])
        emp.add(x.get("employer", ""))
        if p:
            used["pick:" + x["key"]] = as_near(p)
            urls |= {norm_url(p["url"]), norm_url(p["apply_url"])} - {""}
            emp.add(p["employer"])
    emp = {e.strip() for e in emp if e and e.strip()}
    notes = {"ats postings": len(ats), "ats failed fetches": len(fails),
             "real_test postings": len(real_test), "real_test outside ats": len(real_test - set(ats)),
             "real_test/real_train posting ids": len(built_ids), "of them outside the ats set": len(outside),
             "venator picks": len(picks), "picks in store": sum(x["key"] in snap for x in picks)}
    return used, urls, toks, emp, notes


def split(a: argparse.Namespace) -> None:
    used, used_urls, used_toks, used_emp, notes = used_set()
    pick_keys = {x["key"] for x in json.loads(PICKS.read_text())["picks"]}
    store = list(jsonl(SNAP))
    idx = RareIndex(used)
    drop = collections.Counter()
    dropped = []
    kept = []
    for p in store:
        why = None
        if p["key"] in pick_keys:
            why = "venator pick (same id)"
        elif {norm_url(p["url"]), norm_url(p["apply_url"])} & used_urls:
            why = "same URL as a used posting"
        elif venator_ids(p) & used_toks:
            why = "same ATS job id as a used posting"
        else:
            hit = idx.near(as_near(p))
            if hit:
                why = ("near copy of a venator pick: " if hit[0].startswith("pick:") else "near copy of an ATS posting: ") + (
                    hit[1] if hit[1].startswith("same") else "description >= 0.8 alike")
        if why:
            drop[why] += 1
            dropped.append({"key": p["key"], "why": why})
        else:
            kept.append(p)
    # reposts within the store: same text under another id (keep the first discovered)
    kept.sort(key=lambda p: (p["discovered_at"], p["key"]))
    seen, uniq = {}, []
    for p in kept:
        body = norm_text(p["text"])
        h = hashlib.sha256((body if len(body) >= 300 else
                            "\x00".join((company_key(p["employer"] or p["board"]), norm_text(p["title"]), body))
                            ).encode()).hexdigest()
        if h in seen:
            drop["repost within the store (same text, another id)"] += 1
            dropped.append({"key": p["key"], "why": "repost of " + seen[h]})
            continue
        seen[h] = p["key"]
        uniq.append(p)
    emps = Employers([p["employer"] for p in uniq] + sorted(used_emp))
    used_keys = {emps.key(e) for e in used_emp} - {""}
    for p in uniq:
        p["chars"] = len(p["text"])
        p["field"] = field_of(p["title"])
        p["length"] = length_of(p["chars"])
        p["emp_key"] = emps.key(p["employer"]) or ("board:" + p["board"].casefold())
    by_emp = collections.defaultdict(list)
    for p in uniq:
        by_emp[p["emp_key"]].append(p)

    rng = random.Random(SEED)
    # test employers: absent from the used set (both keys), with a named employer, at least one usable posting
    def usable(ps):
        return [p for p in ps if p["opportunity_type"] not in ("talent_pool",)]
    free = sorted(e for e, ps in by_emp.items() if not e.startswith("board:") and e not in used_keys
                  and not PLACEHOLDER.search(e) and len(ps) <= MAX_EMPLOYER and usable(ps))
    cands = list(free)
    rng.shuffle(cands)
    test_emp = choose_employers(cands, by_emp, usable, rng, TEST_TARGET, TEST_CAP, min_emp=TEST_MIN_EMPLOYERS,
                                need_sources=("workday", "greenhouse", "smartrecruiters"))
    # dev employers: also absent from the used set (model selection is not on familiar employers), not test
    rest = [e for e in free if e not in test_emp]
    rng.shuffle(rest)
    dev_emp = choose_employers(rest, by_emp, usable, rng, DEV_TARGET, DEV_CAP, min_emp=20,
                               need_sources=("workday", "greenhouse", "smartrecruiters"))
    pools = {"test": [], "dev": [], "train": []}
    for e, ps in by_emp.items():
        name = "test" if e in test_emp else "dev" if e in dev_emp else "train"
        pools[name].extend(ps)
    # The sealed pools hold the sampled, usable, capped postings; every other posting of a test or dev employer is
    # reserved (in no pool, never trained on).
    reserved = {"test": [], "dev": []}
    for name, emps, cap in (("test", test_emp, TEST_CAP), ("dev", dev_emp, DEV_CAP)):
        keep = []
        for e in sorted(emps):
            sample = emps[e]
            keep.extend(sample)
            chosen = {p["key"] for p in sample}
            reserved[name].extend(p["key"] for p in by_emp[e] if p["key"] not in chosen)
        pools[name] = keep
    # cross-pool near copies: a train posting near a test or dev posting (another employer's copy) is dropped
    for name in ("test", "dev"):
        sealed = RareIndex({p["key"]: as_near(p) for p in pools[name]})
        survivors = []
        for p in pools["train"]:
            hit = sealed.near(as_near(p), title_rule=False)
            if hit:
                drop[f"train posting near a {name} posting (another employer's copy)"] += 1
                dropped.append({"key": p["key"], "why": f"near {name} {hit[0]}"})
            else:
                survivors.append(p)
        pools["train"] = survivors
    dev_vs_test = RareIndex({p["key"]: as_near(p) for p in pools["test"]})
    pools["dev"] = [p for p in pools["dev"] if not dev_vs_test.near(as_near(p), title_rule=False)]

    notes["store fingerprint"] = {"postings": len(store), "sha256 sorted keys": sha_ids(p["key"] for p in store),
                                  "latest discovered_at": max(p["discovered_at"] for p in store)}
    write_split(pools, test_emp, dev_emp, reserved, drop, dropped, notes, len(store))


_SH: dict[str, frozenset] = {}


def near_pair(a: dict, b: dict) -> bool:
    """matcher.nearcopy's description rule between two postings (Jaccard of 6-word windows >= NEAR)."""
    sa = _SH.setdefault(a["key"], shingles(a["text"]))
    sb = _SH.setdefault(b["key"], shingles(b["text"]))
    if not a["text"].strip() or not b["text"].strip() or min(len(sa), len(sb)) < NEAR * max(len(sa), len(sb)):
        return False
    return len(sa & sb) / len(sa | sb) >= NEAR


def choose_employers(cands, by_emp, usable, rng, target, cap, min_emp, need_sources):
    """Employers -> their sampled postings. Greedy on deficits: each step adds the employer whose best postings fill
    the most-missing field and length cells (fields about equal, short teasers at least SHORT_SHARE, empty bodies at
    most EMPTY_SHARE), at most `cap` postings per employer and one per title inside an employer; each source in
    `need_sources` is seeded with one employer first. A posting that is a near copy of one already taken from its employer is
    skipped."""
    fq = {f: target / len(FIELDS) for f in FIELDS}
    short_q, empty_q = target * SHORT_SHARE, target * EMPTY_SHARE
    fc, state = collections.Counter(), {"short": 0, "empty": 0, "n": 0}
    pool = {}
    for e in cands:
        ps = sorted(usable(by_emp[e]), key=lambda p: p["key"])
        rng.shuffle(ps)
        seen, uniq = set(), []
        for p in ps:
            t = norm_text(p["title"])
            if t not in seen:
                seen.add(t)
                uniq.append(p)
        pool[e] = uniq

    relax = {"on": False}

    def gain(p, local_f, local_s, local_e):
        g = max(0.0, fq[p["field"]] - fc[p["field"]] - local_f[p["field"]]) / fq[p["field"]]
        if relax["on"]:
            g = max(g, 0.01)  # every field quota reachable is met: any posting helps fill the target
        if p["chars"] == 0:
            return g + 1.0 if state["empty"] + local_e < empty_q else -1.0
        if p["chars"] < SHORT and state["short"] + local_s < short_q:
            g += 1.0
        return g

    def sample(e):
        local_f, ls, le, out = collections.Counter(), 0, 0, []
        left = list(pool[e])
        while left and len(out) < cap:
            best = max(left, key=lambda p: gain(p, local_f, ls, le))
            if gain(best, local_f, ls, le) <= 0:
                break
            left.remove(best)
            if any(near_pair(best, q) for q in out):
                continue  # the same job under another location or pay line (DaVita "Care Coordinator - IKC - <town>")
            out.append(best)
            local_f[best["field"]] += 1
            ls += best["chars"] < SHORT and best["chars"] > 0
            le += best["chars"] == 0
        return out

    chosen: dict[str, list] = {}

    def add(e, s):
        chosen[e] = s
        for p in s:
            fc[p["field"]] += 1
            state["short"] += 0 < p["chars"] < SHORT
            state["empty"] += p["chars"] == 0
        state["n"] += len(s)

    for src in need_sources:
        best = max((e for e in cands if e not in chosen and collections.Counter(p["source"] for p in pool[e]).most_common(1)
                    and collections.Counter(p["source"] for p in pool[e]).most_common(1)[0][0] == src),
                   key=lambda e: len(sample(e)), default=None)
        if best:
            add(best, sample(best))
    while state["n"] < target or len(chosen) < min_emp:
        scored = []
        for e in cands:
            if e in chosen:
                continue
            s = sample(e)
            if s:
                scored.append((sum(gain(p, collections.Counter(), 0, 0) for p in s) / (1 + 0.15 * len(s)) + rng.random() * 1e-6, e, s))
        if not scored:
            if relax["on"]:
                break
            relax["on"] = True
            continue
        _, e, s = max(scored)
        if state["n"] + len(s) > target and len(chosen) >= min_emp:
            s = s[: max(1, target - state["n"])]
        add(e, s)
    return chosen


def sha_ids(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def mix(ps: list[dict]) -> dict:
    return {"n": len(ps),
            "employers": len({p["emp_key"] for p in ps}),
            "field": dict(sorted(collections.Counter(p["field"] for p in ps).items())),
            "length": {n: c for _, _, n in LENGTHS if (c := sum(p["length"] == n for p in ps))},
            "source": dict(collections.Counter(p["source"] for p in ps).most_common()),
            "max per employer": max(collections.Counter(p["emp_key"] for p in ps).values()) if ps else 0}


def write_split(pools, test_emp, dev_emp, reserved, drop, dropped, notes, store_n) -> None:
    POOLS.mkdir(parents=True, exist_ok=True)
    keep_fields = ("key", "source", "board", "employer", "title", "locations", "url", "posted_at", "workplace_type",
                   "opportunity_type", "description_kind", "text", "chars", "field", "length", "emp_key")
    for name, ps in pools.items():
        with open(POOLS / f"{name}.jsonl", "w") as f:
            for p in sorted(ps, key=lambda p: p["key"]):
                f.write(json.dumps({k: p[k] for k in keep_fields}, ensure_ascii=False) + "\n")
    (FIT / "dropped.jsonl").write_text("".join(json.dumps(d) + "\n" for d in dropped))

    def emp_list(ps):
        c = collections.Counter(p["emp_key"] for p in ps)
        names = collections.defaultdict(set)
        for p in ps:
            names[p["emp_key"]].add(p["employer"])
        return [{"employer": e, "names": sorted(names[e]), "n": n} for e, n in sorted(c.items(), key=lambda x: (-x[1], x[0]))]
    split_doc = {
        "seed": SEED, "store postings": store_n,
        "pools": {name: {"ids": sorted(p["key"] for p in ps), "file": f"data/built/fit/pools/{name}.jsonl",
                         "employers": sorted({p["emp_key"] for p in ps})} for name, ps in pools.items()},
        "reserved (test/dev employers' postings outside the sealed sample; in no pool, never trained on)":
            {k: sorted(v) for k, v in reserved.items()},
    }
    SPLIT.write_text(json.dumps(split_doc, indent=1))
    freeze = {
        "what": "v4 fit pools of venator's posting store, frozen before any fit data is built (scripts/fit_pools.py split)",
        "seed": SEED, "store postings": store_n,
        "used set": notes,
        "dedupe removed": dict(drop.most_common()), "dedupe removed total": sum(drop.values()),
        "rules": {"test cap per employer": TEST_CAP, "dev cap per employer": DEV_CAP, "max postings (after dedupe) of a test or dev employer": MAX_EMPLOYER,
                  "teaser share (test, dev)": f">= {SHORT_SHARE}", "empty-body share (test, dev)": f"<= {EMPTY_SHARE}",
                  "short teaser": f"< {SHORT} chars", "employers": "employer clusters (Employers: company keys "
                  "linked by despaced key, specific first word or first two words, initials); test and dev clusters hold "
                  "no used employer name and no placeholder name; test, dev and train employers disjoint; reserved postings of test "
                  "and dev employers are in no pool"},
        "pools": {name: {**mix(ps), "sha256 sorted ids": sha_ids(p["key"] for p in ps)} for name, ps in pools.items()},
        "reserved": {k: len(v) for k, v in reserved.items()},
        "employers": {"test": emp_list(pools["test"]), "dev": emp_list(pools["dev"]),
                      "train (count only)": len({p["emp_key"] for p in pools["train"]})},
    }
    FREEZE.parent.mkdir(parents=True, exist_ok=True)
    FREEZE.write_text(json.dumps(freeze, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: freeze[k] for k in ("dedupe removed", "dedupe removed total", "reserved")}, indent=1))
    for name in pools:
        print(name, json.dumps(freeze["pools"][name]))


def load_pool(name: str) -> list[dict]:
    """The postings of one pool ("test", "dev", "train"), checked against the frozen id hash."""
    ps = list(jsonl(POOLS / f"{name}.jsonl"))
    want = json.loads(FREEZE.read_text())["pools"][name]["sha256 sorted ids"]
    if sha_ids(p["key"] for p in ps) != want:
        raise SystemExit(f"{name} pool does not match eval/fit_freeze.json")
    return ps


# ---------------------------------------------------------------- protocol

PROTOCOL_DOC = {
    "what": "v4 fit evaluation protocol, frozen before any fit data is built; the pools are eval/fit_freeze.json",
    "systems": {
        "rule": "Each candidate model (g = SFT with reasoning traces, h = SFT without, i = DPO if it runs, d3 as the "
                "reference) is scored as a system: the eligibility layer (src/matcher/hybrid.py H' on the "
                "Adaption-extracted card, salary cap included) decides Skip where it fires, the model decides every "
                "other row.",
        "candidates": {"g": "SFT with reasoning traces", "h": "SFT without reasoning traces", "i": "DPO (if it runs)",
                       "d3": "reference"},
        "eligibility_layer": "src/matcher/hybrid.py H' on the Adaption-extracted card, salary cap included",
    },
    "selection": {
        "rule": "On data/built/fit/dev_fit.jsonl only. Pick the system with the highest macro-F1; if two are within "
                "0.01, prefer the one without reasoning traces (cheaper to serve). The fresh test plays no part in "
                "selection.",
        "data": "data/built/fit/dev_fit.jsonl", "metric": "macro-F1", "tie_margin": 0.01,
        "tie_break": "prefer the system without reasoning traces", "fresh_test_used": False,
    },
    "fresh_test": {
        "rule": "Scored once, for every system and these baselines on the same rows: skills-free rules (rules on the "
                "extracted card with must and nice skills removed), full rules on the extracted card, the untrained "
                "Qwen3.5-4B with the eligibility layer, and d3 + H'. Gold is data/built/fresh_test/gold.jsonl: two "
                "independent blind labelers, disagreements adjudicated blind to cards, rules and models.",
        "scored": "once",
        "baselines": ["skills-free rules (rules on the extracted card with must and nice skills removed)",
                      "full rules on the extracted card",
                      "the untrained Qwen3.5-4B with the eligibility layer",
                      "d3 + H'"],
        "gold": "data/built/fresh_test/gold.jsonl",
        "gold_process": "two independent blind labelers, disagreements adjudicated blind to cards, rules and models",
        "pool": "data/built/fit/pools/test.jsonl (sha256 in eval/fit_freeze.json); the test builder samples 150",
    },
    "headline": {
        "rule": "Accuracy and macro-F1 of the chosen system against skills-free rules, as a paired difference with 95% "
                "intervals from a bootstrap that resamples postings and candidates. Reported on all rows and on the "
                "eligibility-pass slice (rows the eligibility layer lets through, which are the rows that need the "
                "resume). Labeler agreement (raw and Cohen's kappa) is reported as the ceiling.",
        "metrics": ["accuracy", "macro-F1"], "comparison": "chosen system - skills-free rules, paired",
        "interval": {"level": 0.95, "method": "bootstrap resampling postings and candidates"},
        "slices": ["all rows", "eligibility-pass slice (rows the eligibility layer lets through)"],
        "ceiling": ["labeler raw agreement", "Cohen's kappa"],
    },
    "pass": {
        "rule": "The chosen system beats skills-free rules on the eligibility-pass slice with the interval above zero. "
                "Whatever the result, it is reported as it is.",
        "slice": "eligibility-pass slice", "against": "skills-free rules", "condition": "95% interval lower bound > 0",
    },
}


def protocol(_: argparse.Namespace) -> None:
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(PROTOCOL_DOC, indent=1, ensure_ascii=False) + "\n")
    print(PROTOCOL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot").set_defaults(fn=snapshot)
    sub.add_parser("split").set_defaults(fn=split)
    sub.add_parser("protocol").set_defaults(fn=protocol)
    a = ap.parse_args()
    a.fn(a)
