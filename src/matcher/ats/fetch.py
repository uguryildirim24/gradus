"""Fetch full posting text for Simplify listings from public ATS endpoints.

Greenhouse, Lever, Ashby and Workday only. Polite: per-host concurrency cap,
delays, retries with backoff, a clear User-Agent. Resume-safe.
"""
from __future__ import annotations

import asyncio
import html as htmllib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import httpx

CONTACT = os.environ.get("MATCHER_CONTACT", "")  # a contact address for the crawler's User-Agent, if set
UA = "gradus-research/0.1 (student hackathon project" + (f"; contact {CONTACT}" if CONTACT else "") + ")"
HOST_CONCURRENCY = 3
DELAY = (0.15, 0.4)

BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section"}


class _Strip(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "li":
            self.out.append("\n- ")
        elif tag in BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        self.out.append(data)


def html_to_text(s: str | None) -> str:
    if not s:
        return ""
    # Greenhouse double-escapes its HTML
    if "&lt;" in s and "<" not in s:
        s = htmllib.unescape(s)
    p = _Strip()
    p.feed(s)
    t = "".join(p.out).replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" ?\n ?", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def classify(url: str):
    """Return (ats, params) or None."""
    u = urlparse(url)
    host = u.netloc.lower()
    parts = [unquote(x) for x in u.path.split("/") if x]
    if host.endswith("greenhouse.io"):
        q = parse_qs(u.query)
        if "jobs" in parts:
            i = parts.index("jobs")
            if i >= 1 and i + 1 < len(parts) and parts[i + 1].isdigit() and parts[i - 1] != "embed":
                return "greenhouse", {"token": parts[i - 1], "id": parts[i + 1]}
        if "for" in q and "token" in q:
            return "greenhouse", {"token": q["for"][0], "id": q["token"][0]}
        return None
    if host == "jobs.lever.co" and len(parts) >= 2:
        return "lever", {"company": parts[0], "id": parts[1]}
    if host.endswith("ashbyhq.com") and len(parts) >= 2:
        return "ashby", {"org": parts[0], "id": parts[1]}
    if host.endswith("myworkdayjobs.com"):
        tenant = host.split(".")[0]
        if "job" in parts:
            i = parts.index("job")
            site = parts[i - 1] if i >= 1 else None
            if site and not re.fullmatch(r"[a-z]{2}-[A-Z]{2}", site):
                rest = "/".join(parts[i:])
                return "workday", {"host": host, "tenant": tenant, "site": site, "path": rest}
        return None
    return None


class Gone(Exception):
    pass


class Fetcher:
    def __init__(self):
        self.client = httpx.AsyncClient(headers={"User-Agent": UA, "Accept": "application/json"},
                                        timeout=30, follow_redirects=True)
        self.sems: dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(HOST_CONCURRENCY))
        self.ashby_cache: dict[str, asyncio.Future] = {}

    async def get(self, url: str, retries: int = 5):
        host = urlparse(url).netloc
        for attempt in range(retries):
            async with self.sems[host]:
                await asyncio.sleep(random.uniform(*DELAY))
                try:
                    r = await self.client.get(url)
                except httpx.HTTPError:
                    r = None
            if r is not None:
                if r.status_code == 200:
                    return r
                if r.status_code in (404, 410):
                    raise Gone(str(r.status_code))
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"http {r.status_code}")
                ra = r.headers.get("retry-after")
                wait = float(ra) if ra and ra.isdigit() else 0
            else:
                wait = 0
            await asyncio.sleep(max(wait, 2 ** attempt + random.random()))
        raise RuntimeError("retries exhausted")

    async def greenhouse(self, p):
        r = await self.get(f"https://boards-api.greenhouse.io/v1/boards/{p['token']}/jobs/{p['id']}")
        j = r.json()
        return html_to_text(j.get("content")), (j.get("location") or {}).get("name"), j.get("title")

    async def lever(self, p):
        for base in ("https://api.lever.co", "https://api.eu.lever.co"):
            try:
                r = await self.get(f"{base}/v0/postings/{p['company']}/{p['id']}")
            except Gone:
                continue
            j = r.json()
            parts = [j.get("descriptionPlain") or html_to_text(j.get("description"))]
            for lst in j.get("lists") or []:
                parts.append(f"{lst.get('text', '')}\n{html_to_text(lst.get('content'))}")
            parts.append(j.get("additionalPlain") or html_to_text(j.get("additional")))
            cat = j.get("categories") or {}
            return "\n\n".join(x for x in parts if x).strip(), cat.get("location"), j.get("text")
        raise Gone("404")

    async def _ashby_board(self, org):
        if org not in self.ashby_cache:
            fut = asyncio.get_running_loop().create_future()
            self.ashby_cache[org] = fut
            try:
                r = await self.get(f"https://api.ashbyhq.com/posting-api/job-board/{org}?includeCompensation=true")
                fut.set_result({j["id"]: j for j in r.json().get("jobs", [])})
            except Exception as e:  # noqa: BLE001
                fut.set_exception(e)
                fut.exception()  # mark retrieved
        return await self.ashby_cache[org]

    async def ashby(self, p):
        jobs = await self._ashby_board(p["org"])
        j = jobs.get(p["id"])
        if not j:
            raise Gone("not on board")
        text = j.get("descriptionPlain") or html_to_text(j.get("descriptionHtml"))
        comp = (j.get("compensation") or {}).get("compensationTierSummary")
        if comp:
            text += f"\n\nCompensation: {comp}"
        return text.strip(), j.get("location"), j.get("title")

    async def workday(self, p):
        url = f"https://{p['host']}/wday/cxs/{p['tenant']}/{p['site']}/{p['path']}"
        r = await self.get(url)
        info = r.json().get("jobPostingInfo", {})
        return html_to_text(info.get("jobDescription")), info.get("location"), info.get("title")

    async def fetch(self, ats, p):
        return await getattr(self, ats)(p)

    async def close(self):
        await self.client.aclose()


def load_ids(path: Path, key="simplify_id") -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.open():
        try:
            out.add(json.loads(line)[key])
        except Exception:  # noqa: BLE001
            pass
    return out


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


async def run(listings_path: Path, out_dir: Path, limit: int | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    posts_p, fail_p = out_dir / "postings.jsonl", out_dir / "failures.jsonl"
    rows = json.load(listings_path.open())
    todo = []
    unsupported = 0
    for r in rows:
        if not (r.get("active") and r.get("is_visible")):
            continue
        c = classify(r["url"])
        if c is None:
            h = urlparse(r["url"]).netloc
            if h.endswith(("greenhouse.io", "lever.co", "ashbyhq.com", "myworkdayjobs.com")):
                unsupported += 1
            continue
        todo.append((r, *c))
    done = load_ids(posts_p)
    gone = {json.loads(l)["simplify_id"] for l in fail_p.open() if json.loads(l).get("permanent")} if fail_p.exists() else set()
    todo = [t for t in todo if t[0]["id"] not in done and t[0]["id"] not in gone]
    if limit:
        todo = todo[:limit]
    print(f"to fetch: {len(todo)} (already {len(done)}, known-gone {len(gone)}, unparsable ATS urls {unsupported})", flush=True)

    f = Fetcher()
    wlock = asyncio.Lock()
    pf, ff = posts_p.open("a"), fail_p.open("a")
    n = 0
    stats = Counter()
    gsem = asyncio.Semaphore(24)

    async def work(r, ats, p):
        nonlocal n
        async with gsem:
            try:
                text, loc, title = await f.fetch(ats, p)
                if not text or len(text) < 50:
                    raise Gone("empty description")
                rec = {
                    "simplify_id": r["id"], "company": r["company_name"], "title": r["title"],
                    "locations": r.get("locations"), "url": r["url"], "ats": ats,
                    "sponsorship": r.get("sponsorship"), "degrees": r.get("degrees"),
                    "category": r.get("category"), "terms": r.get("terms"),
                    "ats_title": title, "ats_location": loc,
                    "description": text,
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                async with wlock:
                    pf.write(json.dumps(rec, ensure_ascii=False) + "\n"); pf.flush()
                stats[(ats, "ok")] += 1
            except Gone as e:
                async with wlock:
                    ff.write(json.dumps({"simplify_id": r["id"], "ats": ats, "url": r["url"], "error": str(e), "permanent": True}) + "\n"); ff.flush()
                stats[(ats, "gone")] += 1
            except Exception as e:  # noqa: BLE001
                async with wlock:
                    ff.write(json.dumps({"simplify_id": r["id"], "ats": ats, "url": r["url"], "error": repr(e)[:200], "permanent": False}) + "\n"); ff.flush()
                stats[(ats, "error")] += 1
            n += 1
            if n % 100 == 0:
                print(n, dict(stats), flush=True)

    random.shuffle(todo)
    await asyncio.gather(*(work(*t) for t in todo))
    await f.close()
    pf.close(); ff.close()
    print("final", dict(stats))


def report(out_dir: Path):
    posts = [json.loads(l) for l in (out_dir / "postings.jsonl").open()]
    by = Counter(p["ats"] for p in posts)
    lens = [len(p["description"]) for p in posts]
    fails = Counter()
    fp = out_dir / "failures.jsonl"
    if fp.exists():
        latest = {}
        for l in fp.open():
            d = json.loads(l); latest[d["simplify_id"]] = d
        ok = {p["simplify_id"] for p in posts}
        for d in latest.values():
            if d["simplify_id"] not in ok:
                fails[(d["ats"], d["error"] if d["permanent"] else "transient")] += 1
    print("postings:", len(posts), dict(by))
    print("median description chars:", median(lens), "min", min(lens), "max", max(lens))
    for ats in by:
        print(" ", ats, "median", median([len(p["description"]) for p in posts if p["ats"] == ats]))
    print("failures:", dict(fails))
