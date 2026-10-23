"""Read the O*NET 31.0 text files into per-occupation facts.

Only the tables the card sampler needs: occupation titles, job zone, modal
education level, median related-work-experience, and ranked skill pools
(essential skills, knowledge areas, software). Pure standard library.

Must-have skills come from `core_knowledge` (knowledge areas distinctive for
the occupation) and software. Names are shown the way postings write them
(`display_name`: "SQL", not "Structured query language SQL"); computing and
engineering occupations rank their software by O*NET's Hot Technology flags. The O*NET basic skills ("Essential Skills":
Critical Thinking, Active Learning, ...) and the knowledge areas every job rates
high (English Language, Mathematics) are ABSTRACT: real postings do not reject
anyone for them, so they are only ever nice-to-haves.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .schema import ONET_NAMES, SKILL_SYNONYMS

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "open" / "onet"

# Generic abilities nearly every occupation lists; they would make every
# must-have trivially met, so they stay out of the pools.
GENERIC_SKILLS = frozenset(
    {"Reading Comprehension", "Active Listening", "Speaking", "Writing", "Monitoring"}
)

# Every O*NET basic skill plus the knowledge areas that are generic rather than
# occupational. Never a must-have.
ABSTRACT_SKILLS = frozenset(
    {
        "Reading Comprehension", "Active Listening", "Speaking", "Writing", "Monitoring",
        "Mathematics", "Science", "Critical Thinking", "Active Learning", "Learning Strategies",
        "English Language",
    }
)
# Software every job uses that no posting would require. Never a must-have.
GENERIC_SOFTWARE = frozenset(
    {"Web browser software", "Email software", "Microsoft Windows", "Facebook", "FaceTime", "YouTube", "AOL MapQuest"}
)
# A knowledge area is core for an occupation when the occupation rates it at least
# CORE_MIN_IMPORTANCE (O*NET IM scale 1-5) and at least CORE_MIN_LIFT above the mean
# rating across all occupations.
CORE_MIN_IMPORTANCE = 3.0
CORE_MIN_LIFT = 0.5

# The Office suite: every job lists it, so tech occupations rank it below their real tools.
OFFICE_SUITE = frozenset({
    "Microsoft Excel", "Microsoft Office software", "Microsoft PowerPoint", "Microsoft Word", "Microsoft Outlook",
    "Microsoft Access", "Microsoft Teams", "Microsoft Windows", "Microsoft SharePoint", "Microsoft Visio",
    "Microsoft Project", "Adobe Acrobat", "Microsoft OneNote", "Google Docs", "Google Sheets", "Microsoft OneDrive",
})
# SOC major groups whose software list is ranked by O*NET's Hot Technology / In Demand flags
# (Python, SQL, C++, Git ...) rather than by how common a tool is in the minor group.
TECH_MAJORS = ("15-", "17-")

_PLAIN = {alt: name for table in (SKILL_SYNONYMS, ONET_NAMES) for name, alts in table.items() for alt in alts}


def display_name(onet_name: str) -> str:
    """O*NET element or software name -> the plain name a posting uses
    ("Structured query language SQL" -> "SQL", "Customer and Personal Service" -> "Customer service")."""
    if onet_name in _PLAIN:
        return _PLAIN[onet_name]
    if onet_name.endswith(" software") and len(onet_name.split()) > 2:
        return onet_name[: -len(" software")]
    return onet_name


# O*NET Education category (RL scale) -> our degree level.
_EDU_TO_LEVEL = {
    1: "none", 2: "none", 3: "none", 4: "none",
    5: "associate", 6: "bachelor", 7: "bachelor",
    8: "master", 9: "master", 10: "doctoral", 11: "doctoral", 12: "doctoral",
}
# Related work experience category (RW scale) -> years.
_RW_TO_YEARS = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 2, 8: 4, 9: 6, 10: 8, 11: 10}
_ZONE_TO_LEVEL = {1: "none", 2: "none", 3: "associate", 4: "bachelor", 5: "master"}

# Hand list: SOC prefix -> license the occupation needs. Longest prefix wins.
LICENSES: dict[str, str] = {
    "29-1141": "RN license",
    "29-1171": "NP license",
    "29-1151": "RN license",
    "29-2061": "LPN license",
    "31-1131": "CNA certification",
    "31-1132": "CNA certification",
    "31-1133": "CNA certification",
    "29-2052": "Pharmacy technician certification",
    "29-1051": "Pharmacist license",
    "13-2011": "CPA license",
    "13-2052": "Series 65",
    "25-2": "State teaching license",
    "53-3032": "CDL Class A",
    "47-2111": "Journeyman electrician license",
    "47-2152": "Plumber license",
    "29-2042": "EMT certification",
    "29-2043": "EMT certification",
    "29-1126": "RRT license",
    "29-1123": "Physical therapist license",
    "29-1122": "Occupational therapist license",
    "29-1021": "Dental license",
    "29-12": "Medical license",
    "23-1011": "Bar admission",
    "21-1022": "LCSW license",
    "29-2011": "Clinical laboratory scientist license",
    "29-2012": "MLT certification",
    "29-2034": "ARRT certification",
    "29-2056": "Veterinary technician credential",
    "29-2021": "Dental hygiene license",
    "31-9097": "Phlebotomy certification",
    "39-5012": "Cosmetology license",
    "33-9032": "Security guard license",
    "29-1131": "Veterinary license",
    "33-2011": "Firefighter certification",
    "33-3051": "Police academy certification",
    "53-2011": "FAA commercial pilot certificate",
    "51-8031": "Water operator license",
    "49-9021": "EPA 608 certification",
}




@dataclass(frozen=True)
class LicenseNeeds:
    """The education a license follows from. fields: the degree fields that lead to it (None:
    any field); level: the lowest degree that does ("none": a certificate or apprenticeship,
    so no degree or a degree in one of `fields`). domain groups licenses for swaps."""

    domain: str
    fields: tuple[str, ...] | None
    level: str
    max_level: str = "doctoral"  # Cosmetology is a certificate or associate program, never a bachelor's
    # Entry certificates (CNA, LPN, pharmacy technician, security guard, EMT, phlebotomy) are held up
    # to a bachelor's, often by students on the way to the next step; not by someone with a finished
    # graduate degree. Veterinary technology is an associate or bachelor's program.


_TRADES = ("Electrical Technology", "Engineering Technology", "Industrial Technology")
LICENSE_EDUCATION: dict[str, LicenseNeeds] = {
    "RN license": LicenseNeeds("nursing", ("Nursing",), "associate"),
    "NP license": LicenseNeeds("nursing", ("Nursing",), "master"),
    "LPN license": LicenseNeeds("nursing", None, "none", "bachelor"),
    "CNA certification": LicenseNeeds("nursing", None, "none", "bachelor"),
    "Pharmacy technician certification": LicenseNeeds("pharmacy", None, "none", "bachelor"),
    "Pharmacist license": LicenseNeeds("pharmacy", ("Pharmacy",), "doctoral"),
    "CPA license": LicenseNeeds("finance", ("Accounting", "Finance", "Business Administration"), "bachelor"),
    "Series 65": LicenseNeeds("finance", None, "none"),
    "State teaching license": LicenseNeeds("education", ("Education", "Elementary Education", "Special Education",
                                                         "English", "Mathematics", "Biology", "Chemistry", "History"),
                                           "bachelor"),
    "CDL Class A": LicenseNeeds("transport", None, "none"),
    "FAA commercial pilot certificate": LicenseNeeds("transport", None, "none"),
    "Journeyman electrician license": LicenseNeeds("trades", _TRADES, "none"),
    "Plumber license": LicenseNeeds("trades", (*_TRADES, "Construction Management"), "none"),
    "EPA 608 certification": LicenseNeeds("trades", ("HVAC Technology", *_TRADES), "none"),
    "Water operator license": LicenseNeeds("trades", None, "none"),
    "EMT certification": LicenseNeeds("emergency", None, "none", "bachelor"),
    "Firefighter certification": LicenseNeeds("emergency", ("Fire Science",), "none"),
    "Police academy certification": LicenseNeeds("protective", ("Criminal Justice",), "none"),
    "Security guard license": LicenseNeeds("protective", None, "none", "bachelor"),
    "RRT license": LicenseNeeds("allied health", ("Respiratory Therapy",), "associate"),
    "ARRT certification": LicenseNeeds("allied health", ("Radiologic Technology",), "associate"),
    "Phlebotomy certification": LicenseNeeds("allied health", None, "none", "bachelor"),
    "Clinical laboratory scientist license": LicenseNeeds(
        "lab", ("Medical Laboratory Science", "Biology", "Chemistry", "Biochemistry", "Microbiology"), "bachelor"),
    "MLT certification": LicenseNeeds("lab", ("Medical Laboratory Technology", "Biology", "Chemistry"), "associate"),
    "Physical therapist license": LicenseNeeds("therapy", ("Physical Therapy",), "doctoral"),
    "Occupational therapist license": LicenseNeeds("therapy", ("Occupational Therapy",), "master"),
    "Medical license": LicenseNeeds("medicine", ("Medicine",), "doctoral"),
    "Dental license": LicenseNeeds("dental", ("Dentistry",), "doctoral"),
    "Dental hygiene license": LicenseNeeds("dental", ("Dental Hygiene",), "associate"),
    "Veterinary license": LicenseNeeds("veterinary", ("Veterinary Medicine",), "doctoral"),
    "Veterinary technician credential": LicenseNeeds("veterinary", ("Veterinary Technology",), "associate", "bachelor"),
    "Bar admission": LicenseNeeds("legal", ("Law",), "doctoral"),
    "LCSW license": LicenseNeeds("social work", ("Social Work",), "master"),
    "Cosmetology license": LicenseNeeds("personal care", ("Cosmetology",), "none", "associate"),
}
_LEVELS = ("none", "associate", "bachelor", "master", "doctoral")


def license_fits(license: str, degree_level: str, degree_field: str) -> bool:
    """Could someone with this finished degree hold this license? Unknown licenses always fit."""
    from .degrees import canon_field

    need = LICENSE_EDUCATION.get(license)
    if need is None:
        return True
    level = _LEVELS.index(degree_level)
    if level < _LEVELS.index(need.level) or level > _LEVELS.index(need.max_level):
        return False
    if degree_level == "none" or need.fields is None:
        return True
    return canon_field(degree_field) in {canon_field(f) for f in need.fields}


def licenses_fit(licenses: tuple[str, ...], degree_level: str, degree_field: str) -> bool:
    return all(license_fits(x, degree_level, degree_field) for x in licenses)


@dataclass
class Occupation:
    soc: str
    title: str
    description: str
    job_zone: int
    degree_level: str
    typical_years: int
    essential: list[str] = field(default_factory=list)  # ranked, best first
    knowledge: list[str] = field(default_factory=list)
    software: list[str] = field(default_factory=list)
    license: str | None = None
    # knowledge areas distinctive for this occupation, most distinctive first; never abstract
    core_knowledge: list[str] = field(default_factory=list)
    # O*NET Hot Technologies for this occupation, in-demand first (display names)
    hot_tech: list[str] = field(default_factory=list)

    @property
    def tools(self) -> list[str]:
        """Software a posting could require: the ranked list without GENERIC_SOFTWARE."""
        generic = {display_name(x) for x in GENERIC_SOFTWARE}
        return [x for x in self.software if x not in generic]

    @property
    def pool(self) -> list[str]:
        """All named skills, interleaved best-first, no duplicates."""
        out: list[str] = []
        lists = [self.knowledge, self.essential, self.software]
        for i in range(max(len(x) for x in lists)):
            for lst in lists:
                if i < len(lst) and lst[i] not in out:
                    out.append(lst[i])
        return out


def license_for(soc: str) -> str | None:
    best = None
    for prefix, name in LICENSES.items():
        if soc.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, name)
    return best[1] if best else None


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh, delimiter="\t")


def _scores(path: Path, name_col: str, soc_ok: set[str]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """(importance, level) per occupation and element name, generic skills left out."""
    im: dict[str, dict[str, float]] = {}
    lv: dict[str, dict[str, float]] = {}
    for r in _rows(path):
        soc = r["O*NET-SOC Code"]
        if soc not in soc_ok or r["Recommend Suppress"] == "Y" or r["Scale ID"] not in ("IM", "LV"):
            continue
        name = r[name_col]
        if name in GENERIC_SKILLS:
            continue
        target = im if r["Scale ID"] == "IM" else lv
        target.setdefault(soc, {})[name] = float(r["Data Value"])
    return im, lv


def _ranked(im: dict[str, dict[str, float]], lv: dict[str, dict[str, float]], top: int) -> dict[str, list[str]]:
    """Rank element names per occupation by importance (IM), level as tiebreak."""
    out: dict[str, list[str]] = {}
    for soc, scores in im.items():
        levels = lv.get(soc, {})
        names = sorted(scores, key=lambda n: (-scores[n], -levels.get(n, 0), n))
        out[soc] = names[:top]
    return out


def _core(im: dict[str, dict[str, float]], top: int = 5) -> dict[str, list[str]]:
    """Knowledge areas this occupation rates important and well above the all-occupation
    mean, most distinctive first. "Food Production" for cooks, never "English Language"."""
    names = {n for scores in im.values() for n in scores}
    mean = {n: sum(s.get(n, 0.0) for s in im.values()) / len(im) for n in names}
    out: dict[str, list[str]] = {}
    for soc, scores in im.items():
        ok = [n for n, v in scores.items()
              if n not in ABSTRACT_SKILLS and v >= CORE_MIN_IMPORTANCE and v - mean[n] >= CORE_MIN_LIFT]
        out[soc] = sorted(ok, key=lambda n: (-(scores[n] - mean[n]), n))[:top]
    return out


def _modal(path: Path, scale: str, soc_ok: set[str]) -> dict[str, int]:
    best: dict[str, tuple[float, int]] = {}
    for r in _rows(path):
        soc = r["O*NET-SOC Code"]
        if soc not in soc_ok or r["Scale ID"] != scale or r["Category"] == "n/a":
            continue
        v, cat = float(r["Data Value"]), int(r["Category"])
        if soc not in best or v > best[soc][0]:
            best[soc] = (v, cat)
    return {k: v[1] for k, v in best.items()}


def _median_rw(path: Path, soc_ok: set[str]) -> dict[str, int]:
    dist: dict[str, dict[int, float]] = {}
    for r in _rows(path):
        soc = r["O*NET-SOC Code"]
        if soc in soc_ok and r["Scale ID"] == "RW" and r["Category"] != "n/a":
            dist.setdefault(soc, {})[int(r["Category"])] = float(r["Data Value"])
    out = {}
    for soc, d in dist.items():
        total, acc = sum(d.values()) or 1.0, 0.0
        for cat in sorted(d):
            acc += d[cat]
            if acc >= total / 2:
                out[soc] = cat
                break
    return out


def load_occupations(onet_dir: Path = DEFAULT_DIR) -> list[Occupation]:
    """Every occupation with a job zone and skill data, sorted by SOC code."""
    zones = {r["O*NET-SOC Code"]: int(r["Job Zone"]) for r in _rows(onet_dir / "Job Zones.txt")}
    occ_rows = {
        r["O*NET-SOC Code"]: r
        for r in _rows(onet_dir / "Occupation Data.txt")
        if "All Other" not in r["Title"] and r["O*NET-SOC Code"] in zones
    }
    ok = set(occ_rows)
    essential = _ranked(*_scores(onet_dir / "Essential Skills.txt", "Element Name", ok), 12)
    k_im, k_lv = _scores(onet_dir / "Knowledge.txt", "Element Name", ok)
    knowledge = _ranked(k_im, k_lv, 10)
    core = _core(k_im)

    # Software: prefer tools used across the occupation's minor group (e.g. 29-1),
    # so a nurse gets Epic, not a one-off data-science library.
    soft_score: dict[str, dict[str, int]] = {}
    in_group: dict[tuple[str, str], set[str]] = {}
    for r in _rows(onet_dir / "Software Skills.txt"):
        soc = r["O*NET-SOC Code"]
        if soc not in ok:
            continue
        score = (2 if r["Hot Technology"] == "Y" else 0) + (1 if r["In Demand"] == "Y" else 0)
        d = soft_score.setdefault(soc, {})
        d[r["Workplace Example"]] = max(score, d.get(r["Workplace Example"], 0))
        in_group.setdefault((soc[:4], r["Workplace Example"]), set()).add(soc)
    hot_count = Counter(n for (_, n), socs in in_group.items() for s in socs if soft_score[s][n] >= 2)
    software = {}
    hot = {}
    for soc, d in soft_score.items():
        ranked = sorted(d, key=lambda n: (-len(in_group[(soc[:4], n)]), -d[n], n))
        common = [n for n in ranked if len(in_group[(soc[:4], n)]) >= 3]
        hot_ranked = sorted((n for n in d if d[n] >= 2 and n not in GENERIC_SOFTWARE),
                            key=lambda n: (-d[n], n in OFFICE_SUITE, -hot_count[n], n))
        if soc.startswith(TECH_MAJORS):
            real_tools = [n for n in hot_ranked if n not in OFFICE_SUITE]
            pick = real_tools[:10] + [n for n in (common or ranked) if n not in real_tools][: max(0, 8 - len(real_tools))]
            software[soc] = pick
        else:
            software[soc] = (common or ranked)[:8]
        hot[soc] = [display_name(n) for n in hot_ranked[:16]]

    edu = _modal(onet_dir / "Education.txt", "RL", ok)
    rw = _median_rw(onet_dir / "Training and Experience.txt", ok)

    occs: list[Occupation] = []
    for soc in sorted(ok):
        if soc not in essential or soc not in knowledge:
            continue
        zone = zones[soc]
        level = _EDU_TO_LEVEL.get(edu.get(soc, 0)) or _ZONE_TO_LEVEL.get(zone, "none")
        occs.append(
            Occupation(
                soc=soc,
                title=occ_rows[soc]["Title"],
                description=occ_rows[soc]["Description"],
                job_zone=zone,
                degree_level=level,
                typical_years=_RW_TO_YEARS.get(rw.get(soc, 1), 0),
                essential=essential[soc],
                knowledge=[display_name(x) for x in knowledge[soc]],
                software=list(dict.fromkeys(display_name(x) for x in software.get(soc, []))),
                license=license_for(soc),
                core_knowledge=[display_name(x) for x in core.get(soc, [])],
                hot_tech=hot.get(soc, []),
            )
        )
    return occs
