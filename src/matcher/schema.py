"""Card schemas: what a posting requires and what a candidate has.

Everything the label rule may look at lives here. Fields marked NUISANCE exist
only so fairness swaps can vary them; the rule never reads them.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from typing import Any

DEGREE_LEVELS = ("none", "associate", "bachelor", "master", "doctoral")
SENIORITIES = ("entry", "mid", "senior", "lead")
WORK_MODES = ("onsite", "hybrid", "remote")
# will_not_sponsor_or_opt: the posting also refuses OPT / STEM OPT (I-983), now or later.
SPONSORSHIP = ("sponsors", "will_not_sponsor", "will_not_sponsor_or_opt", "unstated")
EMPLOYMENT_TYPES = ("internship", "full_time", "part_time", "contract")
WORK_AUTH = (
    "citizen_or_pr",
    "f1_cpt",
    "f1_opt",
    "stem_opt",
    "needs_h1b_now",
    "other_visa",
)
# What work modes a candidate accepts.
REMOTE_PREFS = ("any", "hybrid_or_remote", "remote_only")

BUCKETS = ("For you", "Explore", "Excluded")

# The year the rule treats as "now": a candidate whose graduation_year is later
# is still enrolled. A constant, never the clock, so labels stay deterministic.
CURRENT_YEAR = 2026

SkillGroup = tuple[str, ...]  # alternatives; any one meets the group


def norm(text: str) -> str:
    """Lowercase and collapse whitespace so set lookups are stable."""
    return " ".join(text.lower().split())


_SKILL_STRIP = " .,;:"


def skill_key(text: str) -> str:
    """Light skill normalization: casefold, hyphens and underscores as spaces,
    collapsed whitespace, no trailing punctuation. "C++", "C#" and ".NET" survive."""
    t = text.casefold().replace("-", " ").replace("_", " ")
    return " ".join(t.split()).rstrip(_SKILL_STRIP)


def _synonym_index() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, alts in SKILL_SYNONYMS.items():
        for x in (name, *alts):
            out.setdefault(skill_key(x), skill_key(name))
    return out


def skill_canon(text: str) -> str:
    """skill_key, then every known wording of a skill mapped to its plain name's key."""
    k = skill_key(text)
    return _SYNONYMS.get(k, k)


def is_soft_skill(text: str) -> bool:
    return skill_canon(text) in SOFT_SKILLS


# Plain skill name -> other wordings people write for the same skill on resumes and postings
# (abbreviations, vendor prefixes, versions). Cards use the plain name; the rule and the
# visibility check treat every wording as that name (skill_canon), and the synonym variant
# swaps one in. O*NET's own element names live in ONET_NAMES, never here.
SKILL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "SQL": ("SQL programming",),
    "Java": ("Oracle Java", "Java programming"),
    "JavaScript": ("JS", "ECMAScript"),
    "TypeScript": ("TS",),
    "Python": ("Python programming", "Python 3", "Python3"),
    "C++": ("CPP",),
    "C#": ("C sharp", "CSharp"),
    "HTML": ("HTML5",),
    "CSS": ("Cascading style sheets", "CSS3"),
    "AWS": ("Amazon Web Services", "Amazon AWS"),
    "Azure": ("Microsoft Azure",),
    "Google Cloud": ("Google Cloud Platform", "GCP"),
    "Angular": ("AngularJS",),
    "React": ("React.js", "ReactJS"),
    "Node.js": ("NodeJS", "Node"),
    "Terraform": ("HashiCorp Terraform",),
    "Jenkins": ("Jenkins CI",),
    "Jira": ("Atlassian JIRA", "JIRA"),
    "Confluence": ("Atlassian Confluence",),
    "Bitbucket": ("Atlassian Bitbucket",),
    "Subversion": ("SVN",),
    "Kubernetes": ("K8s",),
    "Linux": ("GNU/Linux",),
    "Unix": ("UNIX",),
    "Shell scripting": ("Shell script", "Bash scripting", "UNIX Shell"),
    "MATLAB": ("Matlab",),
    "AutoCAD": ("Autodesk AutoCAD",),
    "AutoCAD Civil 3D": ("Autodesk AutoCAD Civil 3D", "Civil 3D"),
    "Revit": ("Autodesk Revit",),
    "SolidWorks": ("SOLIDWORKS",),
    "ArcGIS": ("ESRI ArcGIS", "Esri ArcGIS"),
    "SPSS": ("IBM SPSS Statistics", "IBM SPSS"),
    "Power BI": ("Microsoft Power BI", "PowerBI"),
    "Tableau": ("Salesforce Tableau",),
    "QuickBooks": ("Intuit QuickBooks",),
    "Salesforce": ("Salesforce CRM",),
    "SAP": ("SAP ERP",),
    "Splunk": ("Splunk Enterprise",),
    "VBA": ("Visual Basic for Applications",),
    "Visual Basic": ("Microsoft Visual Basic",),
    "SQL Server": ("Microsoft SQL Server", "MS SQL Server", "MSSQL"),
    "PowerShell": ("Microsoft PowerShell",),
    "SharePoint": ("Microsoft SharePoint",),
    "Microsoft Office": ("MS Office", "Office 365", "Microsoft 365"),
    "Microsoft Excel": ("Excel", "MS Excel"),
    "Microsoft Word": ("MS Word",),
    "Microsoft PowerPoint": ("PowerPoint", "MS PowerPoint"),
    "Microsoft Access": ("MS Access",),
    "Microsoft Project": ("MS Project",),
    "Microsoft Visio": ("Visio", "MS Visio"),
    "Microsoft Outlook": ("Outlook",),
    "Epic": ("Epic Systems", "Epic EHR", "Epic EMR"),
    "Electronic medical record EMR": ("Patient electronic medical record EMR", "Electronic medical records", "EMR"),
    "Adobe Creative Cloud": ("Adobe Creative Suite",),
    "Photoshop": ("Adobe Photoshop",),
    "Illustrator": ("Adobe Illustrator",),
    "InDesign": ("Adobe InDesign",),
    "Premiere Pro": ("Adobe Premiere Pro",),
    "After Effects": ("Adobe After Effects",),
    "Oracle Database": ("Oracle DB",),
    "PostgreSQL": ("Postgres",),
    "MongoDB": ("Mongo",),
    "Git": ("Git version control",),
    "Machine learning": ("ML",),
    "Artificial intelligence": ("AI",),
    "Natural language processing": ("NLP",),
    "Customer service": ("Client service", "Customer support"),
    "Human resources": ("HR",),
    "Design": ("Technical design",),
    "Go": ("Golang",),
    "Programming": ("Programming fundamentals", "General purpose programming", "Programming languages",
                    "Programming skills", "Computer programming", "Software programming"),
    "Coding": ("Coding skills",),
    "Debugging": ("Software debugging",),
    "Scripting": ("Scripting languages",),
    "Mathematics": ("Math", "Maths", "Mathematical skills", "Mathematics skills"),
    "Research": ("Research skills", "Research experience"),
}

# Plain name -> O*NET's element or software name for it. Only onet.display_name reads this:
# cards show the plain name, and no resume writes "Medicine and Dentistry".
ONET_NAMES: dict[str, tuple[str, ...]] = {
    "SQL": ("Structured query language SQL", "Structured query language"),
    "HTML": ("Hypertext markup language HTML", "Hypertext markup language"),
    "CSS": ("Cascading style sheets CSS",),
    "XML": ("Extensible markup language XML",),
    "JSON": ("JavaScript Object Notation JSON",),
    "AWS": ("Amazon Web Services AWS software",),
    "Azure": ("Microsoft Azure software",),
    "Angular": ("Google Angular",),
    "Terraform": ("IBM Terraform",),
    "Subversion": ("Apache Subversion SVN",),
    "MATLAB": ("The MathWorks MATLAB",),
    "Simulink": ("The MathWorks Simulink",),
    "SolidWorks": ("Dassault Systemes SolidWorks",),
    "CATIA": ("Dassault Systemes CATIA",),
    "ArcGIS": ("ESRI ArcGIS software",),
    "Tableau": ("Tableau software",),
    "Salesforce": ("Salesforce software",),
    "SAP": ("SAP software",),
    "ServiceNow": ("ServiceNow software",),
    "VBA": ("Microsoft Visual Basic for Applications VBA",),
    "Microsoft Office": ("Microsoft Office software",),
    "MEDITECH": ("MEDITECH software",),
    "eClinicalWorks": ("eClinicalWorks EHR software",),
    "Adobe Creative Cloud": ("Adobe Creative Cloud software",),
    "Ansible": ("Ansible software",),
    "Customer service": ("Customer and Personal Service",),
    "Clinical medicine": ("Medicine and Dentistry",),
    "Computer systems": ("Computers and Electronics",),
    "Engineering principles": ("Engineering and Technology",),
    "Accounting": ("Economics and Accounting",),
    "Business management": ("Administration and Management",),
    "Office administration": ("Administrative", "Clerical"),
    "Counseling": ("Therapy and Counseling",),
    "Law and regulations": ("Law and Government",),
    "Human resources": ("Personnel and Human Resources",),
    "Teaching and training": ("Education and Training",),
    "Public safety": ("Public Safety and Security",),
    "Construction methods": ("Building and Construction",),
    "Production processes": ("Production and Processing",),
    "Media production": ("Communications and Media",),
    "Mechanical systems": ("Mechanical",),
    "Sociology": ("Sociology and Anthropology",),
    "Philosophy": ("Philosophy and Theology",),
    "History": ("History and Archeology",),
    "Transportation logistics": ("Transportation",),
    "Foreign language": ("Foreign Language",),
    "Food production": ("Food Production",),
    "Fine arts": ("Fine Arts",),
    "Sales and marketing": ("Sales and Marketing",),
}

# Abilities every posting can ask for and no resume can prove. A must-have group made only
# of these is treated as a nice-to-have that never counts against anyone. Language ability
# ("English", "English fluency") and writing ability are here too: US resumes do not list them.
# Programming, Coding, Debugging, Research and Mathematics are not: a resume shows them through
# concrete skills (`skill_meets`).
SOFT_SKILLS = frozenset({
    "communication", "communication skills", "written communication", "verbal communication",
    "oral communication", "written and verbal communication", "verbal and written communication",
    "excellent communication skills", "strong communication skills", "written communication skills",
    "verbal communication skills", "communication and teamwork", "technical communication",
    "interpersonal skills", "interpersonal communication", "interpersonal", "problem solving", "problem solving skills",
    "complex problem solving", "analytical problem solving", "troubleshooting",
    "teamwork", "team work", "team player", "collaboration", "collaborative", "team collaboration",
    "cross functional collaboration", "multitasking", "multi tasking",
    "analytical skills", "analytical thinking", "analytical", "analytical ability", "analytical mindset",
    "analytical reasoning", "strong analytical skills", "critical thinking", "logical thinking", "decision making",
    "quantitative ability", "quantitative skills", "attention to detail", "detail oriented", "time management",
    "organization", "organizational skills", "organizational", "organization skills", "planning", "prioritization",
    "priority management", "task management", "leadership", "self motivated", "self starter", "self motivation",
    "self directed", "motivation", "adaptability", "flexibility", "creativity", "work ethic", "positive attitude",
    "accountability", "integrity", "confidentiality", "active learning", "learning strategies", "fast learner",
    "eagerness to learn", "willingness to learn", "curiosity", "ownership", "initiative",
    "active listening", "reading comprehension", "speaking", "writing", "monitoring", "judgment and decision making",
    "coordination", "social perceptiveness", "english language", "english", "english fluency", "english proficiency",
    "english communication", "science", "presentation skills", "public speaking",
    "customer focus", "professionalism", "reliability", "independence", "work independently", "independent work",
    "independently", "computer skills",
})

# Generic skills a resume shows through concrete ones: "Programming" is met by any programming
# language, "Mathematics" by a math subject, "Research" by any research skill, "SQL" by a SQL
# database, "Object-oriented programming" by an object-oriented language, "CAD" by a CAD program
# (`skill_meets`). A suite and its apps go both ways: "Microsoft Office" on a resume meets
# "Microsoft Excel", and Excel meets "Microsoft Office". Not transitive: Excel does not meet Word.
PROGRAMMING_LANGUAGES = (
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#", "C/C++", "Go", "Rust", "Ruby", "PHP", "Swift",
    "Kotlin", "Scala", "R", "MATLAB", "Perl", "Julia", "Haskell", "Fortran", "VBA", "Visual Basic", "Shell scripting",
    "Bash", "PowerShell", "Tcl", "Verilog", "SystemVerilog", "VHDL", "SAS", "Assembly", "Lua", "Dart", "Objective-C",
    "Elixir", "Clojure", "F#", "OCaml", "Lisp", "Groovy", "Solidity", "CUDA",
)
SCRIPTING_LANGUAGES = ("Python", "Perl", "Bash", "Shell scripting", "PowerShell", "Tcl", "Ruby", "JavaScript", "Lua",
                       "VBA", "Groovy")
MATH_SUBJECTS = ("Mathematics", "Linear algebra", "Calculus", "Multivariable calculus", "Statistics", "Probability",
                 "Probability theory", "Discrete mathematics", "Numerical methods", "Numerical analysis",
                 "Differential equations", "Stochastic calculus", "Mathematical modeling", "Real analysis",
                 "Optimization", "Applied mathematics")
ML_FRAMEWORKS = ("PyTorch", "TensorFlow", "Keras", "JAX", "scikit-learn")
OO_LANGUAGES = ("Java", "C++", "C#", "Python", "Kotlin", "Swift", "Ruby", "Scala", "Objective-C", "TypeScript")
SQL_DATABASES = ("SQL Server", "MySQL", "PostgreSQL", "Oracle Database", "SQLite", "T-SQL", "PL/SQL", "MariaDB")
CAD_TOOLS = ("AutoCAD", "AutoCAD Civil 3D", "SolidWorks", "Revit", "CATIA", "Bentley MicroStation",
             "DigiTools Roof CAD", "AppliCad Roof Wizard", "OTP ArmaCAD")
OFFICE_APPS = ("Microsoft Word", "Microsoft Excel", "Microsoft PowerPoint", "Microsoft Outlook")
ADOBE_APPS = ("Photoshop", "Illustrator", "InDesign", "After Effects", "Premiere Pro")
LMS_PRODUCTS = ("Blackboard Learn", "Blackboard software", "Schoology", "Google Classroom")
_OFFICE = ("Microsoft Office",)
_ADOBE = ("Adobe Creative Cloud",)
SHOWN_BY = {"Programming": PROGRAMMING_LANGUAGES, "Coding": PROGRAMMING_LANGUAGES, "Debugging": PROGRAMMING_LANGUAGES,
            "Scripting": SCRIPTING_LANGUAGES, "Mathematics": MATH_SUBJECTS,
            "Object-oriented programming": OO_LANGUAGES, "SQL": SQL_DATABASES,
            "Machine learning": (*ML_FRAMEWORKS, "Deep learning"),
            "Version control": ("Git", "GitHub", "GitLab", "Bitbucket", "Subversion", "Mercurial"),
            "Machine learning frameworks": ML_FRAMEWORKS, "Deep learning frameworks": ML_FRAMEWORKS,
            "Cloud platforms": ("AWS", "Azure", "Google Cloud"),
            # a suite and its core apps show each other: a resume's "Microsoft Office" is Word, Excel,
            # PowerPoint and Outlook (Access is not in every edition, so it shows the suite only)
            "Microsoft Office": (*OFFICE_APPS, "Microsoft Access"),
            **{app: _OFFICE for app in OFFICE_APPS},
            "Adobe Creative Cloud": ADOBE_APPS,
            **{app: _ADOBE for app in ADOBE_APPS},
            # Civil 3D runs on AutoCAD; every CAD program shows "CAD"
            "AutoCAD": ("AutoCAD Civil 3D",),
            "CAD": CAD_TOOLS,
            "Database software": (*SQL_DATABASES, "SQL", "Microsoft Access", "Relational databases"),
            "Learning management system LMS": (*LMS_PRODUCTS, "Course management system"),
            "Course management system": (*LMS_PRODUCTS, "Learning management system LMS"),
            # bookkeeping software shows O*NET's Economics and Accounting knowledge
            "Accounting": ("QuickBooks", "Sage 50 Accounting"),
            "Electronic medical record EMR": ("Epic", "MEDITECH", "eClinicalWorks", "Bizmatics PrognoCIS EMR",
                                              "Allscripts Professional EHR", "GE Healthcare Centricity EMR",
                                              "Automatic Data Processing AdvancedMD EHR", "ChartWare EMR"),
            # O*NET's Design knowledge is technical plans and drawings: CAD and circuit-board design show it
            "Design": ("CAD", *CAD_TOOLS, "PCB design", "Circuit design")}
SHOWN_BY_WORD = {"Research": "research"}  # any resume skill with this word ("Laboratory research", "UX research")


def _check_enum(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r} not in {allowed}")


def _tuplify(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Turn JSON lists back into tuples and drop unknown keys."""
    names = {f.name for f in fields(cls)}
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key not in names:
            continue
        if key == "must_skills" and isinstance(value, list):
            out[key] = tuple(tuple(g) if isinstance(g, list) else g for g in value)
        else:
            out[key] = tuple(value) if isinstance(value, list) else value
    return out


_SYNONYMS = _synonym_index()
_SHOWN_BY = {skill_canon(k): frozenset(skill_canon(x) for x in v) for k, v in SHOWN_BY.items()}
_SHOWN_BY_WORD = {skill_canon(k): re.compile(rf"\b{re.escape(w)}\b") for k, w in SHOWN_BY_WORD.items()}


def _shows(required_key: str, held_key: str) -> bool:
    if required_key in _SHOWN_BY:
        return held_key in _SHOWN_BY[required_key]
    word = _SHOWN_BY_WORD.get(required_key)
    return word is not None and word.search(held_key) is not None


def skill_meets(required: str, held: str) -> bool:
    """Does the resume skill `held` meet the required skill `required`? The same skill in any
    wording (skill_canon), or a concrete skill that shows a generic one: Python meets
    "Programming", Linear algebra meets "Mathematics", UX research meets "Research"."""
    r, h = skill_canon(required), skill_canon(held)
    return r == h or _shows(r, h)


def group_met(group: tuple[str, ...], have: set[str]) -> bool:
    """Is any alternative of a skill group met by `have` (a set of skill_canon keys)?"""
    for x in group:
        k = skill_canon(x)
        if k in have or ((k in _SHOWN_BY or k in _SHOWN_BY_WORD) and any(_shows(k, h) for h in have)):
            return True
    return False


@dataclass(frozen=True)
class RequirementCard:
    """Structured requirements pulled from one posting."""

    source_posting_id: str
    title: str
    field: str = ""  # occupation / field, free text
    onet_soc: str | None = None
    seniority: str = "mid"
    industry: str = ""
    degree_level: str = "none"  # minimum degree already held
    # "for students currently pursuing level X or higher"; none = no enrollment needed
    enrolled_in: str = "none"
    degree_fields: tuple[str, ...] = ()  # acceptable fields; empty = any
    # the posting also accepts "a related field" / equivalent / STEM: a field mismatch is soft
    related_ok: bool = False
    min_years: int = 0
    must_skills: tuple[SkillGroup, ...] = ()  # every group must be met
    nice_skills: tuple[str, ...] = ()
    license: str | None = None  # a credential the applicant must already hold
    locations: tuple[str, ...] = ()  # empty with onsite/hybrid = unconstrained
    work_mode: str = "onsite"
    sponsorship: str = "unstated"
    # US citizens or permanent residents only: citizenship, green card, "permanent"
    # work authorization, ITAR / export control or a security clearance
    citizens_only: bool = False
    e_verify: bool = False
    salary_min: int | None = None  # annual USD; None when unstated or not in USD
    salary_max: int | None = None
    employment_type: str = "full_time"
    # the posting says too little to judge the fit: no skill, degree, years or license on the card and a short
    # source text (rules.mark_thin sets it; extraction never does). Left out of to_dict when False, so every
    # card without it serializes as before.
    thin_posting: bool = False

    def __post_init__(self) -> None:
        _check_enum("seniority", self.seniority, SENIORITIES)
        _check_enum("degree_level", self.degree_level, DEGREE_LEVELS)
        _check_enum("enrolled_in", self.enrolled_in, DEGREE_LEVELS)
        for group in self.must_skills:
            if isinstance(group, str) or not group or not all(isinstance(x, str) and x for x in group):
                raise ValueError(f"must_skills groups must be non-empty tuples of names, got {group!r}")
        _check_enum("work_mode", self.work_mode, WORK_MODES)
        _check_enum("sponsorship", self.sponsorship, SPONSORSHIP)
        _check_enum("employment_type", self.employment_type, EMPLOYMENT_TYPES)
        if self.min_years < 0:
            raise ValueError("min_years must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d["thin_posting"]:
            del d["thin_posting"]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequirementCard:
        return cls(**_tuplify(cls, data))


@dataclass(frozen=True)
class CandidateCard:
    """A candidate's facts and preferences. Deliberately has no name."""

    field: str = ""
    degree_level: str = "bachelor"  # highest degree held or being pursued
    degree_field: str = ""
    # when degree_level was or will be finished; after CURRENT_YEAR = still enrolled
    graduation_year: int | None = None
    years: int = 0
    skills: tuple[str, ...] = ()
    licenses: tuple[str, ...] = ()
    current_location: str = ""
    willing_locations: tuple[str, ...] = ()
    remote_preference: str = "any"
    work_auth: str = "citizen_or_pr"
    # preferences
    salary_floor: int | None = None
    employment_types_wanted: tuple[str, ...] = ()  # empty = any
    industries_wanted: tuple[str, ...] = ()  # empty = any
    industries_avoided: tuple[str, ...] = ()
    roles_avoided: tuple[str, ...] = ()  # words matched against the title
    # NUISANCE: present for fairness swaps; label() never reads these.
    education_country: str = ""
    language: str = ""

    def __post_init__(self) -> None:
        _check_enum("degree_level", self.degree_level, DEGREE_LEVELS)
        _check_enum("remote_preference", self.remote_preference, REMOTE_PREFS)
        _check_enum("work_auth", self.work_auth, WORK_AUTH)
        for t in self.employment_types_wanted:
            _check_enum("employment_types_wanted", t, EMPLOYMENT_TYPES)
        if self.years < 0:
            raise ValueError("years must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateCard:
        return cls(**_tuplify(cls, data))


__all__ = [
    "BUCKETS",
    "CURRENT_YEAR",
    "CandidateCard",
    "DEGREE_LEVELS",
    "EMPLOYMENT_TYPES",
    "REMOTE_PREFS",
    "RequirementCard",
    "SENIORITIES",
    "SPONSORSHIP",
    "WORK_AUTH",
    "WORK_MODES",
    "SkillGroup",
    "ONET_NAMES",
    "SKILL_SYNONYMS",
    "SOFT_SKILLS",
    "group_met",
    "is_soft_skill",
    "norm",
    "skill_canon",
    "skill_key",
    "skill_meets",
]
