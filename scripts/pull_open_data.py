"""Download the open training sources into data/raw/open/ (idempotent).

Run: uv run python scripts/pull_open_data.py
"""
import io
import zipfile
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "open"
HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
ONET_URL = "https://www.onetcenter.org/dl_files/database/db_31_0_text.zip"
# O*NET 31.0 renamed several files. Mapping to the names the plan uses:
#   Skills -> Essential Skills; Technology Skills -> Software Skills;
#   Alternate Titles -> Job Titles; Education Training and Experience ->
#   Education + Training and Experience (plus their category files).
ONET_KEEP = [
    "Occupation Data", "Job Zones", "Job Zone Reference", "Knowledge",
    "Essential Skills", "Software Skills", "Job Titles",
    "Education", "Education Categories",
    "Training and Experience", "Training and Experience Categories",
]


def fetch(url: str, dest: Path | None = None) -> bytes | None:
    if dest is not None and dest.exists():
        print("have", dest)
        return None
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        data = b"".join(r.iter_bytes())
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print("wrote", dest, len(data))
        return None
    return data


def main() -> None:
    fetch(HF.format(repo="NextGig-Rocks/global-job-postings-multi-ats",
                    path="nextgig_jobs_2026-06.parquet"),
          OUT / "nextgig" / "nextgig_jobs_2026-06.parquet")
    fetch(HF.format(repo="lang-uk/recruitment-dataset-job-descriptions-english",
                    path="data/train-00000-of-00001.parquet"),
          OUT / "djinni" / "train-00000-of-00001.parquet")
    onet = OUT / "onet"
    if not (onet / "Occupation Data.txt").exists():
        data = fetch(ONET_URL)
        onet.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                stem = Path(name).stem
                if stem in ONET_KEEP and name.endswith(".txt"):
                    (onet / f"{stem}.txt").write_bytes(z.read(name))
                    print("wrote", onet / f"{stem}.txt")
    else:
        print("have onet")


if __name__ == "__main__":
    main()
