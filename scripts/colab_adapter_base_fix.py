"""Rewrite base_model_name_or_path in a Colab-trained adapter archive (amendment 6b) from the VM-local path the
trainer saved ("base") to the HF id it was loaded from (Qwen/Qwen3.5-4B), so merge_lora --expect-base accepts it.
Only adapter_config.json changes; adapter_model.safetensors and train_log.json are copied byte for byte (checked).

  colab_adapter_base_fix.py TAG      TAG in Bcolab, C
  -> data/models/main_4b_c3/TAG_colab_<stamp>.tar.gz rewritten (the original kept as .orig.tar.gz),
     data/models/c3_colab/TAG/archive.json and eval/colab_train_TAG.json updated with both sha256s
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen3.5-4B"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    tag = sys.argv[1]
    rec_p = ROOT / f"data/models/c3_colab/{tag}/archive.json"
    rec = json.loads(rec_p.read_text())
    arc = ROOT / rec["archive"]
    if sha(arc.read_bytes()) != rec["archive_sha256"]:
        raise SystemExit("archive differs from its record")
    orig = arc.with_name(arc.name.replace(".tar.gz", ".orig.tar.gz"))
    shutil.copy2(arc, orig)
    members = {}
    with tarfile.open(orig, "r:gz") as t:
        for m in t.getmembers():
            members[m.name] = t.extractfile(m).read()
    cfg = json.loads(members["adapter_config.json"])
    old = cfg["base_model_name_or_path"]
    cfg["base_model_name_or_path"] = BASE
    new_members = dict(members, **{"adapter_config.json": json.dumps(cfg, indent=2).encode()})
    with tarfile.open(arc, "w:gz") as t:
        for name, data in new_members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    with tarfile.open(arc, "r:gz") as t:
        back = {m.name: t.extractfile(m).read() for m in t.getmembers()}
    for k in members:
        if k != "adapter_config.json" and sha(back[k]) != sha(members[k]):
            raise SystemExit(f"{k} changed")
    rec |= {"archive_sha256": sha(arc.read_bytes()),
            "base_fix": {"what": f"adapter_config.json base_model_name_or_path {old!r} -> {BASE!r}; weights unchanged",
                         "original_archive": str(orig.relative_to(ROOT)), "original_sha256": sha(orig.read_bytes()),
                         "adapter_model_safetensors_sha256": sha(members["adapter_model.safetensors"])}}
    rec_p.write_text(json.dumps(rec, indent=1))
    (ROOT / f"eval/colab_train_{tag}.json").write_text(json.dumps(rec, indent=1))
    print(tag, old, "->", BASE, rec["archive_sha256"])


if __name__ == "__main__":
    main()
