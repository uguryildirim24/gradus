"""Merge a PEFT LoRA adapter into its base HF checkpoint with torch on CPU. Runs on oci-pi, never the Mac.

Usage: merge_lora.py ADAPTER_DIR OUT_DIR --base BASE_DIR [--expect-base HF_ID] [--keep-float32]

ADAPTER_DIR holds adapter_config.json + adapter_model.safetensors (a pickle .bin is refused).
BASE_DIR is a local snapshot of the base model (e.g. `hf download Qwen/Qwen3.5-0.8B --local-dir BASE_DIR`).
W' = W + scale * B @ A with scale = alpha / r (alpha / sqrt(r) under rslora), computed in
float32 and cast back to the base dtype (--keep-float32: the merged weights stay float32, so an update smaller
than bfloat16's step at W is not rounded away; the other weights keep the base dtype). OUT_DIR gets the merged weights in the base's HF
layout plus the base's config, tokenizer and chat template, so transformers loads it like the
base model. Every LoRA A/B pair must land on a base weight of the same shape, or nothing is written.
Needs torch and safetensors only.

--expect-base HF_ID checks the base before merging (as demo/modal_app.py prepare does). The adapter's
adapter_config.json names the base Adaption trained on (togethercomputer/Qwen3.5-4B: Together's copy of the public
checkpoint, not on the Hub). That id, with togethercomputer/ read as Qwen/, must be HF_ID, and the architecture
fields of the adapter's own config.json must equal BASE_DIR/config.json's (CONFIG_KEYS, TEXT_KEYS); any difference
stops the merge. OUT_DIR/MERGED_FROM.json records the check and whether tokenizer.json and the chat template match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

COPY = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "special_tokens_map.json", "preprocessor_config.json", "video_preprocessor_config.json")


CONFIG_KEYS = ("architectures", "model_type", "tie_word_embeddings")
TEXT_KEYS = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
             "head_dim", "vocab_size", "layer_types", "linear_num_key_heads", "linear_num_value_heads",
             "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim", "rope_parameters",
             "rms_norm_eps", "full_attention_interval", "attn_output_gate", "tie_word_embeddings", "eos_token_id")


def public_base(repo_id: str) -> str:
    """togethercomputer/X -> Qwen/X (Adaption trains on Together's copies of the public Qwen checkpoints)."""
    return "Qwen/" + repo_id.split("/", 1)[1] if repo_id.startswith("togethercomputer/") else repo_id


def config_diff(own: dict, pub: dict) -> dict[str, list[str]]:
    """Architecture fields where the adapter's config.json and the base's differ (text_config fields too). A
    text_config field one side leaves out is read from that config's top level (Qwen/Qwen3.5-9B keeps
    tie_word_embeddings only there; Together's re-save writes it into text_config as well)."""
    ot, pt = own.get("text_config", own), pub.get("text_config", pub)
    diff = {k: [str(own.get(k)), str(pub.get(k))] for k in CONFIG_KEYS if own.get(k) != pub.get(k)}
    for k in TEXT_KEYS:
        o, p = ot.get(k, own.get(k)), pt.get(k, pub.get(k))
        if o != p:
            diff[f"text_config.{k}"] = [str(o), str(p)]
    return diff


def check_base(adir: Path, base_dir: Path, expect: str, cfg: dict) -> dict:
    declared = cfg.get("base_model_name_or_path") or ""
    report = {"declared_base": declared, "base_used": expect}
    if public_base(declared) != expect:
        raise SystemExit(f"adapter was trained on {declared!r}, not {expect!r}")
    if (adir / "config.json").exists():
        diff = config_diff(json.loads((adir / "config.json").read_text()), json.loads((base_dir / "config.json").read_text()))
        report["config_fields_compared"] = len(CONFIG_KEYS) + len(TEXT_KEYS)
        report["config_diff"] = diff
        if diff:
            raise SystemExit(f"{expect} config differs from the adapter's config.json: {diff}")
    elif public_base(declared) != declared:
        raise SystemExit(f"adapter names {declared!r} and has no config.json to compare with {expect}")
    for f in ("tokenizer.json", "chat_template.jinja"):
        if (adir / f).exists() and (base_dir / f).exists():
            report[f"same_{f}"] = hashlib.sha256((adir / f).read_bytes()).hexdigest() == \
                hashlib.sha256((base_dir / f).read_bytes()).hexdigest()
    print("base check:", json.dumps(report))
    return report


def base_key(peft_key: str, base_keys: set[str]) -> str | None:
    mod = peft_key.removeprefix("base_model.model.").split(".lora_")[0]
    for cand in (mod, mod.replace("model.layers.", "model.language_model.layers."),
                 "model.language_model." + mod.removeprefix("model."), "model." + mod):
        if cand + ".weight" in base_keys:
            return cand + ".weight"
    return None


def merge_weight(W, A, B, scale: float, keep_float32: bool = False):
    """W + scale * B @ A in float32, cast back to W's dtype unless keep_float32."""
    merged = W.float() + scale * (B.float() @ A.float())
    return merged if keep_float32 else merged.to(W.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--base", required=True)
    ap.add_argument("--expect-base", help="HF id the base must be (togethercomputer/ in the adapter read as Qwen/)")
    ap.add_argument("--keep-float32", action="store_true", help="save the merged weights in float32, not the base dtype")
    a = ap.parse_args()
    import torch  # here, so tests import the base check without torch
    from safetensors.torch import load_file, save_file

    adir, out, base_dir = Path(a.adapter_dir), Path(a.out_dir), Path(a.base)
    cfg = json.loads((adir / "adapter_config.json").read_text())
    if not (adir / "adapter_model.safetensors").exists():
        raise SystemExit("no adapter_model.safetensors (refusing pickle weights)")
    r, alpha = cfg["r"], cfg.get("lora_alpha", cfg["r"])
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    print(f"adapter r={r} alpha={alpha} scale={scale} targets={cfg.get('target_modules')} "
          f"base={cfg.get('base_model_name_or_path')} modules_to_save={cfg.get('modules_to_save')}")
    report = check_base(adir, base_dir, a.expect_base, cfg) if a.expect_base else {}
    lora = load_file(str(adir / "adapter_model.safetensors"))

    weights: dict[str, torch.Tensor] = {}
    for f in sorted(base_dir.glob("*.safetensors")):
        weights.update(load_file(str(f)))
    keys = set(weights)

    merged, unmatched = 0, []
    for k in sorted(lora):
        if ".lora_A." not in k:
            continue
        kb = k.replace(".lora_A.", ".lora_B.")
        tgt = base_key(k, keys)
        if tgt is None or kb not in lora:
            unmatched.append(k)
            continue
        A, B, W = lora[k], lora[kb], weights[tgt]
        if (B.shape[0], A.shape[1]) != tuple(W.shape):
            raise SystemExit(f"shape mismatch {tgt}: base {tuple(W.shape)} vs delta {(B.shape[0], A.shape[1])}")
        weights[tgt] = merge_weight(W, A, B, scale, a.keep_float32)
        merged += 1
    other = [k for k in lora if ".lora_A." not in k and ".lora_B." not in k]
    print(f"merged {merged} LoRA pairs; unmatched {len(unmatched)} {unmatched[:5]}; non-LoRA tensors {len(other)} {other[:5]}")
    if unmatched or other:
        raise SystemExit("adapter has tensors this script does not handle")
    n_b = sum(".lora_B." in k for k in lora)
    if not merged or n_b != merged:
        raise SystemExit(f"merged {merged} pairs but the adapter has {n_b} lora_B tensors")

    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in weights.items()}, str(out / "model.safetensors"), metadata={"format": "pt"})
    for name in COPY:
        if (base_dir / name).exists():
            shutil.copy(base_dir / name, out / name)
    if report:
        (out / "MERGED_FROM.json").write_text(json.dumps(report | {"lora_pairs_merged": merged,
                                                                  "merged_dtype": "float32" if a.keep_float32 else "base"}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
