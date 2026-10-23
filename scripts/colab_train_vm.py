"""PEFT LoRA training of one matcher model on a Colab GPU with B's recipe (eval/protocol_amendment_6b.json
colab_training). Runs on the Colab VM only (scripts/colab_train.py uploads it); never on the Mac.

  colab_train_vm.py --train TRAIN.csv --eval EVAL.csv --out DIR [--base Qwen/Qwen3.5-4B] [--epochs 3]
                    [--steps-per-epoch 21] [--lr 2e-4] [--warmup 4] [--seed 20261024] [--max-rows N]

Recipe: LoRA r 16, alpha 32, dropout 0 on the language model's q/k/v/o/gate/up/down projections (128 modules, checked);
base in bfloat16; AdamW (0.9, 0.999, eps 1e-8, weight decay 0); max grad norm 2.0; lr 2e-4 with `warmup` linear
warmup steps then cosine (half a cycle) down to 0.1 x lr at the last step; epochs x steps-per-epoch optimizer steps,
each over ceil(N / steps-per-epoch) whole rows taken one at a time (gradient accumulation), the loss the mean over
every completion token of the step's rows. Row order: torch.randperm per epoch from a Generator seeded `seed`.

Framing: the prompt through the tokenizer's chat template as local_eval.frame scores it (user turn,
add_generation_prompt, enable_thinking False); the completion + <|im_end|> tokenized on its own and appended; only
those tokens are trained. The first completion token must be the label word's single token.

Writes DIR/adapter_model.safetensors, DIR/adapter_config.json, DIR/train_log.json (every step's loss, grad norm, lr,
rows and tokens; eval loss after each epoch; GPU, versions, wall time) and touches DIR/TRAIN_DONE. Progress lines go
to stdout. A thread touches /content/heartbeat every minute while the process lives.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import platform
import threading
import time


def beat() -> None:
    while True:
        pathlib.Path("/content/heartbeat").touch()
        time.sleep(60)


def read_csv(p: str) -> list[tuple[str, str]]:
    csv.field_size_limit(10 ** 9)
    with open(p, newline="") as f:
        return [(r["prompt"], r["completion"]) for r in csv.DictReader(f)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--base-id", default="Qwen/Qwen3.5-4B", help="the HF id written into adapter_config.json")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps-per-epoch", type=int, default=21)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--max-rows", type=int, default=0, help="timing probe: stop after this many training rows")
    a = ap.parse_args()
    threading.Thread(target=beat, daemon=True).start()
    t_start = time.time()

    import torch
    import transformers
    import peft
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    try:
        from transformers.utils.import_utils import is_flash_linear_attention_available
        fla = bool(is_flash_linear_attention_available())
    except Exception:  # noqa: BLE001
        fla = None
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).cuda()
    model.config.use_cache = False
    target = r".*\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=target,
                     task_type="CAUSAL_LM", use_rslora=False, use_dora=False)
    model = get_peft_model(model, cfg)
    n_lora = sum(1 for n, _ in model.named_modules() if n.endswith(".lora_A"))
    print(f"model {type(model.base_model.model).__name__}, LoRA modules {n_lora}, fla {fla}", flush=True)
    if n_lora != 128:
        raise SystemExit(f"expected 128 LoRA modules (B's adapter), got {n_lora}")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()

    def frame(prompt: str) -> list[int]:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
        return tok(text, add_special_tokens=False)["input_ids"]

    label_tok = {w: tok.encode(w, add_special_tokens=False) for w in ("Keep", "Explore", "Skip")}
    assert all(len(v) == 1 for v in label_tok.values()), label_tok

    def encode(rows: list[tuple[str, str]]) -> list[tuple[list[int], list[int]]]:
        enc = []
        for p, c in rows:
            ids_c = tok(c + "<|im_end|>", add_special_tokens=False)["input_ids"]
            word = c.split()[0]
            if ids_c[0] != label_tok[word][0]:
                raise SystemExit(f"completion {c[:40]!r} does not start with the label token")
            enc.append((frame(p), ids_c))
        return enc

    train = encode(read_csv(a.train))
    ev = encode(read_csv(a.eval))
    n = len(train)
    per_step = math.ceil(n / a.steps_per_epoch)
    total = a.epochs * a.steps_per_epoch
    lens = sorted(len(p) + len(c) for p, c in train)
    print(f"train rows {n}, rows/step {per_step}, steps {total}, tokens median {lens[n // 2]} max {lens[-1]}, "
          f"eval rows {len(ev)}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    def lr_lambda(step: int) -> float:
        if step < a.warmup:
            return step / max(1, a.warmup)
        prog = (step - a.warmup) / max(1, total - a.warmup)
        return a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def row_loss(p: list[int], c: list[int]) -> torch.Tensor:
        ids = torch.tensor([p + c], device="cuda")
        logits = model(input_ids=ids, logits_to_keep=len(c) + 1).logits[0, :-1].float()
        return torch.nn.functional.cross_entropy(logits, torch.tensor(c, device="cuda"), reduction="sum")

    def eval_loss() -> float:
        model.eval()
        tot, ntok = 0.0, 0
        with torch.no_grad():
            for p, c in ev:
                tot += float(row_loss(p, c))
                ntok += len(c)
        model.train()
        return tot / ntok

    g = torch.Generator().manual_seed(a.seed)
    log: dict = {"steps": [], "eval": []}
    seen = 0
    step = 0
    t_train = time.time()
    for epoch in range(a.epochs):
        perm = torch.randperm(n, generator=g).tolist()
        for s in range(a.steps_per_epoch):
            idx = perm[s * per_step:(s + 1) * per_step]
            if not idx:
                continue
            ntok = sum(len(train[i][1]) for i in idx)
            tot = 0.0
            t0 = time.time()
            for i in idx:
                loss = row_loss(*train[i])
                (loss / ntok).backward()
                tot += float(loss)
                seen += 1
                if a.max_rows and seen >= a.max_rows:
                    break
            gn = float(torch.nn.utils.clip_grad_norm_(params, 2.0))
            lr_now = opt.param_groups[0]["lr"]
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": round(epoch + (s + 1) / a.steps_per_epoch, 4), "loss": tot / ntok,
                   "grad_norm": gn, "learning_rate": lr_now, "rows": len(idx), "target_tokens": ntok,
                   "seconds": round(time.time() - t0, 1)}
            log["steps"].append(rec)
            print(json.dumps(rec), flush=True)
            if a.max_rows and seen >= a.max_rows:
                print(f"PROBE rows {seen} in {time.time() - t_train:.0f} s", flush=True)
                (out / "PROBE_DONE").touch()
                return
        el = eval_loss()
        log["eval"].append({"step": step, "epoch": epoch + 1, "eval_loss": el})
        print(json.dumps({"step": step, "eval_loss": el}), flush=True)

    model.save_pretrained(str(out))
    cfg_p = out / "adapter_config.json"
    c = json.loads(cfg_p.read_text())
    c["base_model_name_or_path"] = a.base_id
    cfg_p.write_text(json.dumps(c, indent=2))
    log |= {
        "recipe": {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": target, "lora_modules": n_lora,
                   "dtype": "bfloat16", "optimizer": "torch AdamW (0.9, 0.999), eps 1e-8, weight decay 0",
                   "max_grad_norm": 2.0, "lr": a.lr, "warmup_steps": a.warmup, "scheduler": "cosine, half cycle, "
                   f"min_lr_ratio {a.min_lr_ratio}", "epochs": a.epochs, "steps_per_epoch": a.steps_per_epoch,
                   "steps": total, "rows_per_step": per_step, "seed": a.seed, "gradient_checkpointing": True,
                   "loss": "completion + <|im_end|> tokens only, token mean over the step",
                   "framing": "chat template, user turn, add_generation_prompt, enable_thinking False"},
        "train_rows": n, "eval_rows": len(ev), "base": a.base,
        "final_train_loss": log["steps"][-1]["loss"], "final_eval_loss": log["eval"][-1]["eval_loss"],
        "gpu": torch.cuda.get_device_name(0), "gpu_memory_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                     "python": platform.python_version(), "flash_linear_attention": fla},
        "wall_seconds": {"total": round(time.time() - t_start), "training": round(time.time() - t_train)},
    }
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    (out / "TRAIN_DONE").touch()
    print("TRAIN_DONE", json.dumps({k: log[k] for k in ("final_train_loss", "final_eval_loss", "gpu", "wall_seconds")}),
          flush=True)


if __name__ == "__main__":
    main()
