"""The eval's label read after the modal/demo merge gives the same prediction as the read run b's reviewed numbers
came from (scripts/local_eval.py and src/matcher/evalstats.py at commit 1762a19). The 1762a19 read is copied
below line for line and fed the same hidden states and head rows as evalstats.LabelHead + evalstats.pick, on small
random tensors (no checkpoint is loaded), in float32, bfloat16 and float16, with forced exact ties. A second test
runs local_eval.py's own `score` command on a fake model and checks every row's pred against the 1762a19 read.
Needs torch, which the project venv leaves out; run with
    uv run --offline --with torch pytest tests/test_label_read_1762a19.py
"""
import argparse
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from matcher import evalstats as ev  # noqa: E402

LABELS_1762 = ("Keep", "Explore", "Skip")
TIE_ORDER_1762 = (2, 1, 0)
VOCAB, HIDDEN = 64, 16
KEEP, EXPLORE, SKIP = 10, 11, 12


def pick_1762a19(scores) -> int:
    """evalstats.pick at 1762a19, verbatim."""
    best = max(scores)
    for i in TIE_ORDER_1762:
        if scores[i] == best:
            return i
    raise ValueError("empty scores")


class Read1762a19:
    """local_eval.score's label read at 1762a19, the same statements in the same order."""

    def __init__(self, model, tok):
        head = model.get_output_embeddings()
        ids = {w: tok.encode(w, add_special_tokens=False) for w in LABELS_1762}
        assert all(len(v) == 1 for v in ids.values()), ids
        self.lid = [ids[w][0] for w in LABELS_1762]
        self.w32 = head.weight[self.lid].detach().float()  # (3, hidden): the label rows, in float32
        self.captured: list = []
        head.register_forward_hook(lambda m, inp, out: self.captured.append(inp[0].detach()))

    def clear(self):
        self.captured.clear()

    def scores(self):
        hidden = self.captured[-1][:, -1].float()  # (B, hidden), after the final norm
        return (hidden @ self.w32.T).cpu()  # float32 label scores

    def preds(self):
        return [LABELS_1762[pick_1762a19([float(x) for x in row])] for row in self.scores()]


class Tok:
    """Label words have one id each; any other text is one id per character."""
    words = {"Keep": KEEP, "Explore": EXPLORE, "Skip": SKIP}
    padding_side = "right"
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [self.words[text]] if text in self.words else [13 + ord(c) % (VOCAB - 13) for c in text]

    def apply_chat_template(self, msgs, add_generation_prompt, tokenize, enable_thinking):
        return "<u>" + msgs[0]["content"] + "<a>"

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + i % 26) for i in ids)

    def __call__(self, texts, add_special_tokens=False, return_tensors="pt", padding=True):
        seqs = [self.encode(t) for t in texts]
        n = max(len(s) for s in seqs)
        pad = [[self.pad_token_id] * (n - len(s)) for s in seqs]
        ids = [p + s if self.padding_side == "left" else s + p for p, s in zip(pad, seqs)]
        mask = [[0] * len(p) + [1] * len(s) if self.padding_side == "left" else [1] * len(s) + [0] * len(p)
                for p, s in zip(pad, seqs)]
        return _Enc(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))


class _Enc(dict):
    def to(self, dev):
        return self


class FakeLM(torch.nn.Module):
    """Random embedding, a nonlinear mix over positions, a final norm and a bias-free head, all in `dtype`."""

    def __init__(self, dtype, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.config = SimpleNamespace(tie_word_embeddings=False,
                                      get_text_config=lambda: SimpleNamespace(final_logit_softcapping=None))
        self.emb = torch.nn.Embedding(VOCAB, HIDDEN, dtype=dtype)
        self.mix = torch.nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=dtype)
        self.norm = torch.nn.LayerNorm(HIDDEN, dtype=dtype)
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False, dtype=dtype)
        with torch.no_grad():
            for p in (self.emb.weight, self.mix.weight, self.lm_head.weight):
                p.copy_(torch.randn(p.shape, generator=g) * 2)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
        x = self.emb(input_ids)
        x = torch.tanh(self.mix(x.cumsum(1)))
        h = self.norm(x)
        logits = self.lm_head(h[:, -logits_to_keep:] if logits_to_keep else h)
        return SimpleNamespace(logits=logits)


def merged_preds(head):
    return [ev.LABELS[ev.pick([float(x) for x in row])] for row in head.scores().cpu()]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_same_scores_and_preds_on_random_rows(dtype):
    rows = 0
    for seed in range(6):
        m = FakeLM(dtype, seed)
        tok = Tok()
        old = Read1762a19(m, tok)  # both hooks sit on the same head and see the same hidden state
        new = ev.LabelHead(m, tok)
        assert new.ids == old.lid and torch.equal(new.rows, old.w32)
        g = torch.Generator().manual_seed(100 + seed)
        for _ in range(20):
            ids = torch.randint(13, VOCAB, (8, int(torch.randint(1, 12, (1,), generator=g))), generator=g)
            old.clear()
            m(ids, logits_to_keep=1)
            assert torch.equal(new.scores().cpu(), old.scores())  # bit for bit
            assert merged_preds(new) == old.preds()
            rows += ids.shape[0]
    assert rows == 960


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tie", ["keep=explore", "explore=skip", "keep=skip", "all"])
def test_same_preds_on_exact_ties(dtype, tie):
    m = FakeLM(dtype, 7)
    with torch.no_grad():
        w = m.lm_head.weight
        if tie in ("keep=explore", "all"):
            w[EXPLORE] = w[KEEP]
        if tie in ("explore=skip", "all"):
            w[SKIP] = w[EXPLORE]
        if tie == "keep=skip":
            w[SKIP] = w[KEEP]
    tok = Tok()
    old, new = Read1762a19(m, tok), ev.LabelHead(m, tok)
    ids = torch.randint(13, VOCAB, (64, 6), generator=torch.Generator().manual_seed(3))
    m(ids, logits_to_keep=1)
    s = old.scores()
    tied = {"keep=explore": (0, 1), "explore=skip": (1, 2), "keep=skip": (0, 2), "all": (0, 2)}[tie]
    assert torch.equal(s[:, tied[0]], s[:, tied[1]])
    preds = old.preds()
    assert merged_preds(new) == preds
    # where the tied pair is the top score, the safer label of the pair wins in both reads
    top = s.max(1).values
    hits = [j for j in range(64) if s[j, tied[0]] == top[j]]
    assert hits and all(preds[j] == LABELS_1762[tied[1]] for j in hits)


def test_local_eval_score_command_matches_the_1762a19_read(tmp_path, monkeypatch):
    """local_eval.py score, as merged, run on a fake model: every row's pred equals the 1762a19 read's."""
    import local_eval

    models = {}

    def from_pretrained(name, dtype=None):
        models["m"] = FakeLM(dtype, 11).eval()
        return models["m"]

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoModelForCausalLM = SimpleNamespace(from_pretrained=from_pretrained)
    fake_tf.AutoTokenizer = SimpleNamespace(from_pretrained=lambda name: Tok())
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    buckets = ["For you", "Explore", "Excluded"]
    prompts = [f"resume {i} " + "x" * (i % 9) + f" posting {i * 7}" for i in range(30)]
    dev = tmp_path / "dev.jsonl"
    dev.write_text("".join(json.dumps({"pair_id": f"p{i}", "bucket": buckets[i % 3], "prompt": p}) + "\n"
                           for i, p in enumerate(prompts)))
    for batch in (1, 4):
        out = tmp_path / f"b{batch}.json"
        a = argparse.Namespace(model="fake", dev=str(dev), device="cpu", shard="0/2", batch=batch, dtype="bfloat16",
                               framing="chat", limit=0, gen=0, max_tokens=8, threads=1, out=str(out))
        local_eval.score(a)
        got = json.loads(out.read_text())["rows"]
        assert [r["pair_id"] for r in got] == [f"p{i}" for i in range(0, 30, 2)]

        # the 1762a19 read on the same model, the same left-padded batches
        m, tok = models["m"], Tok()
        tok.padding_side = "left"
        old = Read1762a19(m, tok)
        rows = [json.loads(line) for line in dev.read_text().splitlines()][0::2]
        want, want_scores = [], []
        with torch.inference_mode():
            for start in range(0, len(rows), batch):
                chunk = rows[start:start + batch]
                enc = tok([local_eval.frame(tok, r["prompt"], "chat") for r in chunk])
                old.clear()
                m(**enc, logits_to_keep=1)
                want += old.preds()
                want_scores += [[float(x) for x in row] for row in old.scores()]
        assert [r["pred"] for r in got] == want
        assert [r["scores"] for r in got] == want_scores
