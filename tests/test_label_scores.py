"""The demo reads a label the way the eval does (evalstats.LabelHead + evalstats.pick), checked on a tiny fake
model: no checkpoint is loaded. Needs torch, which the project venv leaves out; run with
    uv run --offline --with torch pytest tests/test_label_scores.py
"""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import demo_backend_torch as bt  # noqa: E402
import demo_common as dc  # noqa: E402
from matcher import evalstats as ev  # noqa: E402

KEEP, EXPLORE, SKIP, END = 300, 301, 302, 303
VOCAB = 304
# Float32 label scores 16.0, 16.03125, 9.0. A bfloat16 logit has 8 significant bits, so at magnitude 16-32
# it moves in steps of 0.125 and 16.03125 rounds to 16.0: Keep and Explore tie in the model's own logits.
H = torch.tensor([1.0, 1.0])
ROWS = {KEEP: [16.0, 0.0], EXPLORE: [16.0, 0.03125], SKIP: [9.0, 0.0], END: [24.0, 0.0]}


class Tok:
    """One token per character; label words and the end token have ids of their own."""
    words = {"Keep": KEEP, "Explore": EXPLORE, "Skip": SKIP}
    eos_token_id = END

    def apply_chat_template(self, msgs, add_generation_prompt, tokenize, enable_thinking):
        return "<U>" + msgs[0]["content"] + "<A>"

    def encode(self, text, add_special_tokens=False):
        return [self.words[text]] if text in self.words else [ord(c) % 256 for c in text]

    def convert_tokens_to_ids(self, t):
        return END

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i < 256)


class FakeLM(torch.nn.Module):
    """Every position's final hidden state is H; the head is a bias-free bfloat16 Linear (or float32)."""

    def __init__(self, dtype=torch.bfloat16, bias=False, tied=False):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tied)
        self.lm_head = torch.nn.Linear(2, VOCAB, bias=bias, dtype=dtype)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            for i, r in ROWS.items():
                self.lm_head.weight[i] = torch.tensor(r)
        self.h = H.to(dtype)
        self.calls = 0

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, past_key_values=None, use_cache=True, logits_to_keep=0):
        self.calls += 1
        hidden = self.h.expand(input_ids.shape[0], input_ids.shape[1], 2)
        logits = self.lm_head(hidden[:, -logits_to_keep:] if logits_to_keep else hidden)
        return SimpleNamespace(logits=logits, past_key_values=copy.deepcopy(past_key_values) or [0])


def test_bfloat16_logits_tie_where_float32_scores_do_not():
    m = FakeLM()
    logits = m(torch.tensor([[1, 2]]), logits_to_keep=1).logits[0, -1]
    assert logits.dtype == torch.bfloat16 and logits[KEEP] == logits[EXPLORE]  # the tie in the model's logits
    old = max(dc.LABELS, key=lambda w: logits[Tok.words[w]].float().item())  # the old read: max(), ties to Keep
    assert old == "Keep"
    head = ev.LabelHead(m, Tok())
    m(torch.tensor([[1, 2]]), logits_to_keep=1)
    s = head.scores()
    assert s.dtype == torch.float32 and s.shape == (1, 3)
    assert s[0].tolist() == [16.0, 16.03125, 9.0]
    assert ev.LABELS[ev.pick(s[0].tolist())] == "Explore"


def test_demo_scorer_reads_the_eval_label_through_the_prefix_cache():
    bt._attach(torch, Tok(), FakeLM(), "cpu")
    out = bt._score_chunk("RESUME abc\nJOB POSTING:\n", [{"id": "a", "suffix": "alpha\n\nAnswer:"},
                                                        {"id": "b", "suffix": "beta\n\nAnswer:"}], True, 8)
    for r in out:
        assert r["scores"] == {"Keep": 16.0, "Explore": 16.03125, "Skip": 9.0}  # float32, not rounded
        assert r["label"] == "Explore"
        assert r["p"] == dc.label_probs(r["scores"])
        assert r["reason"] == "" and r["cached_tokens"] > 0  # END is the top next token, so the reason stops at once


def test_demo_scorer_p_keep_uses_the_models_temperature():
    bt._attach(torch, Tok(), FakeLM(), "cpu", temperature=1.3199)  # d3's dev-fitted temperature
    (r,) = bt._score_chunk("P:", [{"id": "a", "suffix": "x"}], False, 0)
    assert r["label"] == "Explore" and r["p"] == dc.label_probs(r["scores"], 1.3199) != dc.label_probs(r["scores"])
    bt._attach(torch, Tok(), FakeLM(), "cpu")


def test_demo_scorer_float32_model_gives_the_same_label():
    bt._attach(torch, Tok(), FakeLM(dtype=torch.float32), "cpu")
    (r,) = bt._score_chunk("P:", [{"id": "a", "suffix": "x"}], False, 0)
    assert r["label"] == "Explore" and r["reason"] is None


def test_exact_float32_tie_goes_to_the_safer_label():
    m = FakeLM(dtype=torch.float32)
    with torch.no_grad():
        m.lm_head.weight[SKIP] = torch.tensor(ROWS[EXPLORE])
    bt._attach(torch, Tok(), m, "cpu")
    (r,) = bt._score_chunk("P:", [{"id": "a", "suffix": "x"}], False, 0)
    assert r["scores"]["Explore"] == r["scores"]["Skip"] and r["label"] == "Skip"


def test_label_head_refuses_a_head_bias_and_reads_tied_from_the_config():
    with pytest.raises(AssertionError, match="bias"):
        ev.LabelHead(FakeLM(bias=True), Tok())
    assert ev.LabelHead(FakeLM(tied=True), Tok()).tied
    assert ev.LabelHead(FakeLM(), Tok()).ids == [KEEP, EXPLORE, SKIP]


def test_eval_and_demo_share_one_label_read():
    import local_eval

    assert local_eval.LabelHead is ev.LabelHead is bt.LabelHead
    assert local_eval.pick is ev.pick is dc.pick
