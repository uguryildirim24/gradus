"""The Mac scorer (scripts/demo_score.py) reads the label scores the way the eval does, on tiny mlx arrays: no model is
loaded; the label is run B's decider on them. Also the parity rule both demo checks use (demo_lib.label_parity)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "demo"))

import demo_common as dc  # noqa: E402
import demo_lib as lib  # noqa: E402
import demo_score as ds  # noqa: E402

IDS = [3, 4, 5]  # Keep, Explore, Skip
ROWS = [[16.0, 0.0], [16.0, 0.03125], [9.0, 0.0]]


def inner(tied: bool, dims: int = 2):
    w = mx.zeros((8, dims))
    for i, r in zip(IDS, ROWS):
        w[i, :2] = mx.array(r)
    w = w.astype(mx.bfloat16)
    emb, head = nn.Embedding(8, dims), nn.Linear(dims, 8, bias=False)
    emb.weight, head.weight = w, w
    return SimpleNamespace(args=SimpleNamespace(tie_word_embeddings=tied), model=SimpleNamespace(embed_tokens=emb),
                           lm_head=head)


@pytest.mark.parametrize("tied", [True, False])
def test_mlx_label_read_is_float32_where_bfloat16_logits_tie(tied):
    m = inner(tied)
    hidden = mx.ones((1, 1, 2), dtype=mx.bfloat16)  # the decoder's output at the last position
    logits = m.model.embed_tokens.as_linear(hidden) if tied else m.lm_head(hidden)
    assert logits.dtype == mx.bfloat16 and logits[0, 0, IDS[0]].item() == logits[0, 0, IDS[1]].item()  # tie
    s = ds.label_scores(hidden, ds.head_label_rows(m, IDS))
    assert s == {"Keep": 16.0, "Explore": 16.03125, "Skip": 9.0}
    assert dc.decider_label(s) == "Keep"  # s = P(Keep) + 0.5 P(Explore) = 0.746 on the float32 scores


def test_mlx_label_rows_refuse_a_quantized_head():
    m = inner(True, dims=64)
    m.model.embed_tokens = nn.QuantizedEmbedding.from_embedding(m.model.embed_tokens)
    with pytest.raises(AssertionError, match="quantized"):
        ds.head_label_rows(m, IDS)


def test_label_parity_counts_near_cuts_apart():
    # the eval's label is the decider on its scores: a Keep (s 0.90), an Explore just under T_HI (s 0.583), an Explore
    pairs = [("a", "Keep", {"Keep": 3.0, "Explore": 1.0, "Skip": 0.0}, [3.0, 1.0, 0.0]),
             ("b", "Keep", {"Keep": 0.52, "Explore": 0.0, "Skip": 0.0}, [0.47, 0.0, 0.0]),  # near-cut
             ("c", "Skip", {"Keep": 0.0, "Explore": 0.0, "Skip": 5.0}, [0.0, 5.0, 1.0])]  # real flip
    r = lib.label_parity(pairs)
    assert r["agree"] == 1 and [t[0] for t in r["near_cuts"]] == ["b"] and [f[0] for f in r["flips"]] == ["c"]
    assert r["max_score_diff"] == 5.0 and not r["ok"]
    assert lib.label_parity(pairs[:2])["ok"] and not lib.label_parity([])["ok"]
