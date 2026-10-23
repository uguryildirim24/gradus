import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_lora as ml  # noqa: E402

PUB = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "tie_word_embeddings": True,
       "text_config": {"hidden_size": 2560, "num_hidden_layers": 32, "layer_types": ["linear_attention"] * 3 +
                       ["full_attention"], "vocab_size": 248320, "eos_token_id": 248044}}


def _dirs(tmp_path, own: dict | None, declared="togethercomputer/Qwen3.5-4B"):
    adir, bdir = tmp_path / "adapter", tmp_path / "base"
    adir.mkdir(parents=True)
    bdir.mkdir(parents=True)
    (bdir / "config.json").write_text(json.dumps(PUB))
    if own is not None:
        (adir / "config.json").write_text(json.dumps(own))
    for d in (adir, bdir):
        (d / "tokenizer.json").write_text("{}")
    return adir, bdir, {"base_model_name_or_path": declared, "r": 16, "lora_alpha": 32}


def test_together_base_maps_to_qwen_when_the_config_fields_match(tmp_path):
    adir, bdir, cfg = _dirs(tmp_path, dict(PUB, use_cache=False))  # re-serialization fields are not compared
    rep = ml.check_base(adir, bdir, "Qwen/Qwen3.5-4B", cfg)
    assert rep["config_diff"] == {} and rep["same_tokenizer.json"] and rep["declared_base"] == "togethercomputer/Qwen3.5-4B"


def test_a_config_field_or_base_size_that_differs_stops_the_merge(tmp_path):
    own = dict(PUB, text_config=dict(PUB["text_config"], num_hidden_layers=24))
    adir, bdir, cfg = _dirs(tmp_path, own)
    with pytest.raises(SystemExit, match="num_hidden_layers"):
        ml.check_base(adir, bdir, "Qwen/Qwen3.5-4B", cfg)
    adir, bdir, cfg = _dirs(tmp_path / "b", PUB, "togethercomputer/Qwen3.5-0.8B")
    with pytest.raises(SystemExit, match="not 'Qwen/Qwen3.5-4B'"):
        ml.check_base(adir, bdir, "Qwen/Qwen3.5-4B", cfg)
    adir, bdir, cfg = _dirs(tmp_path / "c", None)
    with pytest.raises(SystemExit, match="no config.json"):
        ml.check_base(adir, bdir, "Qwen/Qwen3.5-4B", cfg)


def test_a_text_field_one_side_leaves_out_is_read_from_its_top_level(tmp_path):
    pub = dict(PUB, tie_word_embeddings=False)  # Qwen/Qwen3.5-9B: tie_word_embeddings at the top level only
    own = dict(pub, text_config=dict(PUB["text_config"], tie_word_embeddings=False))
    assert ml.config_diff(own, pub) == {}
    own["text_config"]["tie_word_embeddings"] = True
    assert ml.config_diff(own, pub) == {"text_config.tie_word_embeddings": ["True", "False"]}


def test_keep_float32_keeps_an_update_bfloat16_would_round_away():
    torch = pytest.importorskip("torch")
    W = torch.full((2, 2), 1.0, dtype=torch.bfloat16)  # bfloat16's step at 1.0 is 2**-7
    A, B = torch.full((1, 2), 1e-3), torch.ones((2, 1))
    assert torch.equal(ml.merge_weight(W, A, B, 1.0), W)
    kept = ml.merge_weight(W, A, B, 1.0, keep_float32=True)
    assert kept.dtype == torch.float32 and torch.allclose(kept, torch.full((2, 2), 1.001))
