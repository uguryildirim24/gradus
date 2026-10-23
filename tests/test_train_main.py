"""train_main: its own state, the CSV md5 binding, the run settings and pins, the retrying wait loop and the cap."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_main as tm  # noqa: E402

REC = {"model": tm.MODEL, "hyperparams": {
    "training_type": "full", "n_epochs": 2.0, "learning_rate": 1e-4, "batch_size": "max", "lora_r": 16.0,
    "lora_alpha": 32.0, "lora_dropout": 0.0, "lora_trainable_modules": "all-linear", "train_on_inputs": None,
    "dpo_beta": None, "rpo_alpha": None, "simpo_gamma": None, "dpo_normalize_logratios_by_length": None}}


def test_a_dataset_from_another_csv_is_refused(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("prompt,completion\nx,Keep — y.\n")
    b.write_text("prompt,completion\nx,Keep — y.\nz,Skip — w.\n")
    s = {"dataset": {"dataset_id": "d1", "csv_md5": tm.md5_of(a), "csv_rows": tm.csv_rows(a), "model": tm.MODEL}}
    assert tm.dataset_problem(s, a) is None
    why = tm.dataset_problem(s, b)
    assert why and "d1" in why and "md5" in why and "2 rows" in why
    assert "not Qwen/Qwen3.5-4B" in tm.dataset_problem({"dataset": s["dataset"] | {"model": "Qwen/Qwen3.5-0.8B"}}, a)
    assert tm.dataset_problem({}, b) is None  # nothing uploaded yet


def test_adaption_row_count_must_match_the_csv():
    d = {"dataset_id": "d1", "csv_rows": 6891}
    assert tm.row_count_problem({"dataset": d | {"status": {"status": "succeeded", "row_count": 6891}}}) is None
    assert "6880" in tm.row_count_problem({"dataset": d | {"status": {"status": "succeeded", "row_count": 6880}}})
    assert "processing" in tm.row_count_problem({"dataset": d | {"status": {"status": "processing", "row_count": 6891}}})


def test_settings_take_the_recommendation_and_pin_lora_inputs_iterations_and_augmentation():
    assert tm.STATE.name == "main_4b.json" and tm.MODEL == "Qwen/Qwen3.5-4B"
    with pytest.raises(SystemExit):
        tm.run_settings({})
    with pytest.raises(SystemExit):  # a recommendation for another model is not used
        tm.run_settings({"recommended": REC | {"model": "Qwen/Qwen3.5-0.8B"}})
    st = tm.run_settings({"recommended": REC})
    assert st["max_iterations"] == 1 and st["target_win_rate"] == 0.8
    assert st["augmentation_domain_rows"] == 0 and st["augmentation_general_rows"] == 0
    hp = st["hyperparams"]
    assert hp["train_on_inputs"] is False and hp["training_type"] == "lora"
    assert hp["n_epochs"] == 2.0 and hp["learning_rate"] == 1e-4 and hp["lora_r"] == 16.0 and hp["batch_size"] == "max"
    assert not any(k.startswith(("dpo", "rpo", "simpo")) for k in hp)  # nulls and fields create does not take
    assert set(hp) <= tm.CREATE_HP_KEYS
    assert tm.run_settings({"recommended": REC}, 3)["hyperparams"]["n_epochs"] == 3
    # a recommendation saying train_on_inputs True is still overridden
    rec_true = {"model": tm.MODEL, "hyperparams": REC["hyperparams"] | {"train_on_inputs": True}}
    assert tm.run_settings({"recommended": rec_true})["hyperparams"]["train_on_inputs"] is False


def test_settings_outside_adaptions_limits_are_refused():
    ok = REC["hyperparams"] | {"warmup_ratio": 0.03}
    assert tm.limit_problems(ok) == []
    assert tm.limit_problems(ok | {"n_epochs": 1.0, "lora_r": 64.0, "lora_alpha": 64.0}) == []  # run a sent floats
    for bad, word in [({"batch_size": 16}, "batch_size"), ({"batch_size": 8}, "batch_size"),
                      ({"n_epochs": 6}, "n_epochs"), ({"n_epochs": 2.5}, "whole"), ({"learning_rate": 1e-3}, "learning_rate"),
                      ({"lora_r": 128, "lora_alpha": 256}, "lora_r"), ({"lora_r": 24, "lora_alpha": 48}, "lora_r"),
                      ({"lora_alpha": 48.0}, "lora_alpha"), ({"warmup_ratio": 1.5}, "warmup_ratio")]:
        why = tm.limit_problems(ok | bad)
        assert len(why) == 1 and word in why[0], (bad, why)
    with pytest.raises(SystemExit, match="batch_size"):
        tm.run_settings({"recommended": {"model": tm.MODEL, "hyperparams": ok | {"batch_size": 16}}})
    with pytest.raises(SystemExit, match="n_epochs"):
        tm.run_settings({"recommended": REC}, 6)


def test_retry_sleeps_with_backoff_and_returns(capsys):
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 5:
            raise ConnectionError("network down")
        return 42

    assert tm.retry(flaky, "run get", sleeps.append, start=10, cap=40) == 42
    assert sleeps == [10, 20, 40, 40]
    out = capsys.readouterr().out
    assert out.count("retry run get") == 4 and "ConnectionError" in out
    assert out.splitlines()[0][:4].isdigit() and "Z retry" in out  # UTC timestamp first


class FakeAS:
    """autoscientist + datasets: get/credits fail now and then; the run spends 1 credit a poll."""

    def __init__(self, statuses, fail_get=(), fail_credits=(), fail_cancel=0, spend=1.0):
        self.statuses, self.fail_get, self.fail_credits = list(statuses), set(fail_get), set(fail_credits)
        self.fail_cancel, self.spend = fail_cancel, spend
        self.n_get = self.n_credits = 0
        self.balance, self.cancelled = 300.0, 0
        self.autoscientist, self.datasets = self, self

    def get(self, rid):
        self.n_get += 1
        if self.n_get in self.fail_get:
            raise ConnectionError("get dropped")
        st = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        self.balance -= self.spend
        return SimpleNamespace(status=st, iterations_completed=1.0 if st == "succeeded" else 0.0, max_iterations=1.0,
                               best_win_rate=0.8 if st == "succeeded" else None,
                               download_available=st == "succeeded", created_at=datetime(2026, 10, 20, tzinfo=timezone.utc),
                               completed_at=None, model_dump_json=lambda: '{"status": "%s"}' % st)

    def invent(self, **kw):
        self.n_credits += 1
        if self.n_credits in self.fail_credits:
            raise TimeoutError("credits timed out")
        return SimpleNamespace(available_credits=self.balance)

    def cancel(self, rid):
        if self.fail_cancel:
            self.fail_cancel -= 1
            raise ConnectionError("cancel dropped")
        self.cancelled += 1
        self.statuses = ["cancelled"]
        return "cancelled"


def state(cap=100):
    return {"runs": {"a": {"run_id": "r1", "credits_before": 300.0, "max_credits": cap}}}


def test_wait_loop_survives_network_errors_and_downloads_at_the_end(capsys):
    c = FakeAS(["running", "running", "succeeded"], fail_get={1, 3}, fail_credits={2})
    s, sleeps, archived = state(), [], []
    run = tm.wait_loop(c, s, "a", 100, sleep=sleeps.append, save_fn=lambda s: None,
                       archive=lambda c, s, n, stem, f: archived.append(stem))
    assert run.status == "succeeded" and c.cancelled == 0
    assert archived == ["r1"]  # the finished run's archive is fetched once
    r = s["runs"]["a"]
    assert len(r["credit_trace"]) == 3 and r["credits_spent"] == 300.0 - r["credits_after"]
    out = capsys.readouterr().out
    assert out.count("retry run get") == 2 and out.count("retry credits read") == 1
    assert out.count(" running iter 0/1") == 2 and " succeeded iter 1/1" in out
    assert tm.RETRY_START in sleeps and tm.POLL_START in sleeps


def test_wait_loop_cancels_over_the_cap_even_when_cancel_fails_once():
    c = FakeAS(["running"], fail_cancel=1, spend=40.0)
    s, sleeps = state(cap=100), []
    run = tm.wait_loop(c, s, "a", 100, sleep=sleeps.append, save_fn=lambda s: None, archive=lambda *a: None)
    assert run.status == "cancelled" and c.cancelled == 1 and s["runs"]["a"]["cancelled_for_budget"]
    assert s["runs"]["a"]["credits_spent"] > 100


def test_the_cap():
    assert not tm.over_cap(300, 290, 10) and tm.over_cap(300, 289, 10)


STATE_B = {"dataset": {"dataset_id": "66d2ad3d", "model": tm.MODEL}, "recommended": {"model": tm.MODEL, "hyperparams": {
    "training_type": "lora", "n_epochs": 1.0, "learning_rate": 1e-05, "batch_size": "max", "lora_r": 16.0,
    "lora_alpha": 32.0, "lora_dropout": 0.0, "lora_trainable_modules": "all-linear", "lr_scheduler_type": "cosine",
    "min_lr_ratio": 0.1, "scheduler_num_cycles": 0.5, "warmup_ratio": 0.1, "max_grad_norm": 2.0, "weight_decay": 0.0,
    "train_on_inputs": False, "dpo_beta": None, "rpo_alpha": None, "simpo_gamma": None,
    "dpo_normalize_logratios_by_length": None}}}  # the 4B recommendation as stored


class FakeCreate:
    def __init__(self):
        self.sent, self.autoscientist = [], self

    def create(self, **kw):
        self.sent.append(kw)
        return SimpleNamespace(id="run-b-1")


def test_run_b_create_sends_its_overrides_pins_and_own_key():
    c = FakeCreate()
    run, kw = tm.create_run(c, STATE_B, "b")
    assert run.id == "run-b-1" and c.sent == [kw]
    sent = c.sent[0]
    assert sent["dataset_id"] == "66d2ad3d" and sent["model"] == "Qwen/Qwen3.5-4B"
    assert sent["idempotency_key"] == "main-b-66d2ad3d" != tm.create_kwargs(STATE_B, "a")["idempotency_key"]
    assert sent["column_mapping"] == tm.TRAIN_MAPPING and sent["training_method"] == "instruction"
    assert sent["max_iterations"] == 1 and sent["target_win_rate"] == 0.8
    assert sent["augmentation_domain_rows"] == 0 and sent["augmentation_general_rows"] == 0
    hp = sent["hyperparams"]
    assert hp == {"training_type": "lora", "train_on_inputs": False, "learning_rate": 1e-4, "n_epochs": 2,
                  "warmup_ratio": 0.03, "lr_scheduler_type": "cosine", "lora_r": 16, "lora_alpha": 32,
                  "lora_trainable_modules": "all-linear", "lora_dropout": 0.0, "batch_size": "max",
                  # the rest of the recommendation, untouched
                  "min_lr_ratio": 0.1, "scheduler_num_cycles": 0.5, "max_grad_norm": 2.0, "weight_decay": 0.0}
    assert type(hp["n_epochs"]) is int and tm.step_estimate(hp) == 42
    # run a still gets the recommendation as it was
    hp_a = tm.create_kwargs(STATE_B, "a")["hyperparams"]
    assert hp_a["learning_rate"] == 1e-05 and hp_a["n_epochs"] == 1.0 and hp_a["warmup_ratio"] == 0.1


def test_no_run_override_can_undo_a_pin(monkeypatch):
    monkeypatch.setitem(tm.RUN_OVERRIDES, "b", tm.RUN_OVERRIDES["b"] | {"train_on_inputs": True, "training_type": "full"})
    hp = tm.create_kwargs(STATE_B, "b")["hyperparams"]
    assert hp["train_on_inputs"] is False and hp["training_type"] == "lora"
    assert tm.create_kwargs(STATE_B, "b", 5)["hyperparams"]["n_epochs"] == 5  # --epochs goes over the override


def test_wait_loop_runs_on_run_b():
    c = FakeAS(["running", "succeeded"])
    s = {"runs": {"a": {"run_id": "r1", "credits_before": 300.0, "max_credits": 100, "archive": "a.tar.gz"},
                  "b": {"run_id": "r2", "credits_before": 300.0, "max_credits": 100}}}
    archived = []
    run = tm.wait_loop(c, s, "b", 100, sleep=lambda d: None, save_fn=lambda s: None,
                       archive=lambda c, s, n, stem, f: archived.append((n, stem)))
    assert run.status == "succeeded" and archived == [("b", "r2")]
    assert "credit_trace" not in s["runs"]["a"] and s["runs"]["b"]["credits_spent"] == 2.0


B_HP = {"training_type": "lora", "train_on_inputs": False, "learning_rate": 1e-4, "n_epochs": 2, "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine", "lora_r": 16, "lora_alpha": 32, "lora_trainable_modules": "all-linear",
        "lora_dropout": 0.0, "batch_size": "max", "min_lr_ratio": 0.1, "scheduler_num_cycles": 0.5,
        "max_grad_norm": 2.0, "weight_decay": 0.0}  # run b's create, as sent on 2026-10-20


def test_run_c_create_sends_its_overrides_pins_and_own_key_and_leaves_a_and_b_alone():
    assert "c" in tm.RUNS
    c = FakeCreate()
    run, kw = tm.create_run(c, STATE_B, "c")
    assert c.sent == [kw]
    sent = c.sent[0]
    assert sent["idempotency_key"] == "main-c-66d2ad3d"
    keys = {tm.create_kwargs(STATE_B, n)["idempotency_key"] for n in ("a", "b", "c")}
    assert keys == {"main-a-66d2ad3d", "main-b-66d2ad3d", "main-c-66d2ad3d"}
    assert sent["dataset_id"] == "66d2ad3d" and sent["model"] == "Qwen/Qwen3.5-4B"
    assert sent["column_mapping"] == tm.TRAIN_MAPPING and sent["training_method"] == "instruction"
    assert sent["max_iterations"] == 1 and sent["target_win_rate"] == 0.8
    assert sent["augmentation_domain_rows"] == 0 and sent["augmentation_general_rows"] == 0
    hp = sent["hyperparams"]
    assert hp == B_HP | {"learning_rate": 2e-4, "n_epochs": 4, "warmup_ratio": 0.05}
    assert hp["training_type"] == "lora" and hp["train_on_inputs"] is False  # the pins
    assert type(hp["n_epochs"]) is int and tm.step_estimate(hp) == 84 and tm.limit_problems(hp) == []
    # runs a and b are what they were
    assert tm.create_kwargs(STATE_B, "b")["hyperparams"] == B_HP
    hp_a = tm.create_kwargs(STATE_B, "a")["hyperparams"]
    assert hp_a["learning_rate"] == 1e-05 and hp_a["n_epochs"] == 1.0 and hp_a["warmup_ratio"] == 0.1
    assert tm.RUN_OVERRIDES["a"] == {}


def test_no_run_c_override_can_undo_a_pin(monkeypatch):
    monkeypatch.setitem(tm.RUN_OVERRIDES, "c", tm.RUN_OVERRIDES["c"] | {"train_on_inputs": True, "training_type": "full"})
    hp = tm.create_kwargs(STATE_B, "c")["hyperparams"]
    assert hp["train_on_inputs"] is False and hp["training_type"] == "lora"


def test_run_b_keeps_main_4b_json_and_run_c_has_its_own_file():
    root = Path(tm.__file__).resolve().parents[1]
    assert tm.STATE == root / "data/adaption/main_4b.json"
    assert tm.state_path("a") == tm.state_path("b") == tm.state_path(None) == tm.STATE  # b's live wait's file
    assert tm.state_path("c") == root / "data/adaption/main_4b_c.json" != tm.STATE


class FakeLive:
    """Enough of the client for `c create` and `c wait` through main()."""

    def __init__(self):
        self.autoscientist = self.datasets = self
        self.balance, self.sent = 319.0, []

    def create(self, **kw):
        self.sent.append(kw)
        return SimpleNamespace(id="run-c-1", model_dump_json=lambda: '{"id": "run-c-1"}')

    def get(self, rid):
        return SimpleNamespace(status="succeeded", iterations_completed=1.0, max_iterations=1.0, best_win_rate=0.9,
                               download_available=False, created_at=datetime(2026, 10, 20, tzinfo=timezone.utc),
                               completed_at=None, model_dump_json=lambda: '{"status": "succeeded"}')

    def invent(self, **kw):
        return SimpleNamespace(available_credits=self.balance)


def test_two_runs_state_files_never_touch_each_other(tmp_path, monkeypatch):
    csv_path = tmp_path / "smoke.csv"
    csv_path.write_text("prompt,completion\nx,Keep — y.\n")
    main_f, c_f = tmp_path / "main_4b.json", tmp_path / "main_4b_c.json"
    monkeypatch.setattr(tm, "STATE", main_f)
    monkeypatch.setitem(tm.STATES, "c", c_f)
    live = FakeLive()
    monkeypatch.setattr(tm, "client", lambda: live)
    monkeypatch.setattr(tm.time, "sleep", lambda d: None)
    dataset = STATE_B["dataset"] | {"csv_md5": tm.md5_of(csv_path), "csv_rows": 1,
                                    "status": {"status": "succeeded", "row_count": 1}}
    b_state = {"dataset": dataset, "recommended": STATE_B["recommended"],
               "runs": {"a": {"run_id": "ra"}, "b": {"run_id": "rb", "credits_before": 319.0, "max_credits": 100}}}
    main_f.write_text(json.dumps(b_state, indent=1))
    main_bytes = main_f.read_bytes()

    def run(*words):
        monkeypatch.setattr(sys, "argv", ["train_main.py", *words, "--csv", str(csv_path)])
        tm.main()

    with pytest.raises(SystemExit, match="seed"):  # c never falls back to main_4b.json
        run("config", "c")
    run("c", "seed")
    seeded = json.loads(c_f.read_text())
    assert seeded["dataset"] == dataset and seeded["recommended"] == b_state["recommended"] and "runs" not in seeded
    with pytest.raises(FileExistsError):  # a second seed never writes over c's file
        run("c", "seed")
    run("c", "create", "--max-credits", "100")
    run("c", "wait")
    assert main_f.read_bytes() == main_bytes  # c's seed, create and wait never wrote main_4b.json
    rc = json.loads(c_f.read_text())["runs"]["c"]
    assert rc["run_id"] == "run-c-1" and rc["credits_before"] == 319.0 and rc["max_credits"] == 100
    assert rc["credits_spent"] == 0.0 and live.sent[0]["idempotency_key"] == "main-c-66d2ad3d"
    assert (tmp_path / "main_4b_run_c.txt").read_text().count("run-c-1") == 1
    with pytest.raises(SystemExit, match="already created"):
        run("c", "create", "--max-credits", "100")
    # b's live wait rewriting main_4b.json from its old copy leaves c's file as it was
    c_bytes = c_f.read_bytes()
    tm.save(b_state | {"runs": b_state["runs"] | {"b": b_state["runs"]["b"] | {"last": {"status": "running"}}}}, main_f)
    assert c_f.read_bytes() == c_bytes
    # and b's own commands still read and write main_4b.json only
    run("b", "wait")
    assert json.loads(main_f.read_text())["runs"]["b"]["credits_spent"] == 0.0
    assert c_f.read_bytes() == c_bytes and set(json.loads(main_f.read_text())["runs"]) == {"a", "b"}


# Run e: AutoScientist's own search. A supplied hyperparam is pinned for the run (Adaption's docs), so e sends the
# two task pins and nothing else, with the loop settings of its own.

def test_run_e_sends_only_the_pins_and_its_own_loop_settings_and_key():
    assert "e" in tm.RUNS and "e" in tm.SEARCH
    c = FakeCreate()
    _, kw = tm.create_run(c, STATE_B, "e")
    sent = c.sent[0]
    assert sent == kw and sent["idempotency_key"] == "main-e-66d2ad3d"
    assert sent["hyperparams"] == {"training_type": "lora", "train_on_inputs": False}  # no recipe: the search is open
    c_win = 0.8579  # run c's one iteration: a c-like pass must not stop e's loop
    assert sent["max_iterations"] == 5 and sent["target_win_rate"] == 0.95 > c_win
    assert sent["augmentation_domain_rows"] == 0 and sent["augmentation_general_rows"] == 0
    assert sent["dataset_id"] == "66d2ad3d" and sent["model"] == "Qwen/Qwen3.5-4B"
    assert sent["column_mapping"] == tm.TRAIN_MAPPING and sent["training_method"] == "instruction"
    assert tm.step_estimate(sent["hyperparams"]) is None
    keys = {tm.create_kwargs(STATE_B, n)["idempotency_key"] for n in tm.RUNS}
    assert len(keys) == len(tm.RUNS) == 12
    # a, b and c are what they were: one iteration at target 0.8, their recipes unchanged
    for n in ("a", "b", "c"):
        st = tm.create_kwargs(STATE_B, n)
        assert st["max_iterations"] == 1 and st["target_win_rate"] == 0.8
    assert tm.create_kwargs(STATE_B, "b")["hyperparams"] == B_HP
    assert tm.create_kwargs(STATE_B, "c")["hyperparams"] == B_HP | {"learning_rate": 2e-4, "n_epochs": 4,
                                                                    "warmup_ratio": 0.05}


def test_run_e_needs_no_recommendation_and_refuses_epochs():
    assert tm.run_settings({}, name="e")["hyperparams"] == tm.PINS
    with pytest.raises(SystemExit, match="pin"):
        tm.run_settings(STATE_B, 3, "e")
    with pytest.raises(SystemExit, match="pin"):
        tm.create_kwargs(STATE_B, "e", 3)


def test_loop_settings_outside_adaptions_limits_are_refused(monkeypatch):
    assert tm.loop_problems({"max_iterations": 5, "target_win_rate": 1.0}) == []
    assert tm.loop_problems({"max_iterations": 1, "target_win_rate": 0.51}) == []
    for bad, word in [({"max_iterations": 6}, "max_iterations"), ({"max_iterations": 0}, "max_iterations"),
                      ({"max_iterations": 2.5}, "max_iterations"), ({"target_win_rate": 0.5}, "target_win_rate"),
                      ({"target_win_rate": 1.01}, "target_win_rate")]:
        why = tm.loop_problems({"max_iterations": 3, "target_win_rate": 0.9} | bad)
        assert len(why) == 1 and word in why[0], (bad, why)
    monkeypatch.setitem(tm.LOOPS, "e", {"max_iterations": 6, "target_win_rate": 0.95})
    with pytest.raises(SystemExit, match="max_iterations"):
        tm.create_kwargs(STATE_B, "e")


def test_no_override_reaches_run_e(monkeypatch):
    monkeypatch.setitem(tm.RUN_OVERRIDES, "e", {"learning_rate": 1e-3, "train_on_inputs": True})
    assert tm.create_kwargs(STATE_B, "e")["hyperparams"] == tm.PINS


def test_run_e_has_its_own_state_file():
    root = Path(tm.__file__).resolve().parents[1]
    assert tm.state_path("e") == root / "data/adaption/main_4b_e.json"
    assert len({tm.state_path("e"), tm.state_path("c"), tm.STATE}) == 3
    assert tm.state_path("a") == tm.state_path("b") == tm.STATE


HP1 = {"training_type": "lora", "n_epochs": 1.0, "learning_rate": 1e-5, "train_on_inputs": False}
HP2 = HP1 | {"n_epochs": 3.0, "learning_rate": 1e-4}


class FakeLoop:
    """A 3-iteration run: the best appears after iteration 1, improves after 2, stays after 3."""

    def __init__(self, polls, spend=0.0):
        self.polls, self.spend = list(polls), spend
        self.autoscientist = self.datasets = self
        self.balance, self.cancelled = 284.0, 0

    def get(self, rid):
        st, done, win, hp, dl = self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]
        self.balance -= self.spend
        body = {"status": st, "iterations_completed": float(done), "max_iterations": 3.0, "best_win_rate": win,
                "best_hyperparams": hp, "download_available": dl}
        return SimpleNamespace(**body, created_at=datetime(2026, 10, 20, tzinfo=timezone.utc), completed_at=None,
                               model_dump_json=lambda: json.dumps(body))

    def invent(self, **kw):
        return SimpleNamespace(available_credits=self.balance)

    def cancel(self, rid):
        self.cancelled += 1
        last = self.polls[-1]
        self.polls = [("cancelled",) + last[1:]]
        return "cancelled"


def e_state(cap=100):
    return {"runs": {"e": {"run_id": "re", "credits_before": 284.0, "max_credits": cap}}}


def fake_archive(saved):
    def archive(c, s, name, stem, save_fn):  # what save_archive records, without a download
        r = s["runs"][name]
        r["archive"] = stem
        r["archive_best"] = tm.best_key(r["last"])
        saved.append(stem)
    return archive


def test_wait_loop_logs_each_iteration_and_keeps_each_new_best_and_the_final_best(capsys):
    c = FakeLoop([("running", 0, None, None, False),
                  ("running", 1, 0.55, HP1, True),
                  ("running", 1, 0.55, HP1, True),   # same best: not downloaded twice
                  ("running", 2, 0.83, HP2, True),   # new best mid-run: saved
                  ("running", 3, 0.83, HP2, True),   # iteration 3 scored at or under it
                  ("succeeded", 3, 0.83, HP2, True)])
    s, saved = e_state(), []
    run = tm.wait_loop(c, s, "e", 100, sleep=lambda d: None, save_fn=lambda s: None, archive=fake_archive(saved))
    assert run.status == "succeeded" and c.cancelled == 0
    assert saved == ["re_iter1", "re_iter2"]  # the final best is already on disk: no third download
    its = s["runs"]["e"]["iterations"]
    assert [i["iteration"] for i in its] == [1, 2, 3] and [i["new_best"] for i in its] == [True, True, False]
    assert its[1]["best"] == {"best_win_rate": 0.83, "best_hyperparams": HP2}
    out = capsys.readouterr().out
    assert "iteration 1 done: best win 0.55 NEW BEST" in out and "learning_rate=1e-05" in out
    assert "iteration 2 done: best win 0.83 NEW BEST" in out and "n_epochs=3.0" in out
    assert "iteration 3 done: best win 0.83 (best unchanged" in out


def test_wait_loop_downloads_the_final_best_when_it_changed_after_the_last_save():
    c = FakeLoop([("running", 1, 0.55, HP1, True), ("running", 2, 0.55, HP1, False),
                  ("succeeded", 3, 0.9, HP2, True)])
    s, saved = e_state(), []
    tm.wait_loop(c, s, "e", 100, sleep=lambda d: None, save_fn=lambda s: None, archive=fake_archive(saved))
    assert saved == ["re_iter1", "re"] and s["runs"]["e"]["archive_best"]["best_win_rate"] == 0.9


def test_wait_loop_cap_cancels_a_loop_and_keeps_the_best_saved_before():
    c = FakeLoop([("running", 1, 0.6, HP1, True), ("running", 1, 0.6, HP1, True)], spend=60.0)
    s, saved = e_state(cap=100), []
    run = tm.wait_loop(c, s, "e", 100, sleep=lambda d: None, save_fn=lambda s: None, archive=fake_archive(saved))
    assert run.status == "cancelled" and c.cancelled == 1 and s["runs"]["e"]["cancelled_for_budget"]
    assert saved == ["re_iter1"]  # iteration 1's best was saved before the cancel; not fetched again after it


def test_wait_loop_after_a_restart_does_not_relog_iterations(capsys):
    s = e_state()
    s["runs"]["e"]["iterations"] = [{"iteration": 1, "best": {"best_win_rate": 0.55, "best_hyperparams": HP1},
                                     "new_best": True}]
    c = FakeLoop([("succeeded", 1, 0.55, HP1, True)])
    tm.wait_loop(c, s, "e", 100, sleep=lambda d: None, save_fn=lambda s: None, archive=lambda *a: None)
    assert len(s["runs"]["e"]["iterations"]) == 1 and "iteration 1 done" not in capsys.readouterr().out


def test_run_e_seed_create_wait_never_touch_main_or_c(tmp_path, monkeypatch):
    csv_path = tmp_path / "smoke.csv"
    csv_path.write_text("prompt,completion\nx,Keep — y.\n")
    main_f, c_f, e_f = tmp_path / "main_4b.json", tmp_path / "main_4b_c.json", tmp_path / "main_4b_e.json"
    monkeypatch.setattr(tm, "STATE", main_f)
    monkeypatch.setitem(tm.STATES, "c", c_f)
    monkeypatch.setitem(tm.STATES, "e", e_f)
    live = FakeLive()
    monkeypatch.setattr(tm, "client", lambda: live)
    monkeypatch.setattr(tm.time, "sleep", lambda d: None)
    dataset = STATE_B["dataset"] | {"csv_md5": tm.md5_of(csv_path), "csv_rows": 1,
                                    "status": {"status": "succeeded", "row_count": 1}}
    main_f.write_text(json.dumps({"dataset": dataset, "recommended": STATE_B["recommended"],
                                  "runs": {"a": {"run_id": "ra"}, "b": {"run_id": "rb"}}}, indent=1))
    c_f.write_text(json.dumps({"dataset": dataset, "runs": {"c": {"run_id": "rc"}}}, indent=1))
    main_bytes, c_bytes = main_f.read_bytes(), c_f.read_bytes()

    def run(*words):
        monkeypatch.setattr(sys, "argv", ["train_main.py", *words, "--csv", str(csv_path)])
        tm.main()

    with pytest.raises(SystemExit, match="seed"):  # e never falls back to main_4b.json
        run("e", "create", "--max-credits", "100")
    run("e", "seed")
    assert json.loads(e_f.read_text())["dataset"] == dataset and "runs" not in json.loads(e_f.read_text())
    with pytest.raises(FileExistsError):
        run("e", "seed")
    with pytest.raises(SystemExit, match="pin"):
        run("e", "create", "--max-credits", "100", "--epochs", "3")
    run("e", "create", "--max-credits", "100")
    run("e", "wait")
    re_ = json.loads(e_f.read_text())["runs"]["e"]
    assert re_["run_id"] == "run-c-1" and re_["max_credits"] == 100 and re_["settings"]["max_iterations"] == 5
    assert re_["settings"]["hyperparams"] == tm.PINS and live.sent[0]["idempotency_key"] == "main-e-66d2ad3d"
    assert (tmp_path / "main_4b_run_e.txt").read_text().count("run e run-c-1") == 1
    with pytest.raises(SystemExit, match="already created"):
        run("e", "create", "--max-credits", "100")
    assert len(live.sent) == 1
    assert main_f.read_bytes() == main_bytes and c_f.read_bytes() == c_bytes


# Run f: run c's recipe on Qwen/Qwen3.5-9B, on a dataset of its own (one training run per dataset at a time).

C_SENT = B_HP | {"learning_rate": 2e-4, "n_epochs": 4, "warmup_ratio": 0.05}  # run c's create, as sent on 2026-10-20
STATE_F = {"dataset": {"dataset_id": "9b9b9b9b", "model": "Qwen/Qwen3.5-9B"}}


def test_run_f_sends_run_cs_recipe_on_the_9b_and_nothing_else_changes():
    assert "f" in tm.RUNS and tm.model_of("f") == "Qwen/Qwen3.5-9B"
    assert all(tm.model_of(n) == tm.MODEL == "Qwen/Qwen3.5-4B" for n in ("a", "b", "c", "e", None))
    c = FakeCreate()
    _, kw = tm.create_run(c, STATE_F, "f")
    sent = c.sent[0]
    assert sent == kw and sent["model"] == "Qwen/Qwen3.5-9B" and sent["dataset_id"] == "9b9b9b9b"
    assert sent["idempotency_key"] == "main-f-9b9b9b9b"
    assert sent["hyperparams"] == C_SENT == tm.RECIPES["f"]  # c's 15 fields exactly, no recommendation under them
    assert sent["hyperparams"] == tm.create_kwargs(STATE_B, "c")["hyperparams"]  # what c sends today, too
    assert sent["max_iterations"] == 1 and sent["target_win_rate"] == 0.8
    assert sent["augmentation_domain_rows"] == 0 and sent["augmentation_general_rows"] == 0
    assert sent["column_mapping"] == tm.TRAIN_MAPPING and sent["training_method"] == "instruction"
    assert tm.limit_problems(sent["hyperparams"], "Qwen/Qwen3.5-9B") == []
    assert tm.step_range(sent["hyperparams"], "f") == (84, 168) and tm.step_range(C_SENT, "c") == (84, 84)
    # a recommendation in f's file is never sent
    rec = {"model": "Qwen/Qwen3.5-9B", "hyperparams": C_SENT | {"learning_rate": 1e-5, "weight_decay": 0.1}}
    assert tm.create_kwargs(STATE_F | {"recommended": rec}, "f")["hyperparams"] == C_SENT
    # a, b, c and e are what they were
    assert tm.create_kwargs(STATE_B, "b")["hyperparams"] == B_HP
    assert tm.create_kwargs(STATE_B, "c")["hyperparams"] == C_SENT
    assert tm.create_kwargs(STATE_B, "e")["hyperparams"] == tm.PINS
    assert {tm.create_kwargs(STATE_B, n)["model"] for n in ("a", "b", "c", "e")} == {"Qwen/Qwen3.5-4B"}


def test_run_fs_pins_hold_and_its_limits_are_checked(monkeypatch):
    monkeypatch.setitem(tm.RECIPES, "f", C_SENT | {"train_on_inputs": True, "training_type": "full"})
    hp = tm.create_kwargs(STATE_F, "f")["hyperparams"]
    assert hp["train_on_inputs"] is False and hp["training_type"] == "lora"
    for bad, word in [({"batch_size": 8}, "batch_size"), ({"lora_r": 128, "lora_alpha": 256}, "lora_r"),
                      ({"learning_rate": 1e-3}, "learning_rate"), ({"n_epochs": 6}, "n_epochs")]:
        monkeypatch.setitem(tm.RECIPES, "f", C_SENT | bad)
        with pytest.raises(SystemExit, match=word):
            tm.create_kwargs(STATE_F, "f")
    monkeypatch.setitem(tm.RECIPES, "f", C_SENT)
    with pytest.raises(SystemExit, match="n_epochs"):
        tm.create_kwargs(STATE_F, "f", 6)
    assert "Qwen/Qwen3.5-9B" in tm.limit_problems(C_SENT | {"batch_size": 8}, "Qwen/Qwen3.5-9B")[0]


def test_run_f_has_its_own_state_file_record_and_archive_dir():
    root = Path(tm.__file__).resolve().parents[1]
    assert tm.state_path("f") == root / "data/adaption/main_9b_f.json"
    assert len({tm.state_path(n) for n in tm.RUNS}) == 11  # a and b share main_4b.json; c, e, f, d, d3, h, h2b, h2n, A, B have their own
    assert tm.archive_dir("f") == root / "data/models/main_9b" != tm.archive_dir("c") == tm.MODELS
    assert tm.tag_of("f") == "main_9b" and tm.tag_of("c") == tm.tag_of("e") == tm.tag_of("b") == "main_4b"


def test_a_4b_dataset_is_refused_for_f_and_a_9b_one_for_the_4b_runs(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("prompt,completion\nx,Keep — y.\n")
    d = {"dataset_id": "d1", "csv_md5": tm.md5_of(a), "csv_rows": 1}
    assert tm.dataset_problem({"dataset": d | {"model": "Qwen/Qwen3.5-9B"}}, a, "Qwen/Qwen3.5-9B") is None
    assert "not Qwen/Qwen3.5-9B" in tm.dataset_problem({"dataset": d | {"model": "Qwen/Qwen3.5-4B"}}, a, "Qwen/Qwen3.5-9B")
    assert "not Qwen/Qwen3.5-4B" in tm.dataset_problem({"dataset": d | {"model": "Qwen/Qwen3.5-9B"}}, a)


class FakeUpload(FakeLive):
    """FakeLive plus a raw upload that comes back with Adaption's row count."""

    def __init__(self, rows):
        super().__init__()
        self.rows, self.uploads, self.recs = rows, [], []

    def upload_file(self, path, **kw):
        self.uploads.append((path, kw))
        return SimpleNamespace(dataset_id=f"ds-9b-{len(self.uploads)}", status="pending")

    def wait_for_completion(self, did, timeout=None):
        return SimpleNamespace(model_dump_json=lambda: json.dumps({"dataset_id": did, "status": "succeeded",
                                                                   "row_count": self.rows}))

    def recommend_hyperparams(self, **kw):
        self.recs.append(kw)
        body = {"model": kw["model"], "hyperparams": C_SENT | {"learning_rate": 1e-5}}
        return SimpleNamespace(model_dump_json=lambda: json.dumps(body))


def f_files(tmp_path, monkeypatch):
    csv_path = tmp_path / "smoke.csv"
    csv_path.write_text("prompt,completion\nx,Keep — y.\nz,Skip — w.\n")
    files = {n: tmp_path / n for n in ("main_4b.json", "main_4b_c.json", "main_4b_e.json", "main_9b_f.json")}
    monkeypatch.setattr(tm, "STATE", files["main_4b.json"])
    monkeypatch.setitem(tm.STATES, "c", files["main_4b_c.json"])
    monkeypatch.setitem(tm.STATES, "e", files["main_4b_e.json"])
    monkeypatch.setitem(tm.STATES, "f", files["main_9b_f.json"])
    monkeypatch.setattr(tm, "MODELS", tmp_path / "models/main_4b")
    monkeypatch.setattr(tm.time, "sleep", lambda d: None)
    ds = {"dataset_id": "66d2ad3d", "model": tm.MODEL, "csv_md5": tm.md5_of(csv_path), "csv_rows": 2,
          "status": {"status": "succeeded", "row_count": 2}}
    files["main_4b.json"].write_text(json.dumps({"dataset": ds, "recommended": STATE_B["recommended"],
                                                 "runs": {"a": {"run_id": "ra"}, "b": {"run_id": "rb"}}}, indent=1))
    files["main_4b_c.json"].write_text(json.dumps({"dataset": ds, "runs": {"c": {
        "run_id": "1bbd8313", "settings": {"max_iterations": 1, "target_win_rate": 0.8, "hyperparams": C_SENT},
        "last": {"model": tm.MODEL, "best_win_rate": 0.8579}}}}, indent=1))
    files["main_4b_e.json"].write_text(json.dumps({"dataset": ds, "runs": {"e": {"run_id": "re"}}}, indent=1))

    def run(*words):
        monkeypatch.setattr(sys, "argv", ["train_main.py", *words, "--csv", str(csv_path)])
        tm.main()
    return files, csv_path, run


def test_run_f_seed_upload_hparams_create_wait_never_touch_the_4b_files(tmp_path, monkeypatch):
    files, csv_path, run = f_files(tmp_path, monkeypatch)
    before = {n: files[n].read_bytes() for n in ("main_4b.json", "main_4b_c.json", "main_4b_e.json")}
    live = FakeUpload(rows=2)
    monkeypatch.setattr(tm, "client", lambda: live)
    f_path = files["main_9b_f.json"]
    with pytest.raises(SystemExit, match="seed"):  # f never falls back to main_4b.json's dataset
        run("f", "upload")
    run("f", "seed")
    seeded = json.loads(f_path.read_text())
    assert "dataset" not in seeded and "recommended" not in seeded  # nothing of 66d2ad3d or the 4B's recommendation
    assert seeded["recipe_from"]["run_id"] == "1bbd8313" and seeded["recipe_from"]["hyperparams"] == C_SENT
    with pytest.raises(FileExistsError):
        run("f", "seed")
    with pytest.raises(SystemExit, match="upload first"):
        run("f", "create", "--max-credits", "100")
    run("f", "upload")
    _, kw = live.uploads[0]
    assert kw["name"] == f"main-train-{tm.md5_of(csv_path)[:8]}-9b" and kw["processing_mode"] == "raw"
    d = json.loads(f_path.read_text())["dataset"]
    assert d["dataset_id"] == "ds-9b-1" and d["model"] == "Qwen/Qwen3.5-9B" and d["csv_rows"] == 2
    assert d["credits_before_upload"] == d["credits_after_upload"] == 319.0
    run("f", "upload")  # again: already uploaded, no second dataset
    assert len(live.uploads) == 1
    run("f", "hparams")
    assert live.recs[0]["model"] == "Qwen/Qwen3.5-9B" and live.recs[0]["dataset_id"] == "ds-9b-1"
    run("config", "f")
    run("f", "create", "--max-credits", "100")
    sent = live.sent[0]
    assert sent["model"] == "Qwen/Qwen3.5-9B" and sent["dataset_id"] == "ds-9b-1"
    assert sent["idempotency_key"] == "main-f-ds-9b-1" and sent["hyperparams"] == C_SENT  # not the recommendation
    with pytest.raises(SystemExit, match="already created"):
        run("f", "create", "--max-credits", "100")
    run("f", "wait")
    rf = json.loads(f_path.read_text())["runs"]["f"]
    assert rf["run_id"] == "run-c-1" and rf["max_credits"] == 100 and rf["credits_spent"] == 0.0
    line = (tmp_path / "main_9b_run_f.txt").read_text()
    assert line.count("run f run-c-1 Qwen/Qwen3.5-9B created") == 1 and not (tmp_path / "main_4b_run_f.txt").exists()
    assert len(live.sent) == 1
    assert {n: files[n].read_bytes() for n in before} == before  # main_4b.json, c's and e's files untouched


def test_run_f_refuses_a_row_drop_a_4b_dataset_and_a_recipe_that_is_not_cs(tmp_path, monkeypatch):
    files, _, run = f_files(tmp_path, monkeypatch)
    live = FakeUpload(rows=1)  # Adaption kept 1 of the CSV's 2 rows
    monkeypatch.setattr(tm, "client", lambda: live)
    run("f", "seed")
    with pytest.raises(SystemExit, match="holds 1 rows"):
        run("f", "upload")
    with pytest.raises(SystemExit, match="holds 1 rows"):
        run("f", "create", "--max-credits", "100")
    assert live.sent == []
    # f's file holding a dataset set up for the 4B (66d2ad3d) is refused at create
    s = json.loads(files["main_9b_f.json"].read_text())
    s["dataset"] = json.loads(files["main_4b.json"].read_text())["dataset"]
    files["main_9b_f.json"].write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="not Qwen/Qwen3.5-9B"):
        run("f", "create", "--max-credits", "100")
    assert live.sent == []
    # the 4B runs cannot upload into their files, and f's seed refuses when c's sent recipe is not RECIPES["f"]
    with pytest.raises(SystemExit, match="main_4b.json's dataset"):
        run("c", "upload")
    files["main_9b_f.json"].unlink()
    monkeypatch.setitem(tm.RECIPES, "f", C_SENT | {"learning_rate": 1e-4})
    with pytest.raises(SystemExit, match="differ"):
        run("f", "seed")
    assert not files["main_9b_f.json"].exists()


# Runs d and d3: run c's recipe on train_mix3.csv (d3 at 3 epochs), each on a dataset of its own.

STATE_D = {"dataset": {"dataset_id": "d4d4d4d4", "model": tm.MODEL}}


def test_runs_d_and_d3_send_cs_recipe_on_the_4b_and_d3_only_changes_epochs():
    assert {"d", "d3"} <= set(tm.RUNS) and tm.model_of("d") == tm.model_of("d3") == "Qwen/Qwen3.5-4B"
    c = FakeCreate()
    _, kw = tm.create_run(c, STATE_D, "d")
    sent = c.sent[0]
    assert sent == kw and sent["dataset_id"] == "d4d4d4d4" and sent["model"] == "Qwen/Qwen3.5-4B"
    assert sent["idempotency_key"] == "main-d-d4d4d4d4"
    assert sent["hyperparams"] == C_SENT == tm.create_kwargs(STATE_B, "c")["hyperparams"]  # c's 15 fields exactly
    assert len(sent["hyperparams"]) == 15
    assert sent["max_iterations"] == 1 and sent["target_win_rate"] == 0.8
    assert sent["augmentation_domain_rows"] == 0 and sent["augmentation_general_rows"] == 0
    assert sent["column_mapping"] == tm.TRAIN_MAPPING and sent["training_method"] == "instruction"
    d3 = tm.create_kwargs(STATE_D, "d3")
    assert d3["idempotency_key"] == "main-d3-d4d4d4d4"
    assert d3["hyperparams"] == C_SENT | {"n_epochs": 3}
    assert {k for k in C_SENT if d3["hyperparams"][k] != C_SENT[k]} == {"n_epochs"}
    assert {k: v for k, v in d3.items() if k not in ("hyperparams", "idempotency_key")} == \
        {k: v for k, v in kw.items() if k not in ("hyperparams", "idempotency_key")}
    for n in ("d", "d3"):
        assert tm.limit_problems(tm.create_kwargs(STATE_D, n)["hyperparams"]) == []
    assert tm.step_range(sent["hyperparams"], "d") == (84, 91) and tm.step_range(d3["hyperparams"], "d3") == (63, 68)
    # a recommendation in d's file is never sent
    rec = {"model": tm.MODEL, "hyperparams": C_SENT | {"learning_rate": 1e-5, "weight_decay": 0.1}}
    assert tm.create_kwargs(STATE_D | {"recommended": rec}, "d")["hyperparams"] == C_SENT
    assert tm.create_kwargs(STATE_D | {"recommended": rec}, "d3")["hyperparams"] == C_SENT | {"n_epochs": 3}
    # the pins hold over a recipe
    assert tm.create_kwargs(STATE_D, "d3")["hyperparams"]["train_on_inputs"] is False
    # a, b, c, e and f are what they were
    assert tm.create_kwargs(STATE_B, "b")["hyperparams"] == B_HP
    assert tm.create_kwargs(STATE_B, "e")["hyperparams"] == tm.PINS
    assert tm.create_kwargs(STATE_F, "f")["hyperparams"] == C_SENT and tm.model_of("f") == "Qwen/Qwen3.5-9B"


def test_runs_d_and_d3_have_their_own_files_csv_and_dataset_names():
    root = Path(tm.__file__).resolve().parents[1]
    assert tm.state_path("d") == root / "data/adaption/main_4b_d.json"
    assert tm.state_path("d3") == root / "data/adaption/main_4b_d3.json"
    assert tm.csv_of("d") == tm.csv_of("d3") == root / "data/built/real_train/train_mix3.csv"
    assert all(tm.csv_of(n) == tm.CSV == root / "data/built/training/smoke.csv" for n in ("a", "b", "c", "e", "f", None))
    assert {n: tm.CSV_MD5[n] for n in ("d", "d3")} == {"d": "8393d4746f887332b20254ef7135db29",
                                                       "d3": "8393d4746f887332b20254ef7135db29"}
    assert {"d", "d3"} <= tm.OWN_DATASET and tm.UPLOAD_SUFFIX["d"] != tm.UPLOAD_SUFFIX["d3"]
    assert len(set(tm.UPLOAD_SUFFIX.values()) | {""}) == len(tm.UPLOAD_SUFFIX) + 1
    assert tm.archive_dir("d") == tm.archive_dir("d3") == tm.MODELS and tm.tag_of("d") == tm.tag_of("d3") == "main_4b"


class FakeUploadStatus(FakeUpload):
    """FakeUpload plus the dataset status read that `status` makes."""

    def get_status(self, did):
        return SimpleNamespace(status="succeeded", row_count=self.rows)

    def get(self, rid):
        run = super().get(rid)
        run.error = None
        return run


def d_files(tmp_path, monkeypatch):
    """f's fixture plus d's and d3's files, and a small train_mix3.csv that d and d3 take by default."""
    files, smoke, _ = f_files(tmp_path, monkeypatch)
    mix = tmp_path / "train_mix3.csv"
    mix.write_text("prompt,completion\nx,Keep — y.\nz,Skip — w.\nq,Explore — v.\n")
    for n in ("d", "d3"):
        files[f"main_4b_{n}.json"] = tmp_path / f"main_4b_{n}.json"
        monkeypatch.setitem(tm.STATES, n, files[f"main_4b_{n}.json"])
        monkeypatch.setitem(tm.RUN_CSVS, n, mix)
        monkeypatch.setitem(tm.CSV_MD5, n, tm.md5_of(mix))
    files["main_9b_f.json"].write_text(json.dumps({"dataset": {"dataset_id": "83835acc"}, "runs": {"f": {"run_id": "rf"}}}))

    def run(*words):  # no --csv: d and d3 must find train_mix3.csv themselves
        monkeypatch.setattr(sys, "argv", ["train_main.py", *words])
        tm.main()
    return files, smoke, mix, run


def test_runs_d_and_d3_seed_upload_create_wait_on_two_datasets_and_touch_nothing_else(tmp_path, monkeypatch, capsys):
    files, smoke, mix, run = d_files(tmp_path, monkeypatch)
    others = ("main_4b.json", "main_4b_c.json", "main_4b_e.json", "main_9b_f.json")
    before = {n: files[n].read_bytes() for n in others}
    live = FakeUploadStatus(rows=3)
    monkeypatch.setattr(tm, "client", lambda: live)
    with pytest.raises(SystemExit, match="seed"):
        run("d", "upload")
    run("d", "seed")
    run("d3", "seed")
    sd, sd3 = (json.loads(files[f"main_4b_{n}.json"].read_text()) for n in ("d", "d3"))
    assert "dataset" not in sd and "recommended" not in sd and sd["model"] == tm.MODEL
    assert sd["recipe_from"]["run_id"] == "1bbd8313" and sd["recipe_from"]["hyperparams"] == C_SENT
    assert sd["recipe"] == C_SENT and sd["seed_diff"] == {}
    assert sd3["recipe_from"]["hyperparams"] == C_SENT and sd3["recipe"] == C_SENT | {"n_epochs": 3}
    assert sd3["seed_diff"] == {"n_epochs": 3}
    with pytest.raises(FileExistsError):
        run("d", "seed")
    run("d", "upload")
    run("d3", "upload")
    run("d", "upload")  # again: already uploaded, no third dataset
    assert len(live.uploads) == 2
    (p1, k1), (p2, k2) = live.uploads
    assert p1 == p2 == str(mix)  # train_mix3.csv, never smoke.csv
    md5 = tm.md5_of(mix)
    assert k1["name"] == f"main-train-{md5[:8]}-d" and k2["name"] == f"main-train-{md5[:8]}-d3"
    assert k1["processing_mode"] == k2["processing_mode"] == "raw" and k1["column_mapping"] == tm.MAPPING
    dd, dd3 = (json.loads(files[f"main_4b_{n}.json"].read_text())["dataset"] for n in ("d", "d3"))
    assert dd["dataset_id"] == "ds-9b-1" != dd3["dataset_id"] == "ds-9b-2"
    assert dd["csv_md5"] == dd3["csv_md5"] == md5 and dd["csv_rows"] == dd3["csv_rows"] == 3
    assert dd["model"] == dd3["model"] == tm.MODEL and dd["status"]["row_count"] == 3
    run("config", "d")
    run("config", "d3")
    run("d", "create", "--max-credits", "100")
    run("d3", "create", "--max-credits", "100")
    s1, s2 = live.sent
    assert s1["dataset_id"] == "ds-9b-1" and s1["idempotency_key"] == "main-d-ds-9b-1" and s1["hyperparams"] == C_SENT
    assert s2["dataset_id"] == "ds-9b-2" and s2["idempotency_key"] == "main-d3-ds-9b-2"
    assert s2["hyperparams"] == C_SENT | {"n_epochs": 3} and s1["model"] == s2["model"] == tm.MODEL
    with pytest.raises(SystemExit, match="already created"):
        run("d", "create", "--max-credits", "100")
    run("d", "wait")
    run("d3", "wait")
    run("status", "d")
    out = capsys.readouterr().out
    assert "dataset ds-9b-1 succeeded: Adaption rows 3, CSV rows 3" in out and "run d run-c-1: succeeded" in out
    assert "warning" not in out  # status checks d's dataset against train_mix3.csv, not smoke.csv
    assert "csv: " + str(mix) in out and f"steps: 84 to 91" in out and f"steps: 63 to 68" in out
    for n in ("d", "d3"):
        r = json.loads(files[f"main_4b_{n}.json"].read_text())["runs"][n]
        assert r["run_id"] == "run-c-1" and r["max_credits"] == 100 and r["credits_spent"] == 0.0
        assert (tmp_path / f"main_4b_run_{n}.txt").read_text().count(f"run {n} run-c-1 Qwen/Qwen3.5-4B created") == 1
    assert len(live.sent) == 2
    assert {n: files[n].read_bytes() for n in others} == before  # main_4b.json, c's, e's and f's files untouched


def test_runs_d_and_d3_refuse_the_wrong_csv_a_row_drop_and_a_recipe_that_is_not_cs(tmp_path, monkeypatch):
    files, smoke, mix, run = d_files(tmp_path, monkeypatch)
    live = FakeUpload(rows=2)  # Adaption kept 2 of the CSV's 3 rows
    monkeypatch.setattr(tm, "client", lambda: live)
    run("d", "seed")
    # smoke.csv (or any CSV but the reviewed one) is refused before anything is uploaded
    monkeypatch.setattr(sys, "argv", ["train_main.py", "d", "upload", "--csv", str(smoke)])
    with pytest.raises(SystemExit, match="reviewed CSV"):
        tm.main()
    assert live.uploads == [] and "dataset" not in json.loads(files["main_4b_d.json"].read_text())
    with pytest.raises(SystemExit, match="holds 2 rows"):
        run("d", "upload")
    with pytest.raises(SystemExit, match="holds 2 rows"):
        run("d", "create", "--max-credits", "100")
    assert live.sent == []
    # d3's seed refuses when c's sent recipe differs in anything but n_epochs; d's when it differs at all
    for n, bad in (("d3", {"learning_rate": 1e-4}), ("d", {"n_epochs": 3})):
        files[f"main_4b_{n}.json"].unlink(missing_ok=True)
        c_state = json.loads(files["main_4b_c.json"].read_text())
        c_state["runs"]["c"]["settings"]["hyperparams"] = C_SENT | bad
        files["main_4b_c.json"].write_text(json.dumps(c_state))
        with pytest.raises(SystemExit, match="differ"):
            run(n, "seed")
        assert not files[f"main_4b_{n}.json"].exists()


# Run h: d3's recipe on the v4 fit set (train_fit.csv), on a dataset of its own, with no reasoning trace.

def test_run_h_sends_d3s_recipe_on_train_fit_without_a_trace():
    root = Path(tm.__file__).resolve().parents[1]
    assert "h" in tm.RUNS and tm.model_of("h") == "Qwen/Qwen3.5-4B" and tm.tag_of("h") == "main_4b"
    assert tm.state_path("h") == root / "data/adaption/main_4b_h.json"
    assert tm.csv_of("h") == root / "data/built/fit/train_fit.csv"
    assert tm.CSV_MD5["h"] == "b1fdfa85f3cb4b406f692e48b07f29fb"
    assert "h" in tm.OWN_DATASET and tm.UPLOAD_SUFFIX["h"] == "-h" and tm.SEED_FROM["h"] == "d3"
    assert "h" not in tm.SEED_DIFF
    kw = tm.create_kwargs(STATE_D, "h")
    assert kw["hyperparams"] == C_SENT | {"n_epochs": 3} == tm.create_kwargs(STATE_D, "d3")["hyperparams"]
    assert kw["column_mapping"] == tm.TRAIN_MAPPING and "reasoning_trace" not in kw["column_mapping"]
    assert kw["training_method"] == "instruction" and kw["idempotency_key"] == "main-h-d4d4d4d4"
    assert kw["max_iterations"] == 1 and kw["target_win_rate"] == 0.8
    assert kw["augmentation_domain_rows"] == 0 and kw["augmentation_general_rows"] == 0
    assert tm.limit_problems(kw["hyperparams"]) == []
    assert tm.step_range(kw["hyperparams"], "h") == (63, 99)


# Runs h2b and h2n: h's recipe on the cycle-2 sets, each on a dataset and state file of its own.

H_SENT = C_SENT | {"n_epochs": 3}  # run h's create, as sent on 2026-10-21 (d3's recipe)


def test_runs_h2b_and_h2n_send_hs_recipe_on_the_cycle2_sets():
    root = Path(tm.__file__).resolve().parents[1]
    for n, csv_name, md5 in (("h2b", "train_fit2_bal.csv", "dc5ea96f8eab21a567637f3ec2b941c2"),
                             ("h2n", "train_fit2_nat.csv", "0aeb3812df83ce5db175258ad1ce4532")):
        assert n in tm.RUNS and tm.model_of(n) == "Qwen/Qwen3.5-4B" and tm.tag_of(n) == "main_4b"
        assert tm.state_path(n) == root / f"data/adaption/main_4b_{n}.json"
        assert tm.csv_of(n) == root / f"data/built/fit/{csv_name}" and tm.CSV_MD5[n] == md5
        assert n in tm.OWN_DATASET and tm.UPLOAD_SUFFIX[n] == f"-{n}" and tm.SEED_FROM[n] == "h"
        assert n not in tm.SEED_DIFF
        kw = tm.create_kwargs(STATE_D, n)
        assert kw["hyperparams"] == H_SENT == tm.create_kwargs(STATE_D, "h")["hyperparams"]
        assert kw["column_mapping"] == tm.TRAIN_MAPPING and kw["training_method"] == "instruction"
        assert kw["data_format"] == "chat" and kw["idempotency_key"] == f"main-{n}-d4d4d4d4"
        assert kw["max_iterations"] == 1 and kw["target_win_rate"] == 0.8
        assert kw["augmentation_domain_rows"] == 0 and kw["augmentation_general_rows"] == 0
        assert tm.limit_problems(kw["hyperparams"]) == []
    assert tm.step_range(H_SENT, "h2b") == (6, 63) and tm.step_range(H_SENT, "h2n") == (47, 63)
    assert len({tm.state_path(n) for n in ("h", "h2b", "h2n")}) == 3


def test_runs_h2b_and_h2n_seed_from_h_upload_and_create_on_two_datasets_touching_nothing_else(tmp_path, monkeypatch):
    files, smoke, mix, run = d_files(tmp_path, monkeypatch)
    files["main_4b_h.json"] = tmp_path / "main_4b_h.json"
    monkeypatch.setitem(tm.STATES, "h", files["main_4b_h.json"])
    files["main_4b_h.json"].write_text(json.dumps({"dataset": {"dataset_id": "c432d24a"}, "runs": {"h": {
        "run_id": "63659c74", "settings": {"hyperparams": H_SENT}, "last": {"model": tm.MODEL}}}}))
    sets = {}
    for n, body in (("h2b", "prompt,completion\nx,Keep — y.\nz,Skip — w.\n"),
                    ("h2n", "prompt,completion\nx,Keep — y.\nz,Skip — w.\nq,Explore — v.\n")):
        sets[n] = tmp_path / f"{n}.csv"
        sets[n].write_text(body)
        files[f"main_4b_{n}.json"] = tmp_path / f"main_4b_{n}.json"
        monkeypatch.setitem(tm.STATES, n, files[f"main_4b_{n}.json"])
        monkeypatch.setitem(tm.RUN_CSVS, n, sets[n])
        monkeypatch.setitem(tm.CSV_MD5, n, tm.md5_of(sets[n]))
    others = ("main_4b.json", "main_4b_c.json", "main_4b_e.json", "main_9b_f.json", "main_4b_h.json")
    before = {n: files[n].read_bytes() for n in others}
    live = FakeUploadStatus(rows=2)  # h2b's 2 rows land; h2n's 3 must be refused as a row drop
    monkeypatch.setattr(tm, "client", lambda: live)
    run("h2b", "seed")
    run("h2n", "seed")
    for n in ("h2b", "h2n"):
        s = json.loads(files[f"main_4b_{n}.json"].read_text())
        assert s["recipe_from"]["run"] == "h" and s["recipe_from"]["run_id"] == "63659c74"
        assert s["recipe"] == s["recipe_from"]["hyperparams"] == H_SENT and "dataset" not in s
    run("h2b", "upload")
    with pytest.raises(SystemExit, match="holds 2 rows"):
        run("h2n", "upload")
    (p1, k1), (p2, k2) = live.uploads
    assert (p1, p2) == (str(sets["h2b"]), str(sets["h2n"]))
    assert k1["name"] == f"main-train-{tm.md5_of(sets['h2b'])[:8]}-h2b"
    assert k2["name"] == f"main-train-{tm.md5_of(sets['h2n'])[:8]}-h2n"
    assert k1["processing_mode"] == k2["processing_mode"] == "raw" and k1["column_mapping"] == tm.MAPPING
    run("h2b", "create", "--max-credits", "40")
    with pytest.raises(SystemExit, match="holds 2 rows"):
        run("h2n", "create", "--max-credits", "40")
    (sent,) = live.sent
    assert sent["idempotency_key"] == "main-h2b-ds-9b-1" and sent["hyperparams"] == H_SENT
    assert json.loads(files["main_4b_h2b.json"].read_text())["runs"]["h2b"]["max_credits"] == 40
    assert "runs" not in json.loads(files["main_4b_h2n.json"].read_text())
    assert {n: files[n].read_bytes() for n in others} == before
    # a seed refuses when h's sent recipe is not RECIPES["h"]
    files["main_4b_h2n.json"].unlink()
    h = json.loads(files["main_4b_h.json"].read_text())
    h["runs"]["h"]["settings"]["hyperparams"] = H_SENT | {"learning_rate": 1e-4}
    files["main_4b_h.json"].write_text(json.dumps(h))
    with pytest.raises(SystemExit, match="differ"):
        run("h2n", "seed")
    assert not files["main_4b_h2n.json"].exists()


# Runs A and B: h's recipe on the cycle-3 sets (scripts/build_fit_train3.py), each on a dataset and state file of its
# own, their records and archives under the tag main_4b_c3 (never run a's or b's names).

def test_runs_A_and_B_send_hs_recipe_on_the_cycle3_sets_under_their_own_names():
    root = Path(tm.__file__).resolve().parents[1]
    for n in ("A", "B"):
        assert n in tm.RUNS and tm.model_of(n) == "Qwen/Qwen3.5-4B" and tm.tag_of(n) == "main_4b_c3"
        assert tm.state_path(n) == root / f"data/adaption/main_4b_{n}.json"
        assert tm.csv_of(n) == root / f"data/built/fit/cycle3/train_{n}.csv"
        assert n in tm.OWN_DATASET and tm.UPLOAD_SUFFIX[n] == f"-{n}" and tm.SEED_FROM[n] == "h"
        assert n not in tm.SEED_DIFF and tm.RECIPES[n] == tm.RECIPES["h"]
        kw = tm.create_kwargs(STATE_D, n)
        assert kw["hyperparams"] == H_SENT == tm.create_kwargs(STATE_D, "h")["hyperparams"]
        assert kw["column_mapping"] == tm.TRAIN_MAPPING and kw["training_method"] == "instruction"
        assert kw["data_format"] == "chat" and kw["idempotency_key"] == f"main-{n}-d4d4d4d4"
        assert kw["max_iterations"] == 1 and kw["target_win_rate"] == 0.8
        assert kw["augmentation_domain_rows"] == 0 and kw["augmentation_general_rows"] == 0
        assert tm.limit_problems(kw["hyperparams"]) == []
        assert tm.archive_dir(n) == root / "data/models/main_4b_c3"
    # no file of A or B shares a name with one of a's or b's, even on a case-insensitive disk
    names = lambda n: {tm.state_path(n).name.lower(), f"{tm.tag_of(n)}_run_{n}.txt".lower(),  # noqa: E731
                       str(tm.archive_dir(n).relative_to(root) / n).lower()}
    assert not (names("A") | names("B")) & (names("a") | names("b"))
    assert len({tm.state_path(n) for n in ("h", "A", "B")}) == 3
    assert tm.CSV_MD5["A"] == "45baa1edd61628a0bb33eba8372a0231" and tm.CSV_MD5["B"] == "6fb0f861e38924aac848a43c009d229e"
    assert tm.step_range(H_SENT, "A") == (30, 63) and tm.step_range(H_SENT, "B") == (37, 63)


def test_runs_A_and_B_seed_from_h_upload_and_create_on_two_datasets_touching_nothing_else(tmp_path, monkeypatch):
    files, smoke, mix, run = d_files(tmp_path, monkeypatch)
    files["main_4b_h.json"] = tmp_path / "main_4b_h.json"
    monkeypatch.setitem(tm.STATES, "h", files["main_4b_h.json"])
    files["main_4b_h.json"].write_text(json.dumps({"dataset": {"dataset_id": "c432d24a"}, "runs": {"h": {
        "run_id": "63659c74", "settings": {"hyperparams": H_SENT}, "last": {"model": tm.MODEL}}}}))
    sets = {}
    for n in ("A", "B"):
        sets[n] = tmp_path / f"train_{n}.csv"
        sets[n].write_text(f"prompt,completion\n{n}x,Keep — y.\n{n}z,Skip — w.\n")
        files[f"main_4b_{n}.json"] = tmp_path / f"main_4b_{n}.json"
        monkeypatch.setitem(tm.STATES, n, files[f"main_4b_{n}.json"])
        monkeypatch.setitem(tm.RUN_CSVS, n, sets[n])
    others = ("main_4b.json", "main_4b_c.json", "main_4b_e.json", "main_9b_f.json", "main_4b_h.json")
    before = {n: files[n].read_bytes() for n in others}
    live = FakeUploadStatus(rows=2)
    monkeypatch.setattr(tm, "client", lambda: live)
    run("A", "seed")
    run("B", "seed")
    monkeypatch.setitem(tm.CSV_MD5, "A", None)
    with pytest.raises(SystemExit, match="no reviewed CSV md5"):  # no upload before the reviewed md5 is pinned
        run("A", "upload")
    monkeypatch.setitem(tm.CSV_MD5, "A", tm.md5_of(sets["A"]))
    monkeypatch.setitem(tm.CSV_MD5, "B", "0" * 32)
    with pytest.raises(SystemExit, match="reviewed CSV"):
        run("B", "upload")
    monkeypatch.setitem(tm.CSV_MD5, "B", tm.md5_of(sets["B"]))
    run("A", "upload")
    run("B", "upload")
    (p1, k1), (p2, k2) = live.uploads
    assert (p1, p2) == (str(sets["A"]), str(sets["B"]))
    assert k1["name"] == f"main-train-{tm.md5_of(sets['A'])[:8]}-A"
    assert k2["name"] == f"main-train-{tm.md5_of(sets['B'])[:8]}-B"
    assert k1["processing_mode"] == k2["processing_mode"] == "raw" and k1["column_mapping"] == tm.MAPPING
    run("A", "create", "--max-credits", "5")
    run("B", "create", "--max-credits", "5")
    s1, s2 = live.sent
    assert s1["hyperparams"] == s2["hyperparams"] == H_SENT
    assert s1["dataset_id"] != s2["dataset_id"] and s1["idempotency_key"] != s2["idempotency_key"]
    a, b = (json.loads(files[f"main_4b_{n}.json"].read_text()) for n in ("A", "B"))
    assert list(a["runs"]) == ["A"] and list(b["runs"]) == ["B"] and a["runs"]["A"]["max_credits"] == 5
    assert {n: files[n].read_bytes() for n in others} == before
    with pytest.raises(SystemExit, match="already created"):
        run("A", "create", "--max-credits", "5")
