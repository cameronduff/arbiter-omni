"""Tests for the DirectML-native AdamW, telemetry helpers, and v6-Max checkpoint utilities."""

import copy
import importlib.util
import os

import pytest
import torch
import torch.nn as nn

from arbiter_omni.training.optim import DirectMLNativeAdamW
from arbiter_omni.training.telemetry import get_rss_mb, get_system_memory_mb, memory_summary


def _load_train_v6_max():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "train_v6_max.py")
    spec = importlib.util.spec_from_file_location("train_v6_max_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_pair(seed: int = 0):
    torch.manual_seed(seed)
    a = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4))
    b = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4))
    b.load_state_dict(a.state_dict())
    return a, b


@pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
def test_directml_adamw_matches_torch_adamw(weight_decay):
    """The mul_/add_ formulation must be numerically equivalent to torch.optim.AdamW."""
    ref_model, our_model = _make_pair()
    ref = torch.optim.AdamW(ref_model.parameters(), lr=1e-2, weight_decay=weight_decay)
    ours = DirectMLNativeAdamW(our_model.parameters(), lr=1e-2, weight_decay=weight_decay)

    x = torch.randn(32, 8)
    y = torch.randn(32, 4)
    for _ in range(25):
        for model, opt in ((ref_model, ref), (our_model, ours)):
            opt.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            opt.step()

    for p_ref, p_ours in zip(ref_model.parameters(), our_model.parameters()):
        assert torch.allclose(p_ref, p_ours, atol=1e-6, rtol=1e-5)


def test_directml_adamw_skips_params_without_grad():
    model = nn.Linear(4, 4)
    before = model.weight.detach().clone()
    opt = DirectMLNativeAdamW(model.parameters(), lr=1e-1)
    opt.step()  # no grads populated
    assert torch.equal(before, model.weight)


def test_directml_adamw_state_dict_roundtrip():
    model_a, model_b = _make_pair()
    opt_a = DirectMLNativeAdamW(model_a.parameters(), lr=1e-2)
    x, y = torch.randn(16, 8), torch.randn(16, 4)
    for _ in range(3):
        opt_a.zero_grad()
        ((model_a(x) - y) ** 2).mean().backward()
        opt_a.step()

    model_b.load_state_dict(model_a.state_dict())
    opt_b = DirectMLNativeAdamW(model_b.parameters(), lr=1e-2)
    # A real resume loads from disk; deepcopy reproduces that (load_state_dict would
    # otherwise alias same-device tensors and double-update the shared moment buffers).
    opt_b.load_state_dict(copy.deepcopy(opt_a.state_dict()))

    for model, opt in ((model_a, opt_a), (model_b, opt_b)):
        opt.zero_grad()
        ((model(x) - y) ** 2).mean().backward()
        opt.step()
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-7)


def test_telemetry_returns_sane_values():
    assert get_rss_mb() > 0.0
    total, available = get_system_memory_mb()
    assert total >= available >= 0.0
    summary = memory_summary()
    assert "rss=" in summary


class _Holder(nn.Module):
    """Mimics the trainable submodule layout of ArbiterOmniModel."""

    def __init__(self, scale: float):
        super().__init__()
        self.fusion = nn.Linear(4, 4)
        self.decision_head = nn.Linear(4, 2)
        self.speculative_head = nn.Linear(4, 2)
        self.encoder = nn.Linear(1000, 1000)  # frozen and must never be averaged
        with torch.no_grad():
            for p in (self.fusion.weight, self.decision_head.weight, self.speculative_head.weight):
                p.fill_(scale)


def test_running_weight_average_is_equal_weight_mean_and_skips_encoder():
    mod = _load_train_v6_max()
    swa = mod.RunningWeightAverage()
    for scale in (1.0, 2.0, 6.0):
        swa.update(_Holder(scale))

    assert swa.n == 3
    states = swa.module_state_dicts()
    assert set(states) == {"fusion", "decision_head", "speculative_head"}
    for name in states:
        assert torch.allclose(states[name]["weight"], torch.full_like(states[name]["weight"], 3.0))
    assert not any(k.startswith("encoder") for k in swa.avg)
    assert all(t.device.type == "cpu" and t.dtype == torch.float32 for t in swa.avg.values())


def test_running_weight_average_state_roundtrip():
    mod = _load_train_v6_max()
    a = mod.RunningWeightAverage()
    a.update(_Holder(1.0))
    a.update(_Holder(3.0))

    b = mod.RunningWeightAverage()
    b.load_state_dict(a.state_dict())
    b.update(_Holder(5.0))  # (1 + 3 + 5) / 3 = 3
    out = b.module_state_dicts()["fusion"]["weight"]
    assert b.n == 3
    assert torch.allclose(out, torch.full_like(out, 3.0))


def test_atomic_torch_save_never_leaves_partial_file(tmp_path, monkeypatch):
    mod = _load_train_v6_max()
    target = tmp_path / "ckpt.pt"

    mod.atomic_torch_save({"x": torch.arange(4)}, str(target))
    assert torch.load(target)["x"].tolist() == [0, 1, 2, 3]
    assert not (tmp_path / "ckpt.pt.tmp").exists()

    # A failure mid-save must leave the previous good checkpoint untouched.
    def boom(obj, path):
        with open(path, "wb") as f:
            f.write(b"partial")
        raise RuntimeError("simulated kill during write")

    monkeypatch.setattr(mod.torch, "save", boom)
    with pytest.raises(RuntimeError):
        mod.atomic_torch_save({"x": torch.arange(99)}, str(target))
    monkeypatch.undo()
    assert torch.load(target)["x"].tolist() == [0, 1, 2, 3]


def test_delta_formatting():
    mod = _load_train_v6_max()
    assert mod._delta([0.5]) == "first epoch"
    assert mod._delta([0.50, 0.55], 100.0, "%") == "d+5.0000%"


def test_to_cpu_tree_moves_nested_tensors_and_keeps_scalars():
    mod = _load_train_v6_max()
    tree = {"state": {0: {"step": 3, "exp_avg": torch.ones(2)}}, "param_groups": [{"lr": 1e-3, "params": [0]}]}
    out = mod.to_cpu_tree(tree)
    assert out["state"][0]["step"] == 3
    assert out["state"][0]["exp_avg"].device.type == "cpu"
    assert out["param_groups"][0]["lr"] == 1e-3


def test_resume_dir_roundtrip_and_old_fallback(tmp_path):
    mod = _load_train_v6_max()
    model = _Holder(2.0)
    opt = mod.DirectMLNativeAdamW(
        [p for n, p in model.named_parameters() if not n.startswith("encoder")], lr=1e-2
    )
    for p in opt.param_groups[0]["params"]:
        p.grad = torch.ones_like(p)
    opt.step()
    swa = mod.RunningWeightAverage()
    swa.update(model)

    resume_dir = str(tmp_path / "ckpt.pt.resume")
    meta = {"epoch": 3, "history": {"loss": [1.0]}, "model_config": {}, "scheduler": {}, "swa_scheduler": None}
    mod.save_resume_dir(resume_dir, meta, model, opt, swa)
    assert not os.path.exists(resume_dir + ".tmp")
    assert not os.path.exists(resume_dir + ".old")

    loaded = mod.load_resume_dir(resume_dir)
    assert loaded["meta"]["epoch"] == 3
    assert set(loaded["weights"]) == {"fusion", "decision_head", "speculative_head"}
    assert torch.equal(loaded["weights"]["fusion"]["weight"], model.fusion.weight.detach())
    assert loaded["swa"]["n"] == 1
    assert len(loaded["optim"]["state"]) == 6  # weight + bias for each of the 3 trained heads

    # Interrupted swap: only <dir>.old survives -> loader must fall back to it
    os.replace(resume_dir, resume_dir + ".old")
    assert mod.load_resume_dir(resume_dir)["meta"]["epoch"] == 3

    with pytest.raises(FileNotFoundError):
        mod.load_resume_dir(str(tmp_path / "does_not_exist"))


def _tiny_trainer(log_interval=1):
    from arbiter_omni import (
        ArbiterOmniModel, ArbiterOmniTrainer, MockMultimodalEncoder, MultimodalDecisionDataset,
        TrainingConfig, generate_synthetic_dataset,
    )

    ds = MultimodalDecisionDataset(generate_synthetic_dataset(num_samples=16, seed=7))
    model = ArbiterOmniModel(encoder=MockMultimodalEncoder(embed_dim=64), hidden_dim=64, scoring_dim=64)
    cfg = TrainingConfig(batch_size=8, num_epochs=1, learning_rate=1e-3, fp16=False, log_interval=log_interval)
    return ArbiterOmniTrainer(model=model, config=cfg), ds


def test_trainer_save_checkpoint_is_atomic_and_loadable(tmp_path):
    """save_checkpoint must write via a temp file + rename and produce CPU tensors."""
    trainer, _ = _tiny_trainer()
    path = str(tmp_path / "t.pt")
    trainer.save_checkpoint(path)

    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert all(t.device.type == "cpu" for t in state["fusion"].values())
    assert all(t.device.type == "cpu" for t in state["decision_head"].values())
    trainer.load_checkpoint(path)  # roundtrip into the model


def test_trainer_emits_verbose_batch_progress(caplog):
    """Per-batch telemetry must include loss components, grad norm, throughput, ETA and memory."""
    import logging

    trainer, ds = _tiny_trainer(log_interval=1)
    from torch.utils.data import DataLoader
    from arbiter_omni import collate_multimodal_decision

    loader = DataLoader(ds, batch_size=8, collate_fn=collate_multimodal_decision)
    with caplog.at_level(logging.INFO, logger="arbiter_omni.training.trainer"):
        trainer.train_epoch(loader)

    lines = [r.getMessage() for r in caplog.records if "[train" in r.getMessage()]
    assert len(lines) == 2  # 16 samples / batch 8, logged every batch
    for key in ("loss=", "ce=", "margin=", "moe_aux=", "acc=", "grad_norm=", "lr=", "samples/s", "epoch_eta=", "rss="):
        assert key in lines[-1], f"missing {key!r} in: {lines[-1]}"
