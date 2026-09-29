import torch

from boltz.model.modules.confidence import (
    _concat_confidence_outputs as concat_confidence_v1,
)
from boltz.model.modules.confidencev2 import (
    _concat_confidence_outputs as concat_confidence_v2,
)


def _sample_outputs():
    return [
        {
            "ptm": torch.tensor([1.0]),
            "chains_pae": {
                0: torch.tensor([2.0]),
            },
            "pair_chains_pae": {
                0: {
                    0: torch.tensor([3.0]),
                    1: torch.tensor([4.0]),
                }
            },
            "pair_chains_iptm": {
                0: {
                    0: torch.tensor([5.0]),
                }
            },
        },
        {
            "ptm": torch.tensor([10.0]),
            "chains_pae": {
                0: torch.tensor([20.0]),
            },
            "pair_chains_pae": {
                0: {
                    0: torch.tensor([30.0]),
                    1: torch.tensor([40.0]),
                }
            },
            "pair_chains_iptm": {
                0: {
                    0: torch.tensor([50.0]),
                }
            },
        },
    ]


def test_concat_confidence_outputs_v1_handles_nested_dicts():
    result = concat_confidence_v1(_sample_outputs())

    assert torch.equal(result["ptm"], torch.tensor([1.0, 10.0]))
    assert torch.equal(result["chains_pae"][0], torch.tensor([2.0, 20.0]))
    assert torch.equal(
        result["pair_chains_pae"][0][1], torch.tensor([4.0, 40.0])
    )
    assert torch.equal(
        result["pair_chains_iptm"][0][0], torch.tensor([5.0, 50.0])
    )


def _run_v2_sequential(training):
    """Drive the v2 sequential branch with a stub per-sample forward."""
    from types import SimpleNamespace

    from boltz.model.modules.confidencev2 import ConfidenceModule

    def fake_forward(*args, **kwargs):
        return {
            "pae": torch.ones(1, 3, 3),
            "pae_logits": torch.ones(1, 3, 3, 64),
            "pde_logits": torch.ones(1, 3, 3, 64),
            "ptm": torch.tensor([1.0]),
        }

    fake_module = SimpleNamespace(training=training, forward=fake_forward)
    return ConfidenceModule.forward(
        fake_module,
        s_inputs=None,
        s=None,
        z=torch.zeros(1, 3, 3, 1),
        x_pred=torch.zeros(2, 5, 3),
        feats=None,
        pred_distogram_logits=None,
        multiplicity=2,
        run_sequentially=True,
    )


def test_v2_sequential_inference_drops_unused_logits():
    result = _run_v2_sequential(training=False)

    assert "pae_logits" not in result
    assert "pde_logits" not in result
    assert result["pae"].shape == (2, 3, 3)
    assert torch.equal(result["ptm"], torch.tensor([1.0, 1.0]))


def test_v2_sequential_training_keeps_logits_for_loss():
    result = _run_v2_sequential(training=True)

    assert result["pae_logits"].shape == (2, 3, 3, 64)
    assert result["pde_logits"].shape == (2, 3, 3, 64)


def test_concat_confidence_outputs_v2_handles_nested_dicts():
    result = concat_confidence_v2(_sample_outputs())

    assert torch.equal(result["ptm"], torch.tensor([1.0, 10.0]))
    assert torch.equal(result["chains_pae"][0], torch.tensor([2.0, 20.0]))
    assert torch.equal(
        result["pair_chains_pae"][0][1], torch.tensor([4.0, 40.0])
    )
    assert torch.equal(
        result["pair_chains_iptm"][0][0], torch.tensor([5.0, 50.0])
    )
