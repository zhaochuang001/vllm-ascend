# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock

import pytest

from vllm_ascend import utils


@pytest.mark.parametrize(
    "architectures,expected",
    [
        (["MiniMaxM3SparseForCausalLM"], True),
        (["MiniMaxM3SparseForConditionalGeneration"], True),
        (["OtherForCausalLM", "MiniMaxM3SparseForCausalLM"], True),
        (["MiniMaxM2ForCausalLM"], False),
        (["Qwen3ForCausalLM"], False),
        (["MiniMaxM3SparseForCausalLMLocal"], False),
        ([], False),
        (None, False),
    ],
)
def test_minimax_m3_model_uses_exact_hf_architectures(architectures, expected):
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=architectures, model_type="minimax_m3"))
    )
    with mock.patch.object(utils, "get_current_hardware_profile", side_effect=AssertionError("No hardware lookup")):
        assert utils.is_minimax_m3_model(config) is expected


@pytest.mark.parametrize(
    "config",
    [
        None,
        SimpleNamespace(),
        SimpleNamespace(model_config=None),
        SimpleNamespace(model_config=SimpleNamespace()),
        SimpleNamespace(model_config=SimpleNamespace(hf_config=None)),
        SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="minimax_m3"))),
    ],
)
def test_minimax_m3_model_returns_false_without_hf_architectures(config):
    assert utils.is_minimax_m3_model(config) is False


def test_minimax_m3_model_does_not_guess_from_other_model_metadata():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="MiniMaxM3SparseForCausalLM",
            hf_config=SimpleNamespace(model_type="minimax_m3"),
            hf_text_config=SimpleNamespace(architectures=["MiniMaxM3SparseForCausalLM"]),
        )
    )
    assert utils.is_minimax_m3_model(config) is False


@pytest.mark.parametrize("architecture", ["MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"])
@pytest.mark.parametrize(
    "dtype,skip_layers,fp8_supported,expected",
    [
        ("fp8", ["0"], True, True),
        ("fp8_e4m3", ["0"], True, True),
        ("auto", ["0"], True, False),
        ("bfloat16", ["0"], True, False),
        ("fp8_e5m2", ["0"], True, False),
        ("fp8", [], True, False),
        ("fp8", None, True, False),
        ("fp8", ["0"], False, False),
    ],
)
def test_minimax_m3_fp8_cache_keeps_dtype_skip_and_hardware_conditions(
    architecture, dtype, skip_layers, fp8_supported, expected
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture])),
        cache_config=SimpleNamespace(cache_dtype=dtype, kv_cache_dtype_skip_layers=skip_layers),
    )
    hardware = mock.Mock()
    hardware.supports.return_value = fp8_supported
    with mock.patch.object(utils, "get_current_hardware_profile", return_value=hardware) as hardware_lookup:
        assert utils.is_minimax_m3_model(config) is True
        assert utils.is_minimax_m3_fp8_kv_cache(config) is expected
        if dtype in ("fp8", "fp8_e4m3") and skip_layers:
            hardware_lookup.assert_called_once_with()
            hardware.supports.assert_called_once_with(utils.HardwareCapability.FP8_ATTENTION)
        else:
            hardware_lookup.assert_not_called()


@pytest.mark.parametrize(
    "config",
    [
        None,
        SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=["MiniMaxM3SparseForCausalLM"])),
            cache_config=None,
        ),
        SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=["OtherForCausalLM"])),
            cache_config=SimpleNamespace(cache_dtype="fp8", kv_cache_dtype_skip_layers=["0"]),
        ),
    ],
)
def test_minimax_m3_fp8_cache_short_circuits_without_supported_model_or_cache(config):
    with mock.patch.object(utils, "get_current_hardware_profile", side_effect=AssertionError("No hardware lookup")):
        assert utils.is_minimax_m3_fp8_kv_cache(config) is False
