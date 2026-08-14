import inspect

import numpy as np
import pytest

from BSS.Utils.signal_generation import (
    GaussianNoise,
    LargeShipNoise,
    SinSignal,
    SpikeSignal,
    WhistleSignal,
)
from BSS.Utils.signal_generation.TypedSignal import SinSignal as LegacySinSignal
from BSS.Utils.signal_generation.signals import SinSignal as PackagedSinSignal


def test_sine_has_expected_samples_and_values():
    signal = SinSignal.generate(
        freq=1_000, sin_freq=100, phase=0, amplitude=2, time_duration=0.1
    )

    assert len(signal.data) == 100
    assert signal.duration == pytest.approx(0.1)
    assert signal.data[0] == pytest.approx(0)
    assert np.max(np.abs(signal.data)) <= 2


def test_gaussian_noise_is_reproducible_from_its_seed():
    first = GaussianNoise.generate(freq=1_000, std=0.2, time_duration=0.1, seed=42)
    second = GaussianNoise.generate(freq=1_000, std=0.2, time_duration=0.1, seed=42)

    np.testing.assert_array_equal(first.data, second.data)


def test_random_generation_preserves_fixed_parameters():
    signal = SpikeSignal.generate_random(
        rng=np.random.default_rng(7),
        freq=2_000,
        amplitude=3,
        time_duration=0.01,
    )

    assert signal.amplitude == 3
    assert signal.duration == pytest.approx(0.01)


def test_random_generation_rules_are_inherited_class_attributes():
    class HighToneSignal(SinSignal):
        SIN_FREQUENCY_MIN = 700.0
        SIN_FREQUENCY_MAX = 700.0

    signal = HighToneSignal.generate_random(
        rng=np.random.default_rng(7), freq=2_000, time_duration=0.1
    )

    assert signal.sin_freq == 700.0


def test_unknown_fixed_parameter_is_rejected():
    with pytest.raises(ValueError, match="Parametres inconnus"):
        SinSignal.generate_random(
            np.random.default_rng(0), 1_000, not_a_parameter=1
        )


def test_whistle_rejects_harmonics_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        WhistleSignal.generate(
            freq=8_000,
            time_duration=0.5,
            f_start=1_000,
            f_min=500,
            f_max=2_000,
            segment_duration_range=(0.1, 0.2),
            direction_change_probability=0.2,
            sweep_rate_range=(100, 200),
            jitter_tau=0.05,
            jitter_std=4,
            harmonic_amplitudes=(1.0, 0.5),
            harmonic_phases=(0.0, 0.0),
            envelope_base=0.75,
            envelope_depth=0.15,
            seed=0,
        )


def test_whistle_leaves_fading_to_signal_placement():
    with pytest.raises(ValueError, match="Parametres inconnus"):
        WhistleSignal.generate_random(
            np.random.default_rng(0),
            freq=48_000,
            fade_duration=0.04,
        )


def test_whistle_only_allows_hann_window():
    assert WhistleSignal.allowed_windows == ("hann",)
    assert WhistleSignal.default_window == "hann"


def test_whistle_randomizes_a_normalized_harmonic_structure():
    params = WhistleSignal.generate_random_params(
        rng=np.random.default_rng(12), freq=48_000
    )
    amplitudes = params["harmonic_amplitudes"]
    phases = params["harmonic_phases"]

    assert WhistleSignal.HARMONIC_COUNT_RANGE[0] <= len(amplitudes)
    assert len(amplitudes) <= WhistleSignal.HARMONIC_COUNT_RANGE[1]
    assert amplitudes[0] == pytest.approx(1.0)
    assert len(phases) == len(amplitudes)


def test_whistle_fixed_harmonics_remain_prioritary():
    signal = WhistleSignal.generate_random(
        rng=np.random.default_rng(4),
        freq=48_000,
        time_duration=0.5,
        harmonic_amplitudes=(1.0, 0.25, 0.05, 0.01, 0.005),
    )

    assert signal.harmonic_amplitudes == (1.0, 0.25, 0.05, 0.01, 0.005)
    assert len(signal.harmonic_phases) == 5


def test_historical_typed_signal_import_remains_compatible():
    assert LegacySinSignal is SinSignal
    assert PackagedSinSignal is SinSignal


@pytest.mark.parametrize(
    "signal_cls",
    [SinSignal, SpikeSignal, GaussianNoise, WhistleSignal, LargeShipNoise],
)
def test_deterministic_generate_has_no_default_synthesis_parameters(signal_cls):
    signature = inspect.signature(signal_cls.generate)

    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
