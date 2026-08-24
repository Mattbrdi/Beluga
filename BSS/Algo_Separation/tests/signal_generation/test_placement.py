import numpy as np
import pytest

from BSS.Utils.signal_generation import (
    CompositeSignal,
    SignalPlacement,
    SignalPlacementGenerator,
    SignalPlacementSpec,
    SinSignal,
)


def _sine(duration=0.1):
    return SinSignal.generate(
        freq=1_000, sin_freq=100, phase=0, amplitude=1, time_duration=duration
    )


def test_placement_adds_latency_and_gain():
    placement = SignalPlacement(_sine(), start_time=0.05, window=None, gain=2)
    rendered = placement.render()

    np.testing.assert_array_equal(rendered.data[:50], 0)
    np.testing.assert_allclose(rendered.data[50:], 2 * placement.signal.data)


def test_composite_renders_at_exact_scene_length_and_sums_overlaps():
    signal = _sine()
    composite = CompositeSignal(
        [
            SignalPlacement(signal, 0, None, 1),
            SignalPlacement(signal, 0, None, 1),
        ],
        freq=1_000,
        duration=0.2,
    )

    rendered = composite.render()
    assert len(rendered.data) == 200
    np.testing.assert_allclose(rendered.data[:100], 2 * signal.data)
    np.testing.assert_array_equal(rendered.data[100:], 0)


def test_generator_rejects_unknown_signal_type():
    generator = SignalPlacementGenerator({"sine": SinSignal})

    with pytest.raises(ValueError, match="Type de signal inconnu"):
        generator.generate_placement(
            np.random.default_rng(0),
            freq=1_000,
            scene_duration=1,
            gain_range=(1, 1),
            spec=SignalPlacementSpec(signal_type="missing"),
        )


def test_placement_rejects_negative_start_time():
    with pytest.raises(ValueError, match="start_time"):
        SignalPlacement(_sine(), start_time=-0.1)

