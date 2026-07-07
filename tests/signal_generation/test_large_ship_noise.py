import numpy as np
import pytest

from BSS.Utils.signal_generation import LargeShipNoise


def _ship_noise(seed=42, tonal_mix=0.3, engine_mix=0.2):
    return LargeShipNoise.generate(
        freq=8_000,
        time_duration=1.0,
        shaft_rotation_frequency=1.5,
        propeller_blade_count=5,
        tonal_harmonic_count=5,
        tonal_decay=1.1,
        engine_cylinder_count=8,
        engine_harmonic_count=8,
        engine_harmonic_decay=0.9,
        cavitation_peak_frequency=55.0,
        low_frequency_slope=1.5,
        high_frequency_slope=1.0,
        modulation_frequency=2.0,
        modulation_depth=0.25,
        tonal_mix=tonal_mix,
        engine_mix=engine_mix,
        seed=seed,
    )


def test_large_ship_noise_is_reproducible_and_peak_normalized():
    first = _ship_noise(seed=12)
    second = _ship_noise(seed=12)

    np.testing.assert_array_equal(first.data, second.data)
    assert np.max(np.abs(first.data)) == pytest.approx(1.0)
    assert first.duration == pytest.approx(1.0)


def test_tonal_component_contains_blade_rate_harmonics():
    signal = _ship_noise(tonal_mix=1.0, engine_mix=0.0)
    spectrum = np.abs(np.fft.rfft(signal.data))
    frequencies = np.fft.rfftfreq(len(signal.data), d=1.0 / signal.freq)
    dominant_frequency = frequencies[np.argmax(spectrum[1:]) + 1]

    assert dominant_frequency == pytest.approx(signal.blade_rate, abs=1.0)


def test_engine_component_contains_firing_rate_harmonics():
    signal = _ship_noise(tonal_mix=0.0, engine_mix=1.0)
    spectrum = np.abs(np.fft.rfft(signal.data))
    frequencies = np.fft.rfftfreq(len(signal.data), d=1.0 / signal.freq)
    dominant_frequency = frequencies[np.argmax(spectrum[1:]) + 1]
    expected_firing_rate = (
        signal.shaft_rotation_frequency * signal.engine_cylinder_count
    )

    assert dominant_frequency == pytest.approx(expected_firing_rate, abs=1.0)


def test_random_parameters_follow_large_ship_class_rules():
    params = LargeShipNoise.generate_random_params(
        np.random.default_rng(3), freq=8_000
    )

    assert (
        LargeShipNoise.SHAFT_ROTATION_FREQUENCY_RANGE[0]
        <= params["shaft_rotation_frequency"]
        <= LargeShipNoise.SHAFT_ROTATION_FREQUENCY_RANGE[1]
    )
    assert params["propeller_blade_count"] in LargeShipNoise.PROPELLER_BLADE_COUNTS
    assert (
        LargeShipNoise.TONAL_HARMONIC_COUNT_RANGE[0]
        <= params["tonal_harmonic_count"]
        <= LargeShipNoise.TONAL_HARMONIC_COUNT_RANGE[1]
    )
    assert params["engine_cylinder_count"] in LargeShipNoise.ENGINE_CYLINDER_COUNTS


def test_large_ship_noise_rejects_tonal_harmonics_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        LargeShipNoise.generate(
            freq=200,
            time_duration=1.0,
            shaft_rotation_frequency=1.5,
            propeller_blade_count=5,
            tonal_harmonic_count=20,
            tonal_decay=1.1,
            engine_cylinder_count=8,
            engine_harmonic_count=8,
            engine_harmonic_decay=0.9,
            cavitation_peak_frequency=55.0,
            low_frequency_slope=1.5,
            high_frequency_slope=1.0,
            modulation_frequency=2.0,
            modulation_depth=0.25,
            tonal_mix=0.3,
            engine_mix=0.2,
            seed=1,
        )
