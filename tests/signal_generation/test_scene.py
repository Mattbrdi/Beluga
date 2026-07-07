import inspect

import numpy as np
import pytest

from BSS.Utils.signal_generation import (
    AudioSceneGenerator,
    AudioSceneSpec,
    CompositeSignalSpec,
    SignalPlacementSpec,
)


def _generator(seed=None):
    return AudioSceneGenerator(
        fs=8_000,
        scene_duration=0.5,
        n_sources=2,
        n_mics=3,
        max_delay=4,
        source_placement_rate=0.75,
        local_noise_placement_rate=1.0 / 3.0,
        source_gain_range=(0.5, 1.0),
        local_noise_gain_range=(0.05, 0.3),
        continuous_noise_gain_range=(1.0, 1.0),
        source_registry=None,
        local_noise_registry=None,
        continuous_noise_registry=None,
        seed=seed,
    )


def _stable_spec(snr_db=None):
    source = CompositeSignalSpec(
        n_placements=1,
        allowed_signal_types="sine",
        placements=[SignalPlacementSpec(signal_params={"time_duration": 0.2})],
    )
    return AudioSceneSpec(
        source_specs=[source, source],
        random_local_noise_signal_types="gaussian_noise",
        random_continuous_noise_signal_types="gaussian_noise",
        snr_db=snr_db,
    )


def test_audio_scene_generator_constructor_has_no_default_parameters():
    signature = inspect.signature(AudioSceneGenerator)

    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )


def test_scene_has_expected_channel_and_sample_dimensions():
    scene = _generator().generate(_stable_spec(), seed=10)

    assert scene.sources.data.shape == (2, 4_000)
    assert scene.clean_mixed.data.shape == (3, 4_000)
    assert scene.noise.data.shape == (3, 4_000)
    assert scene.mixed.data.shape == (3, 4_000)
    assert scene.mixing.filters.shape == (3, 2, 5)


def test_same_seed_reproduces_complete_scene():
    generator = _generator()
    first = generator.generate(_stable_spec(), seed=123)
    second = generator.generate(_stable_spec(), seed=123)

    np.testing.assert_array_equal(first.sources.data, second.sources.data)
    np.testing.assert_array_equal(first.mixing.filters, second.mixing.filters)
    np.testing.assert_array_equal(first.mixed.data, second.mixed.data)


def test_generator_seed_is_used_when_generate_seed_is_omitted():
    generator = _generator(seed=321)

    first = generator.generate(_stable_spec())
    second = generator.generate(_stable_spec())

    np.testing.assert_array_equal(first.mixed.data, second.mixed.data)
    assert first.metadata.seed == 321


def test_requested_continuous_noise_snr_is_reached():
    requested_snr = 6.0
    scene = _generator().generate(_stable_spec(snr_db=requested_snr), seed=5)
    signal_energy = sum(float(signal.energy) for signal in scene.clean_mixed.signals)
    continuous_energy = sum(
        noise.energy for noise in scene.metadata.continuous_noises
    )
    measured_snr = 10 * np.log10(signal_energy / continuous_energy)

    assert measured_snr == pytest.approx(requested_snr, abs=1e-10)


def test_too_many_source_specs_are_rejected():
    spec = AudioSceneSpec(source_specs=[CompositeSignalSpec()] * 3)

    with pytest.raises(ValueError, match="Trop de CompositeSignalSpec"):
        _generator().generate(spec, seed=0)


def test_explicit_signal_type_can_be_outside_random_type_selection():
    spec = AudioSceneSpec(
        random_source_signal_types="sine",
        random_local_noise_signal_types="gaussian_noise",
        random_continuous_noise_signal_types="gaussian_noise",
        source_specs=[
            CompositeSignalSpec(
                n_placements=1,
                placements=[
                    SignalPlacementSpec(
                        signal_type="spike",
                        signal_params={"time_duration": 0.01},
                    )
                ],
            ),
            CompositeSignalSpec(n_placements=2),
        ],
    )

    scene = _generator().generate(spec, seed=8)

    assert scene.metadata.source_composites[0].placements[0].signal_type == "spike"
    assert all(
        placement.signal_type == "sine"
        for placement in scene.metadata.source_composites[1].placements
    )


def test_metadata_records_realized_delays_and_seed():
    delays = np.array([[0, 1], [2, 3], [4, 0]])
    spec = _stable_spec()
    spec.delay_matrix = delays
    scene = _generator().generate(spec, seed=99)

    np.testing.assert_array_equal(scene.metadata.delay_matrix, delays)
    assert scene.metadata.seed == 99


def test_large_ship_noise_can_be_selected_as_continuous_noise():
    spec = _stable_spec()
    spec.random_continuous_noise_signal_types = "large_ship_noise"

    scene = _generator().generate(spec, seed=14)

    assert all(
        noise.signal_type == "large_ship_noise"
        for noise in scene.metadata.continuous_noises
    )
