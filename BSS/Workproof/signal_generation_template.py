from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BELUGA_ROOT = Path(__file__).resolve().parents[2]
if str(BELUGA_ROOT) not in sys.path:
    sys.path.insert(0, str(BELUGA_ROOT))

from BSS.Utils.signal_generation import (  # noqa: E402
    AudioSceneGenerator,
    AudioSceneSpec,
    CompositeSignalSpec,
    GaussianNoise,
    SignalPlacementSpec,
    SinSignal,
    SpikeSignal,
)


def build_demo_scene():
    """
    Template minimal pour generer une scene acoustique synthetique.

    Les specs ne fixent que certains parametres. Tout champ omis reste tire
    aleatoirement par le generateur.
    """
    fs = 8_000
    scene_duration = 3.0
    n_sources = 2
    n_mics = 3
    max_delay = 12

    generator = AudioSceneGenerator(
        fs=fs,
        scene_duration=scene_duration,
        n_sources=n_sources,
        n_mics=n_mics,
        max_delay=max_delay,
        source_placements_range=(1, 3),  # Chaque source contient entre 1 et 3 evenements places.
        local_noise_placements_range=(0, 2),  # Chaque micro recoit entre 0 et 2 bruits locaux places.
        source_gain_range=(0.5, 1.0),
        local_noise_gain_range=(0.02, 0.15),
        continuous_noise_gain_range=(0.05, 0.05),
        source_registry={"sine": SinSignal, "spike": SpikeSignal},
        local_noise_registry={"spike": SpikeSignal, "gaussian_noise": GaussianNoise},
        continuous_noise_registry={"gaussian_noise": GaussianNoise},
        seed=42,
    )

    # Exemple: on force seulement quelques choix.
    # Source 0: un sinus de 440 Hz place a 0.25 s avec une fenetre hann.
    # Source 1: laissee entierement aleatoire.
    # Bruit local du micro 0: un spike force.
    # Bruits continus: gaussiens, avec un ecart-type controle.
    spec = AudioSceneSpec(
        source_specs=[
            CompositeSignalSpec(
                n_placements=1,
                placements=[
                    SignalPlacementSpec(
                        signal_type="sine",
                        signal_params={
                            "sin_freq": 440.0,
                            "time_duration": 0.8,
                            "amplitude": 1.0,
                        },
                        start_time=0.25,
                        window="hann",
                        gain=1.0,
                    )
                ],
            ),
            CompositeSignalSpec(
                n_placements=1,
                placements=[
                    SignalPlacementSpec(
                        signal_type=["sine", "spike"],
                    )
                ],
            ),
        ],
        local_noise_specs=[
            CompositeSignalSpec(
                n_placements=1,
                placements=[
                    SignalPlacementSpec(
                        signal_type="spike",
                        signal_params={"amplitude": 1.0, "time_duration": 0.005},
                        start_time=1.4,
                        window=None,
                        gain=0.1,
                    )
                ],
            )
        ],
        continuous_noise_specs=[
            SignalPlacementSpec(
                signal_type="gaussian_noise",
                signal_params={"std": 0.02},
                gain=1.0,
            )
            for _ in range(n_mics)
        ],
        delay_matrix=np.array(
            [
                [0, 0],
                [6, 2],
                [12, 5],
            ]
        ),
    )

    return generator.generate(spec=spec)


def print_scene_summary(scene) -> None:
    print("Scene generated")
    print(f"  fs: {scene.metadata.fs} Hz")
    print(f"  duration: {scene.metadata.duration} s")
    print(f"  sources: {scene.metadata.source_types}")
    print(f"  local noises: {scene.metadata.local_noise_types}")
    print(f"  continuous noises: {scene.metadata.continuous_noise_types}")
    print("  delay matrix:")
    print(scene.metadata.delay_matrix)
    print(f"  sources shape: {scene.sources.data.shape}")
    print(f"  clean_mixed shape: {scene.clean_mixed.data.shape}")
    print(f"  noise shape: {scene.noise.data.shape}")
    print(f"  mixed shape: {scene.mixed.data.shape}")


def plot_scene(scene) -> None:
    fig_sources, _ = scene.sources.plot(overlay=False, figsize=(12, 5))
    fig_sources.suptitle("Sources propres")

    fig_clean, _ = scene.clean_mixed.plot(overlay=False, figsize=(12, 6))
    fig_clean.suptitle("Micros - melange propre avec retards")

    fig_noise, _ = scene.noise.plot(overlay=False, figsize=(12, 6))
    fig_noise.suptitle("Bruit total par micro")

    fig_mixed, _ = scene.mixed.plot(overlay=False, figsize=(12, 6))
    fig_mixed.suptitle("Micros - signal final")

    plt.show()


def main() -> None:
    scene = build_demo_scene()
    print_scene_summary(scene)
    plot_scene(scene)


if __name__ == "__main__":
    main()
