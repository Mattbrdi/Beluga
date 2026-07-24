import numpy as np
from BSS.Utils.signal_class import MultiSignal, Mixture, Signal
from beamforming.classes import *
from beamforming.configuration import *
from beamforming.simulation.tdoas import get_tdoas_from_source


def _apply_delay(delay_matrix: NDArray[np.int32], msignal: MultiSignal) -> MultiSignal:
    mixer = Mixture.create_delay_mixture(E=1, S=4, L=20000, delay_matrix=delay_matrix)
    tetra = mixer.apply(msignal, mode="same")  # dans tetra.data matrice (micro, sample)
    return tetra


def generate_audio_from_source_and_tetrahedra(
    tetrahedra: TetrahedralArray,
    source: Source,
) -> MultiSignal:
    """Simulate the signals recorded by a four-hydrophone array.

    The source signal is delayed at each hydrophone according to the
    source direction, hydrophone positions, and configured sound speed.
    Propagation delays are rounded to the nearest sample.

    Parameters
    ----------
    tetrahedra : TetrahedralArray
        Geometry of the four-hydrophone array. Positions are expected
        in metres.
    source : Source
        Source waveform and three-dimensional source position. The
        position is expected in metres.

    Returns
    -------
    MultiSignal
        Four-channel simulated recording, with one signal per
        hydrophone and a sampling rate of ``SAMPLING_RATE``.

    Notes
    -----
    The simulation uses a far-field plane-wave approximation and
    integer-sample delays. It does not model attenuation, reflections,
    reverberation, or fractional-sample delays.
    """
    audio_signal = Signal(source.signal, SAMPLING_RATE)
    pairwise_tdoas = get_tdoas_from_source(tetrahedra, source)  # Tdoas are in seconds
    relative_delays = np.array(
        [
            [0.0],
            [pairwise_tdoas[0]],  # mic 2 relative to mic 1
            [pairwise_tdoas[1]],  # mic 3 relative to mic 1
            [pairwise_tdoas[2]],  # mic 4 relative to mic 1
        ]
    )

    delay_samples = np.rint(relative_delays * SAMPLING_RATE).astype(int)
    delay_samples -= delay_samples.min()  # make every delay causal/nonnegative

    msignal = MultiSignal([audio_signal])

    return _apply_delay(delay_samples, msignal)
