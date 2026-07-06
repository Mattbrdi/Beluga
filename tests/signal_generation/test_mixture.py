import numpy as np
import pytest

from BSS.Utils.signal_generation import MixtureGenerator


def test_delay_matrix_is_reproduced_in_filters():
    delays = np.array([[0, 2], [1, 3]])
    mixture = MixtureGenerator(max_delay=3).generate(
        np.random.default_rng(0), n_sources=2, n_mics=2, delay_matrix=delays
    )

    recovered = np.argmax(mixture.filters, axis=2)
    np.testing.assert_array_equal(recovered, delays)


def test_delay_matrix_shape_is_validated():
    with pytest.raises(ValueError, match="shape"):
        MixtureGenerator(max_delay=3).generate(
            np.random.default_rng(0),
            n_sources=2,
            n_mics=2,
            delay_matrix=np.zeros((2, 3)),
        )


def test_random_delays_stay_within_configured_limit():
    mixture = MixtureGenerator(max_delay=4).generate(
        np.random.default_rng(0), n_sources=5, n_mics=3
    )

    delays = np.argmax(mixture.filters, axis=2)
    assert np.all((0 <= delays) & (delays <= 4))

