import numpy as np

from graphrnn_v2.stats.heuristics import MMD


def test_emd():

    x = np.array([3, 2, 1])
    y = np.array([1, 2, 3])

    assert MMD.emd(x, y) == 4.0


def test_mmd():

    x = [np.array([1, 2, 3])]
    y = [np.array([3, 2, 1])]

    assert np.allclose(MMD.mmd(x, y), 0.3985251941663839)

    x = [
        np.arange(0, 5, dtype=np.float64),
        np.arange(1, 6, dtype=np.float64),
    ]
    y = [
        np.arange(2, 7, dtype=np.float64),
        np.arange(3, 8, dtype=np.float64),
    ]

    assert np.allclose(MMD.mmd(x, y), 0.13596405410529133)
