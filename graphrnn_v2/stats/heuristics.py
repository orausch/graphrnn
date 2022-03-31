import numpy as np


class MMD:
    @staticmethod
    def emd(x: np.array, y: np.array) -> float:
        """
        Compute EMD between two distributions converting distribution x -> distribution y.

        @param x: 1D pmf of source distribution x
        @param y: 1D pmf of target distribution y
        """

        diffs = x - y
        emd = np.cumsum(diffs)

        return np.sum(np.abs(emd))

    @staticmethod
    def mmd(x: list[np.array], y: list[np.array]) -> float:
        """
        Compute MMD between two lists of lists of values.

        @param x: list of 1D np.arrays
        @param y: list of 1D np.arrays
        """

        assert all(len(x_i.shape) == 1 for x_i in x), "x must be a list of 1D arrays"
        assert all(len(y_i.shape) == 1 for y_i in y), "y must be a list of 1D arrays"

        # Normalize values.
        x = [x_i / np.sum(x_i) for x_i in x]
        y = [y_i / np.sum(y_i) for y_i in y]

        def pairwise_gaussian_emd(x: np.array, y: np.array) -> float:
            # Normalize length.
            max_ = max(len(x), len(y))

            x = np.concatenate((x, np.zeros(max_ - len(x))))
            y = np.concatenate((y, np.zeros(max_ - len(y))))
            # FIXME: What if the middle bins are not the same? (e.g. x = [1, 3, 4] and y = [1, 2, 3]).
            emd = MMD.emd(x, y)
            return np.exp(-emd * emd / 2.0)

        d_xx = sum([pairwise_gaussian_emd(x_i, x_j) for x_i in x for x_j in x]) / len(x) ** 2
        d_yy = sum([pairwise_gaussian_emd(y_i, y_j) for y_i in y for y_j in y]) / len(y) ** 2
        d_xy = sum([pairwise_gaussian_emd(x_i, y_i) for x_i in x for y_i in y]) / (len(x) * len(y))

        return d_xx + d_yy - 2.0 * d_xy
