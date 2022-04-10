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
    def l2(x: float, y: float) -> float:
        return x-y

    @staticmethod
    def mmd_emd(x: list[np.array], y: list[np.array]) -> float:
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
            emd = MMD.emd(x, y)
            return np.exp(-emd * emd / 2.0)

        d_xx = sum([pairwise_gaussian_emd(x_i, x_j) for x_i in x for x_j in x]) / len(x) ** 2
        d_yy = sum([pairwise_gaussian_emd(y_i, y_j) for y_i in y for y_j in y]) / len(y) ** 2
        d_xy = sum([pairwise_gaussian_emd(x_i, y_i) for x_i in x for y_i in y]) / (len(x) * len(y))

        return d_xx + d_yy - 2.0 * d_xy

    @staticmethod
    def mmd_l2(x: list[np.array], y: list[np.array]) -> float:
        """
        Compute MMD between two lists of values.

        @param x: list of values
        @param y: list of values
        """

        def pairwise_gaussian_dist(x: float, y: float) -> float:
            dist = MMD.l2(x, y)
            return np.exp(-dist * dist / 2.0)

        d_xx = sum([pairwise_gaussian_dist(x_i, x_j) for x_i in x for x_j in x]) / len(x) ** 2
        d_yy = sum([pairwise_gaussian_dist(y_i, y_j) for y_i in y for y_j in y]) / len(y) ** 2
        d_xy = sum([pairwise_gaussian_dist(x_i, y_i) for x_i in x for y_i in y]) / (len(x) * len(y))

        return d_xx + d_yy - 2.0 * d_xy