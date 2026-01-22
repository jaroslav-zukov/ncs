from src.ncs.wt_coeffs import WtCoeffs


class MathArray:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index - 1]

    def __setitem__(self, index, value):
        self.data[index - 1] = value

    def __len__(self):
        return len(self.data)


def tree_projection(wt_coeffs: WtCoeffs, k: int) -> WtCoeffs:
    root_count = wt_coeffs.root_count

    if root_count != 1:
        raise ValueError(
            "Only root count 1 is supported at the moment (Use haar wavelet)"
        )

    d = 2
    y = MathArray(wt_coeffs.flat_coeffs)
    n = len(y)
    max_level = wt_coeffs.max_level

    def subtree_size(level):
        return min(int((d ** (max_level + 1 - level) - 1) / (d - 1)), k - level)

    f = {}
    g = {}

    f_temp = {}
    g_temp = {}

    for i in range(d ** (max_level - 1) + 1, d**max_level + 1):
        f[(i, 0)] = 0
        f[(i, 1)] = y[i] ** 2

        g[(i, 0)] = 0
        g[(i, 1)] = 0

    for j in range(max_level - 1, 0, -1):
        for i in range(d ** (j - 1) + 1, (d**j) + 1):
            f[(i, 0)] = 0
            f[(i, 1)] = y[i] ** 2
            g[(i, 0)] = [0, 0]
            g[(i, 1)] = [0, 0]

            for r in range(1, d + 1):
                for l in range(
                    2, min(subtree_size(j), r * subtree_size(j + 1) + 1) + 1
                ):
                    s_minus = max(0, l - ((r - 1) * subtree_size(j + 1) + 1))
                    s_plus = min(l - 1, subtree_size(j + 1))

                    s_hat = max(
                        range(s_minus, s_plus + 1),
                        key=lambda s: f[(d * (i - 1) + r, s)] + f[(i, l - s)],
                    )

                    f_temp[(i, l)] = f[d * (i - 1) + r, s_hat] + f[(i, l - s_hat)]

                    g_temp[(i, l)] = list(g[(i, l - s_hat)])
                    g_temp[(i, l)][r - 1] = s_hat

                for l in range(
                    2, min(subtree_size(j), r * subtree_size(j + 1) + 1) + 1
                ):
                    f[(i, l)] = f_temp[(i, l)]
                    g[(i, l)] = g_temp[(i, l)]

    f[(1, 0)] = 0
    f[(1, 1)] = y[1] ** 2
    g[(1, 0)] = [0, 0]
    g[(1, 1)] = [0, 0]

    for r in range(2, d + 1):
        for l in range(2, min(k, (r - 1) * subtree_size(1) + 1) + 1):
            s_minus = max(1, l - ((r - 2) * subtree_size(1) + 1))
            s_plus = min(l - 1, subtree_size(1))
            s_hat = max(
                range(s_minus, s_plus + 1), key=lambda s: f[(r, s)] + f[(1, l - s)]
            )

            f_temp[(1, l)] = f[(r, s_hat)] + f[(1, l - s_hat)]

            g_temp[(1, l)] = list(g[(1, l - s_hat)])
            g_temp[(1, l)][r - 1] = s_hat

        for l in range(2, min(k, (r - 1) * subtree_size(1) + 1) + 1):
            f[(1, l)] = f_temp[(1, l)]
            g[(1, l)] = g_temp[(1, l)]

    tau = MathArray([0] * n)
    tau[1] = 1
    gamma = MathArray([0] * n)
    gamma[1] = k

    for j in range(max_level):
        start_node = 1 if j == 0 else d ** (j - 1) + 1

        for i in range(start_node, d**j + 1):
            if tau[i] == 1:
                for r in range(max(1, 2 - j), d + 1):
                    if g[(i, gamma[i])][r - 1] > 0:
                        tau[d * (i - 1) + r] = 1
                        gamma[d * (i - 1) + r] = g[(i, gamma[i])][r - 1]

    y_hat = MathArray([0] * n)
    for i in range(1, n + 1):
        y_hat[i] = y[i] * tau[i]

    return WtCoeffs.from_flat_coeffs(
        flat_coeffs=y_hat.data,
        root_count=root_count,
        max_level=max_level,
        wavelet=wt_coeffs.wavelet,
    )
