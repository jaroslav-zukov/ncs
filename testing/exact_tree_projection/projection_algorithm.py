def project(coeffs, sparsity):
    d = 2
    max_level = coeffs.max_level
    y = coeffs.get_one_index_coeffs()

    k = len(y) - 2

    f_table = {}
    g_table = {}

    f_tilde_table = {}
    g_tilde_table = {}

    for i in range((d ** (max_level - 1)) + 1, d ** max_level):
        f_table[(i, 0)] = 0
        f_table[(i, 1)] = y[i] ** 2

    for j in range(max_level - 1, 0, -1):
        for i in range(d ** (j - 1) + 1, d ** j + 1):
            f_table[(i, 0)] = 0
            f_table[(i, 1)] = y[i] ** 2

            g_table[(i, 0)] = 0
            g_table[(i, 1)] = 0

            for r in range(1, d + 1):
                for l in range(2, 1 + min(
                        subtrees(j, k, max_level, d),
                        r * subtrees(j + 1, k, max_level, d) + 1
                )):
                    s_minus = max(1, l - ((r - 1) * l * (j + 1) + 1))
                    s_plus = min(l - 1, subtrees(j + 1, k, max_level, d))

                    s_hat = max(
                        range(s_minus, s_plus + 1),
                        key=lambda s: f_table[(d * (i - 1) + r, s)] + f_table[(i, l - s)]
                    )

                    f_tilde_table[(i, l)] = f_table[(d * (i - 1) + r, s_hat)] + f_table[(i, l - s_hat)]
                    g_tilde_table[(i,l)] = g_table[(i, l-s_hat)]

    print(f_table)


def subtrees(j, k, J, d):
    return int(min((d ** (J + 1 - j) - 1) / (d - 1), k - j))
