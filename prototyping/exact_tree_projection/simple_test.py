class MathArray:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index - 1]

    def __setitem__(self, index, value):
        self.data[index - 1] = value

    def __len__(self):
        return len(self.data)


def print_table(name, table_dict):
    """Print a dictionary with tuple keys as a formatted table."""
    if not table_dict:
        print(f"{name}: (empty)")
        return

    # Extract all unique i and l values
    i_values = sorted(set(key[0] for key in table_dict.keys()))
    l_values = sorted(set(key[1] for key in table_dict.keys()))

    # Determine column widths
    max_val_width = max(len(str(table_dict[key])) for key in table_dict.keys())
    col_width = max(max_val_width, 4) + 2

    # Print header
    print(f"\n{name}:")
    print(f"{'i\\l':<5}", end="")
    for l in l_values:
        print(f"{l:>{col_width}}", end="")
    print()
    print("-" * (5 + col_width * len(l_values)))

    # Print rows
    for i in i_values:
        print(f"{i:<5}", end="")
        for l in l_values:
            value = table_dict.get((i, l), "-")
            print(f"{str(value):>{col_width}}", end="")
        print()


def main():
    y = MathArray([10, 10, 1, 50, 1, 1, 50, 50])
    n = len(y)
    d = 2
    max_level = 3
    k = 6

    def subtree_size(level):
        return min(int((d ** (max_level + 1 - level) - 1) / (d - 1)), k - level)

    f = {}
    g = {}

    f_temp = {}
    g_temp = {}

    # leaves iterator
    for i in range(d ** (max_level - 1) + 1, d**max_level + 1):
        f[(i, 0)] = 0
        f[(i, 1)] = y[i] ** 2

        g[(i, 0)] = 0
        g[(i, 1)] = 0

    print(f"f: {f}")
    print(f"g: {g}")

    # level iterator
    print("Entering level iterator")

    for j in range(max_level - 1, 0, -1):  # j just like in paper
        print(f"j: {j} (processing tree level {j})")
        print(f"\tIterating i through {range(d ** (j - 1) + 1, (d**j) + 1)}")
        for i in range(d ** (j - 1) + 1, (d**j) + 1):  # i just like in paper
            print(f"{'\t' * 2}i: {i} (processing node {i})")
            f[(i, 0)] = 0
            f[(i, 1)] = y[i] ** 2
            g[(i, 0)] = [0, 0]
            g[(i, 1)] = [0, 0]

            print(f"{'\t' * 3}Iterating r through {range(1, d + 1)}")
            for r in range(1, d + 1):  # r just like in paper
                print(f"{'\t' * 4}r: {r} (processing child {r} of node {i})")
                print(
                    f"{'\t' * 5}Iterating l through {range(2, min(subtree_size(j), r * subtree_size(j + 1) + 1) + 1)}"
                )
                for l in range(
                    2, min(subtree_size(j), r * subtree_size(j + 1) + 1) + 1
                ):
                    print(f"{'\t' * 6}l: {l} (calculating budget {l})")
                    s_minus = max(0, l - ((r - 1) * subtree_size(j + 1) + 1))
                    print(f"{'\t' * 7}Calculated s_minus: {s_minus}")
                    s_plus = min(l - 1, subtree_size(j + 1))
                    print(f"{'\t' * 7}Calculated s_plus: {s_plus}")

                    s_hat = max(
                        range(s_minus, s_plus + 1),
                        key=lambda s: f[(d * (i - 1) + r, s)] + f[(i, l - s)],
                    )
                    print(f"{'\t' * 7}Calculated s_hat: {s_hat}")

                    f_temp[(i, l)] = f[d * (i - 1) + r, s_hat] + f[(i, l - s_hat)]

                    g_temp[(i, l)] = list(g[(i, l - s_hat)])
                    # updating the r-th coordinate of g_temp
                    g_temp[(i, l)][r - 1] = s_hat

                print(
                    f"{'\t' * 5}Iterating l through {range(2, min(subtree_size(j), r * subtree_size(j + 1) + 1) + 1)}"
                )
                for l in range(
                    2, min(subtree_size(j), r * subtree_size(j + 1) + 1) + 1
                ):
                    print(f"{'\t' * 6}Updating f[({i}, {l})] = {f_temp[(i, l)]}")
                    f[(i, l)] = f_temp[(i, l)]
                    print(f"{'\t' * 6}Updating g[({i}, {l})] = {g_temp[(i, l)]}")
                    g[(i, l)] = g_temp[(i, l)]

    print("-" * 40)
    print_table("f", f)
    print_table("g", g)
    print("-" * 40)

    f[(1, 0)] = 0
    f[(1, 1)] = y[1] ** 2
    g[(1, 0)] = [0, 0]
    g[(1, 1)] = [0, 0]

    print(f"Iterating r through {range(2, d + 1)}")
    for r in range(2, d + 1):
        print(f"{'\t' * 1}r: {r}")

        print(
            f"{'\t' * 2}Iterating l through {range(2, min(k, (r - 1) * subtree_size(1) + 1) + 1)}"
        )
        for l in range(2, min(k, (r - 1) * subtree_size(1) + 1) + 1):
            print(f"{'\t' * 3}l: {l}")
            s_minus = max(1, l - ((r - 2) * subtree_size(1) + 1))
            print(f"{'\t' * 4}Calculated s_minus: {s_minus}")
            s_plus = min(l - 1, subtree_size(1))
            print(f"{'\t' * 4}Calculated s_plus: {s_plus}")
            s_hat = max(
                range(s_minus, s_plus + 1), key=lambda s: f[(r, s)] + f[(1, l - s)]
            )
            print(f"{'\t' * 4}Calculated s_hat: {s_hat}")

            f_temp[(1, l)] = f[(r, s_hat)] + f[(1, l - s_hat)]

            g_temp[(1, l)] = list(g[(1, l - s_hat)])
            g_temp[(1, l)][r - 1] = s_hat

        for l in range(2, min(k, (r - 1) * subtree_size(1) + 1) + 1):
            f[(1, l)] = f_temp[(1, l)]
            g[(1, l)] = g_temp[(1, l)]

    print("-" * 40)
    print_table("f", f)
    print_table("g", g)
    print("-" * 40)

    print("Backtracking the solution")

    tau = MathArray([0] * n)
    tau[1] = 1
    gamma = MathArray([0] * n)
    gamma[1] = k

    for j in range(max_level):
        print(f"j: {j}")

        start_node = 1 if j == 0 else d ** (j - 1) + 1

        print(f"Iterating i in {range(start_node, d**j + 1)}")
        for i in range(start_node, d**j + 1):
            if tau[i] == 1:
                for r in range(max(1, 2 - j), d + 1):
                    if g[(i, gamma[i])][r - 1] > 0:
                        tau[d * (i - 1) + r] = 1
                        gamma[d * (i - 1) + r] = g[(i, gamma[i])][r - 1]

    print("\nCalculating solution\n")

    y_hat = MathArray([0] * n)
    for i in range(1, n + 1):
        y_hat[i] = y[i] * tau[i]

    print(f"y_hat: {y_hat.data}")


if __name__ == "__main__":
    main()
