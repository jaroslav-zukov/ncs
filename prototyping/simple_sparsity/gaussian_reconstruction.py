import numpy as np


def generate_sparse_x(n, s):
    signal = np.zeros(n)
    indices = np.sort(np.random.choice(n, s, replace=False))
    # in my setup would be replaced by randint with the -300 to 300 range
    signal[indices] = np.random.random(s)
    return signal


def create_gaussian_operator(n: int, m: int, seed: int = None):
    rng = np.random.default_rng(seed)
    phi = rng.normal(0, 1.0 / np.sqrt(m), size=(m, n))
    phi_pinv = np.linalg.pinv(phi)

    def measure(signal: np.ndarray) -> np.ndarray:
        return phi @ signal

    def adjoint(measurements: np.ndarray) -> np.ndarray:
        return phi.T @ measurements

    def pseudo_inverse(measurements: np.ndarray) -> np.ndarray:
        return phi_pinv @ measurements

    return measure, adjoint, pseudo_inverse


def sparse_projection(array, sparsity):
    result = np.zeros_like(array)
    indices = np.argsort(np.abs(array))[-sparsity:]
    result[indices] = array[indices]
    return result


def support(array):
    return set(np.nonzero(array)[0])


def classical_cosamp(
    measure_op,
    adjoint_op,
    pseudo_inv_op,
    y,
    taget_s,
):
    x_hat = np.zeros(1000)
    r = y
    for i in range(50):
        e = adjoint_op(r)
        omega_e_double_support = support(
            sparse_projection(array=e, sparsity=2 * taget_s)
        )
        t = support(x_hat).union(omega_e_double_support)

        b = np.zeros(1000)
        b[list(t)] = pseudo_inv_op(y)[list(t)]
        x_hat = sparse_projection(array=b, sparsity=taget_s)
        r = y - measure_op(x_hat)
    return x_hat


def main():
    print("Gaussian Reconstruction")
    # generate the sparse signal x with n 1000 and s 10.
    n = 1000
    s = 10
    signal = generate_sparse_x(n, s)
    print(signal)

    ## This part will be wrapped in for loop with m
    m = 100
    # take gaussian operators for A, A_transposed, and A_pseudoinverse
    measure, adjoint, pseudo_inverse = create_gaussian_operator(n, m)
    # measure y
    y = measure(signal)

    # implement simple cosamp with normal sparse projection (zero smallest n-s coeffs)
    # plot the resulting reconstructions on relative error (|x-x_hat|/|x|) per m, fixing sparsity at 1%


if __name__ == "__main__":
    main()
