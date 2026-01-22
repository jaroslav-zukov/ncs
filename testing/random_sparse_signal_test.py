from src.ncs.sparse_signal_generator import generate_tree_sparse_signals


def main():
    sparse_signals = generate_tree_sparse_signals(3, 2, 3, 'haar')
    for sparse_signal in sparse_signals:
        print(sparse_signal)

if __name__ == "__main__":
    main()