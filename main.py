from src.signal_loader.loader import load_signal

def print_hi(name):
    print(f'Hi, {name}')

if __name__ == '__main__':
    print_hi('PyCharm')

    # Load a signal with 1000 data points
    signal_size = 1000
    signal = load_signal(signal_size)

    print(f"Signal shape: {signal.shape}")
    print(f"Signal type: {type(signal)}")
    print(f"First 10 values: {signal[:10]}")