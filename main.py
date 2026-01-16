from src.signal_loader.loader import load_signal

def print_hi(name):
    print(f'Hi, {name}')

if __name__ == '__main__':
    print_hi('PyCharm')

    signal = load_signal(13, 2)[0]

    print(f"Signal shape: {signal.shape}")
    print(f"Signal type: {type(signal)}")
    print(f"First 10 values: {signal[:10]}")