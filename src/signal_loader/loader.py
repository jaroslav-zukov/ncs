supported_sources = ['cliveome']

def load_signal(power, source):
    if source not in supported_sources:
        raise ValueError(f"Source '{source}' not supported. Supported sources: {supported_sources}")

    if source == 'cliveome':
        print()

    return "signal"
