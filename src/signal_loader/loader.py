import pyarrow.parquet as pq

from src.signal_loader.config import PROCESSED_DATA_DIR

supported_sources = ['cliveome']

def load_signal(power, count, source='cliveome'):
    if source not in supported_sources:
        raise ValueError(f"Source '{source}' not supported. Supported sources: {supported_sources}")

    signals = []

    if source == 'cliveome':
        read_table = pq.read_table(PROCESSED_DATA_DIR / f"cliveome_category_{power}.parquet")
        read_df = read_table.to_pandas()
        signals = read_df.head(count)['signal']

    return signals
