import pyarrow.parquet as pq

from src.ncs.config import PROCESSED_DATA_DIR

supported_sources = ['cliveome']

def load_signal(power, count, source='cliveome'):
    if source not in supported_sources:
        raise ValueError(f"Source '{source}' not supported. Supported sources: {supported_sources}")

    signals = []

    if source == 'cliveome':
        # todo: verify cliveome parquet file exists -> else advice to run cliveome setup script
        read_table = pq.read_table(PROCESSED_DATA_DIR / f"cliveome_category_{power}.parquet")
        read_df = read_table.to_pandas()
        signals = read_df.head(count)['signal']

    return signals
