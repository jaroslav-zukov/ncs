import math
from math import floor

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pod5

from src.ncs.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def main():
    cliveome_path = RAW_DATA_DIR / "cliveome.pod5"

    data = {}

    with pod5.Reader(cliveome_path) as reader:
        for read in reader.reads():
            signal_length = len(read.signal)
            signal_category = floor(math.log2(signal_length))

            if signal_category not in data:
                data[signal_category] = pd.DataFrame(columns=["read_id", "signal"])

            new_row = pd.DataFrame({"read_id": [str(read.read_id)], "signal": [read.signal[:2**signal_category]]})
            data[signal_category] = pd.concat([data[signal_category], new_row], ignore_index=True)

    categories = data.keys()

    for category in categories:
        output_path = PROCESSED_DATA_DIR / f"cliveome_category_{category}.parquet"
        table = pa.Table.from_pandas(data[category])
        pq.write_table(table, output_path)
        print(f"Wrote category {category} to {output_path}")


def test_parquet():
    read_table = pq.read_table(PROCESSED_DATA_DIR / "cliveome_category_13.parquet")
    read_df = read_table.to_pandas()

    first_entry = read_df.iloc[0]
    print("First entry values and types:")
    for column, value in first_entry.items():
        print(f"{column}: {value} (type: {type(value).__name__})")

    print("Parquet length:", len(read_df))


if __name__ == '__main__':
    # todo: verify if cliveome present, if not download (maybe ask for permission)
    main()
    test_parquet()
    # todo: verify the end state of the parquet files (13-22 exist and have expected number of entries)
