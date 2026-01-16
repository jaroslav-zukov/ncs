import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

df = pd.DataFrame({
    'column1': [1, 2, 3],
    'column2': ['a', 'b', 'c']
})

table = pa.Table.from_pandas(df)

pq.write_table(table, 'example.parquet')

read_table = pq.read_table('example.parquet')
read_df = read_table.to_pandas()
print(read_df)


