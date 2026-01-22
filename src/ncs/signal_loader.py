import pyarrow.parquet as pq

from src.ncs.config import PROCESSED_DATA_DIR

supported_sources = ["cliveome"]


def load_signal(power, count, source="cliveome"):
    """
    Load signal data from a specified source.

    This function loads a specified number of signal arrays from a processed
    data source. Each signal is a NumPy array of length 2^power.

    Args:
        power (int): The power of 2 that determines signal length.
                    Each signal will have length 2^power.
                    Example: power=10 yields signals of length 1024.
        count (int): The number of signal arrays to load.
                    Must be positive and not exceed available signals in source.
        source (str, optional): The data source identifier.
                               Currently supported: "cliveome".
                               Defaults to "cliveome".

    Returns:
        pandas.Series: A pandas Series containing `count` NumPy arrays,
                      where each array represents a signal of length 2^power.

    Raises:
        ValueError: If the specified source is not in the list of supported sources.
        FileNotFoundError: If the required parquet file for the specified power
                          does not exist in the processed data directory.

    Example:
        >>> signals = load_signal(power=10, count=5, source="cliveome")
        >>> print(len(signals))
        5
        >>> print(len(signals.iloc[0]))
        1024

    Notes:
        - For the "cliveome" source, data is read from parquet files located in
          PROCESSED_DATA_DIR with naming convention: 'cliveome_category_{power}.parquet'
        - If the parquet file doesn't exist, ensure the cliveome setup script
          has been run to generate the processed data files.
    """

    if source not in supported_sources:
        raise ValueError(
            f"Source '{source}' not supported. Supported sources: {supported_sources}"
        )

    signals = []

    if source == "cliveome":
        # todo: verify cliveome parquet file exists -> else advice to run cliveome setup script
        read_table = pq.read_table(
            PROCESSED_DATA_DIR / f"cliveome_category_{power}.parquet"
        )
        read_df = read_table.to_pandas()
        signals = read_df.head(count)["signal"]

    return signals
