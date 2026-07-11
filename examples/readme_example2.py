import pandas as pd

from torch_dcr import dcr

source_df = pd.DataFrame(
    {
        "A": [4, 0, 3],
        "B": ["a", "b", "d"],
        "C": [1.12, -1.6, 6],
    }
)

target_df = pd.DataFrame(
    {
        "A": [1, 2, 3, 4, 5],
        "B": ["f", "b", "g", "d", "d"],
        "C": [0.0, 0.0, 2.0, -12.0, -4.0],
    }
)

dcr_df, indexes_df = dcr(
    # Source DataFrame, the DCR is computed for each record in this DataFrame
    source_df=source_df,
    # Target DataFrame where the closest records are searched
    target_df=target_df,
    # If True, the indexes of the closest records are returned along with the distances
    output_indexes=True,
    # Number of closest records to consider for each record in the source DataFrame
    k=2,
    # Distance metric, options are "cosine", "euclidean", and "l1"
    metric="l1",
    # Device on which to perform the computations, can be "cpu" or "cuda"
    device="cpu",
    # Whether to standardize the continuous features before computing distances
    standardize=True,
    # Size of the batches, useful for large DataFrames to avoid memory issues
    batch_size=1000,
)

print("DCR:\n", dcr_df)
print("Indexes:\n", indexes_df)
