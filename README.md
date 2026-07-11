# torch-dcr
[![CI](https://github.com/DavideScassola/torch-dcr/actions/workflows/ci.yml/badge.svg)](https://github.com/DavideScassola/torch-dcr/actions/workflows/ci.yml)

A library for efficient computation of DCR (Distance to Closest Record) of heterogeneous tabular data.

To install the library in editable mode, run the following command:
```
python -m pip install -e .
```


## Example usage
```python
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

dcr(source_df, target_df, metric="cosine")
```
Output:
```
      dcr_1
0  0.667516
1  0.166855
2  0.449738
```

More advanced usage:
```python
dcr_df, indexes_df = dcr(
    source_df=source_df, # Source DataFrame, the DCR is computed for each record in this DataFrame
    target_df=target_df, # Target DataFrame where the closest records are searched
    output_indexes=True, # If True, the indexes of the closest records are returned along with the distances
    k=2,                 # Number of closest records to consider for each record in the source DataFrame
    metric="l1",         # Distance metric to use for computing distances, options are "cosine", "euclidean", and "l1"
    device="cuda",       # Device on which to perform the computations, can be "cpu" or "cuda" for GPU acceleration
    standardize=True,    # Whether to standardize the continuous features before computing distances
    batch_size=1000      # Size of the batches for processing, useful for large DataFrames to avoid memory issues
    )
```

Output:
```
DCR:
       dcr_1     dcr_2
0  2.790001  3.465423
1  1.551357  2.918901
2  2.716115  3.055198

Indexes:
    index_1  index_2
0        2        1
1        1        0
2        2        4
```
