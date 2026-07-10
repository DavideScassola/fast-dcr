import pandas as pd
import torch
from tqdm import tqdm


def dcr(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    k: int = 1,
    device: str = "cpu",
    standardize: bool = True,
    batch_size: int = 1000,
    metric: str = "cosine",
    output_indexes: bool = False,
) -> pd.DataFrame:
    """Computes the distance to the closest record (DCR) between two dataframes.

    For each record in the source dataframe, it outputs the distances to the k
    closest records in the target dataframe. The output has the same number of
    rows as the source dataframe, and k columns.

    Args:
        source_df (pd.DataFrame): The source dataframe containing the records
        for which we want to compute the DCR. target_df (pd.DataFrame): The
        target dataframe containing the records against which we want to compute
        the DCR. k (int, optional): The number of closest records to consider.
        Defaults to 1. device (str, optional): The device on which to perform
        the computations. Defaults to "cpu". standardize (bool, optional):
        Whether to standardize the continuous features. The standardization is
        performed using the mean and standard deviation of the target dataset.
        Defaults to True. batch_size (int, optional): The size of the batches
        for processing. Defaults to 1000. metric (str, optional): The metric to
        use for computing distances. Options are "cosine" and "euclidean".
        Defaults to "cosine". output_indexes (bool, optional): Whether to output
        the indexes of the closest records. If true a tuple of (distances,
        indexes) is returned. Defaults to False.

    Returns:
        pd.DataFrame: A dataframe containing the distances to the k closest
        records in the target dataframe. If output_indexes is True, a tuple of
        (distances, indexes) is returned, where distances is a dataframe
        containing the distances and indexes is a dataframe containing the
        indexes of the closest records in the target dataframe.
    """
    # Avoid modifying the original dataframes
    source_df = source_df.copy()
    target_df = target_df.copy()

    # Detect categorical columns
    cat_columns = source_df.select_dtypes(
        include=["category", "object", str]
    ).columns.tolist()

    # Convert object, str and categorical columns to categorical
    for col in cat_columns:
        source_df[col] = source_df[col].astype("category")
        target_df[col] = target_df[col].astype("category")

    # Convert categorical columns to numerical codes
    for col in cat_columns:
        source_df[col + "_number"] = source_df[col].cat.codes
        target_df[col + "_number"] = target_df[col].cat.codes
        source_df = source_df.drop(columns=[col])
        target_df = target_df.drop(columns=[col])
        source_df = source_df.rename(columns={col + "_number": col})
        target_df = target_df.rename(columns={col + "_number": col})

    # Convert to tensors
    source_continuous = torch.tensor(
        source_df.drop(columns=cat_columns).values, dtype=torch.float32
    ).to(device)
    target_continuous = torch.tensor(
        target_df.drop(columns=cat_columns).values, dtype=torch.float32
    ).to(device)
    source_categorical = torch.tensor(
        source_df[cat_columns].values, dtype=torch.int64
    ).to(device)
    target_categorical = torch.tensor(
        target_df[cat_columns].values, dtype=torch.int64
    ).to(device)

    if standardize:
        # using the mean and std of the target dataset
        mean = target_continuous.mean(dim=0)
        std = target_continuous.std(dim=0)
        source_continuous = (source_continuous - mean) / std
        target_continuous = (target_continuous - mean) / std

    distance_matrix = torch.zeros((len(source_df), k)).to(device)

    if output_indexes:
        indices_matrix = torch.zeros((len(source_df), k), dtype=torch.int64).to(device)

    for i in tqdm(range(0, len(source_df), batch_size)):
        start = i
        end = min((i + 1) + batch_size, len(source_df))

        batch_continuous = source_continuous[start:end]
        batch_categorical = source_categorical[start:end]

        distance = torch.zeros((end - start, len(target_df))).to(device)

        if metric == "cosine":
            distance += 1 - torch.nn.functional.cosine_similarity(
                batch_continuous.unsqueeze(1), target_continuous.unsqueeze(0), dim=-1
            )
        elif metric == "euclidean":
            distance += torch.cdist(batch_continuous, target_continuous, p=2)

        distance += (
            batch_categorical.unsqueeze(1) != target_categorical.unsqueeze(0)
        ).sum(-1)

        distance_matrix[start:end] = torch.topk(distance, k=k, largest=False)[0]
        if output_indexes:
            indices_matrix[start:end] = torch.topk(distance, k=k, largest=False)[1]

    distances = distance_matrix.cpu().numpy()
    dcr_df = pd.DataFrame(distances, columns=[f"dcr_{i + 1}" for i in range(k)])
    if output_indexes:
        index_df = pd.DataFrame(
            indices_matrix.cpu().numpy(), columns=[f"index_{i + 1}" for i in range(k)]
        )
        return dcr_df, index_df
    return dcr_df
