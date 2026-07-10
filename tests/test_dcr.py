import pandas as pd

from torch_dcr import dcr


def test_dcr1():
    source_df = pd.DataFrame(
        {
            "A": [4, 0, 3, -3, 1],
            "B": ["a", "b", "c", "d", "e"],
            "C": [1.12, -20, 3.4, -1.6, 60],
            "D": ["1", "2", "3", "3", "4"],
        }
    )

    target_df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["f", "b", "g", "d", "d"],
            "C": [0.0, 0.0, 2.0, -12.0, -4.0],
            "D": ["2", "2", "9", "4", "4"],
        }
    )

    dcr_df, index_df = dcr(
        source_df, target_df, device="cpu", k=1, output_indexes=True, metric="cosine"
    )

    expected_dcr = pd.DataFrame(
        {"dcr_1": [1.063227, 0.393267, 1.0, 0.943477, 2.006269]}
    ).astype("float32")

    expected_index = pd.DataFrame({"index_1": [1, 3, 2, 2, 2]})

    pd.testing.assert_frame_equal(dcr_df, expected_dcr, check_exact=False, atol=1e-5)
    pd.testing.assert_frame_equal(index_df, expected_index)
