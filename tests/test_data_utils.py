import pandas as pd

from src.data_utils import FEATURES, prepare_classification_data


def test_prepare_classification_data_excludes_converted_and_missing_rows():
    data = pd.DataFrame(
        {
            "Subject ID": ["S1", "S2", "S3", "S4"],
            "Group": ["Nondemented", "Demented", "Converted", "Demented"],
            "Age": [70, 75, 80, 82],
            "EDUC": [16, 12, 14, 10],
            "SES": [2, 3, 2, None],
            "MMSE": [29, 22, 27, 20],
            "eTIV": [1400, 1500, 1450, 1550],
            "nWBV": [0.78, 0.70, 0.72, 0.68],
            "ASF": [1.20, 1.10, 1.15, 1.05],
        }
    )

    clean, X, y, groups = prepare_classification_data(data)

    assert clean["Subject ID"].tolist() == ["S1", "S2"]
    assert y.tolist() == [0, 1]
    assert groups.tolist() == ["S1", "S2"]
    assert list(X.columns) == FEATURES
