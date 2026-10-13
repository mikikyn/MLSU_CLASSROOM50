import numpy as np
import pandas as pd
from src.transformers import CreditCleaner


def make_df(util):
    n = len(util)
    return pd.DataFrame({
        "utilization": util,
        "debt_ratio": [0.5] * n,
        "income": [5000.0] * n,
        "age": [40] * n,
        "late_30_59": [0] * n,
        "late_60_89": [0] * n,
        "late_90": [0] * n,
    })


def test_caps_learned_on_train_applied_to_new_data():
    train = make_df(list(np.linspace(0, 1, 100)))
    cleaner = CreditCleaner().fit(train)
    cap = cleaner.caps_["utilization"]

    new = make_df([1000.0])          # огромное значение в новых данных
    out = cleaner.transform(new)

    assert cap < 1000.0                                  # граница выучена на train
    assert out["utilization"].iloc[0] == cap             # новое значение обрезано по ней
    assert cleaner.caps_["utilization"] == cap           # transform границу не менял


def test_service_codes_become_nan():
    df = make_df([0.5, 0.5])
    df.loc[0, "late_90"] = 98
    out = CreditCleaner().fit(df).transform(df)

    assert np.isnan(out.loc[0, "late_90"])
    assert out.loc[1, "late_90"] == 0


def test_input_is_not_modified():
    df = make_df([0.5, 0.6])
    df.loc[0, "late_90"] = 98
    before = df.copy()
    CreditCleaner().fit(df).transform(df)

    pd.testing.assert_frame_equal(df, before)