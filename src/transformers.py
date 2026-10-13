import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

LATE_COLS = ["late_30_59", "late_60_89", "late_90"]


class CreditCleaner(BaseEstimator, TransformerMixin):
    """Чистит данные кредитного скоринга.

    fit: запоминает верхние границы (квантиль) для столбцов с выбросами.
    transform: заменяет служебные коды 96/98 и невозможный возраст на пропуски,
               ограничивает выбросы сверху, добавляет два новых признака.
    """

    def __init__(self, cap_quantile=0.99, cap_cols=("utilization", "debt_ratio", "income")):
        self.cap_quantile = cap_quantile
        self.cap_cols = cap_cols

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        # запоминаем границы ТОЛЬКО по тем данным, что пришли в fit (train)
        self.caps_ = {c: X[c].quantile(self.cap_quantile) for c in self.cap_cols}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()  # копия, чтобы не портить исходную таблицу

        # служебные коды 96/98 -> пропуск
        for c in LATE_COLS:
            X.loc[X[c].isin([96, 98]), c] = np.nan

        # невозможный возраст -> пропуск
        X.loc[X["age"] < 18, "age"] = np.nan

        # выбросы: обрезаем сверху по границам, запомненным в fit
        for c, cap in self.caps_.items():
            X[c] = X[c].clip(upper=cap)

        # новые признаки
        X["total_late"] = X[LATE_COLS].sum(axis=1, min_count=1)
        X["income_missing"] = X["income"].isna().astype(int)
        return X