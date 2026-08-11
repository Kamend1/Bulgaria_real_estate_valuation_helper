"""
TF-IDF + SVD text features on listings.description_clean, for the segments
where Round 3 found a real accuracy gain (see ROUND3_findings.md):
residential, retail, industrial, hospitality. Office is deliberately
excluded — no measurable gain there.

Only TF-IDF+SVD is productionized. FastText/sentence-transformer embeddings
were tested and rejected (MiniLM lost to TF-IDF on every tested segment
despite costing far more compute) — see the Round 3 findings for why.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

TEXT_FEATURE_PREFIX = "txt_tfidf_"
N_COMPONENTS = 15


def fit_tfidf_svd(texts_train: pd.Series, n_components: int = N_COMPONENTS) -> dict:
    """Fit on train rows only — never on val/test, avoids leakage."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=5)
    tfidf = vectorizer.fit_transform(texts_train.fillna(""))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(tfidf)
    return {"vectorizer": vectorizer, "svd": svd}


def text_feature_columns(fitted: dict) -> list[str]:
    n = fitted["svd"].n_components
    return [f"{TEXT_FEATURE_PREFIX}{i}" for i in range(n)]


def transform_tfidf_svd(texts: pd.Series, fitted: dict) -> pd.DataFrame:
    """Returns a DataFrame with the fitted transformer's column names,
    indexed to match `texts` — caller is responsible for aligning/resetting
    the index before concatenating with other feature columns."""
    arr = fitted["svd"].transform(fitted["vectorizer"].transform(texts.fillna("")))
    return pd.DataFrame(arr, columns=text_feature_columns(fitted), index=texts.index)
