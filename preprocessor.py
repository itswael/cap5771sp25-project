import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

categorical_features = [
        "released",
        "writer",
        "rating",
        "name",
        "genre",
        "director",
        "star",
        "country",
        "company",
    ]

numerical_features = [
        "runtime",
        "score",
        "year",
        "votes",
        "log_budget",
        "budget_vote_ratio",
        "budget_runtime_ratio",
        "budget_score_ratio",
        "vote_score_ratio",
        "budget_year_ratio",
        "vote_year_ratio",
        "score_runtime_ratio",
        "budget_per_minute",
        "votes_per_year",
        "is_recent",
        "is_high_budget",
        "is_high_votes",
        "is_high_score",
    ]

def perform_feature_engineering(df):
    # Feature engineering
    df["budget_vote_ratio"] = df["budget"] / (df["votes"] + 1)
    df["budget_runtime_ratio"] = df["budget"] / (df["runtime"] + 1)
    df["budget_score_ratio"] = df["log_budget"] / (df["score"] + 1)
    df["vote_score_ratio"] = df["votes"] / (df["score"] + 1)
    df["budget_year_ratio"] = df["log_budget"] / (df["year"] - df["year"].min() + 1)
    df["vote_year_ratio"] = df["votes"] / (df["year"] - df["year"].min() + 1)
    df["score_runtime_ratio"] = df["score"] / (df["runtime"] + 1)
    df["budget_per_minute"] = df["budget"] / (df["runtime"] + 1)
    df["votes_per_year"] = df["votes"] / (df["year"] - df["year"].min() + 1)
    df["is_recent"] = (df["year"] >= df["year"].quantile(0.75)).astype(int)
    df["is_high_budget"] = (df["log_budget"] >= df["log_budget"].quantile(0.75)).astype(int)
    df["is_high_votes"] = (df["votes"] >= df["votes"].quantile(0.75)).astype(int)
    df["is_high_score"] = (df["score"] >= df["score"].quantile(0.75)).astype(int)
    return df

def scale_gross_and_budget(df):
    if "gross" in df.columns:
        df["log_gross"] = np.log1p(df["gross"])
    if "budget" in df.columns:
        df["log_budget"] = np.log1p(df["budget"])

    return df

def apply_label_encoding(df):
    for feature in categorical_features:
        df[feature] = df[feature].astype(str)
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
    return df

def remove_unnecessary_columns(df):
    if "gross" in df.columns:
        df = df.drop(["gross"], axis=1)
    if "budget" in df.columns:
        df = df.drop(["budget"], axis=1)
    return df

