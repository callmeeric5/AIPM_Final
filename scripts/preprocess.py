from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import pandas as pd
from scripts.config import (
    CATEGORY_FEATURES,
    CONTINUOUS_FEATURES,
    ENCODER_PATH,
    SCALER_PATH,
)



def preprocess(
    df: pd.DataFrame,
    categorical_columns: list = CATEGORY_FEATURES,
    continuous_columns: list = CONTINUOUS_FEATURES,
    encoder_path: str = ENCODER_PATH,
    scaler_path: str = SCALER_PATH,
) -> pd.DataFrame:
    df = fill_features_nulls(df)
    df = scale_continuous(df, continuous_columns, scaler_path)
    df = encode_categorical(df, categorical_columns, encoder_path)

    return df


def make_encoder(
    df: pd.DataFrame, categorical_columns: list, path: str = ENCODER_PATH
) -> joblib:
    encoder = OneHotEncoder(handle_unknown="ignore", dtype=int)
    encoder.fit(df[categorical_columns])
    joblib.dump(encoder, path)


# Encode the categorial columns
def encode_categorical(
    df: pd.DataFrame, categorical_columns: list, path: str = ENCODER_PATH
) -> pd.DataFrame:
    encoder = joblib.load(path)
    encoded_columns = encoder.transform(df[categorical_columns])
    encoded_df = pd.DataFrame(
        encoded_columns.toarray(),
        columns=encoder.get_feature_names_out(categorical_columns),
    )
    df = pd.concat([encoded_df, df.reset_index(drop=True)], axis=1)
    df.drop(columns=categorical_columns, inplace=True)
    return df


def make_scaler(
    df: pd.DataFrame, continuous_columns: list, path: str = SCALER_PATH
) -> joblib:
    scaler = StandardScaler()
    scaler.fit(df[continuous_columns])
    joblib.dump(scaler, path)


def scale_continuous(
    df: pd.DataFrame, continuous_columns: list, path: str = SCALER_PATH
) -> pd.DataFrame:
    scaler = joblib.load(path)
    scaled_columns = scaler.transform(df[continuous_columns])
    scaled_df = pd.DataFrame(
        scaled_columns, columns=scaler.get_feature_names_out(continuous_columns)
    )
    df = pd.concat([scaled_df, df.reset_index(drop=True)], axis=1)

    return df


def fill_features_nulls(
    df: pd.DataFrame,
    categorical_columns: list = CATEGORY_FEATURES,
    continuous_columns: list = CONTINUOUS_FEATURES,
) -> pd.DataFrame:
    df[continuous_columns] = df[continuous_columns].fillna(
        df[continuous_columns].mean()
    )
    for feature in categorical_columns:
        df[feature] = df[feature].fillna(df[feature].mode()[0])

    return df
