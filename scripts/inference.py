import pandas as pd
import numpy as np
from scripts.preprocess import preprocess
from joblib import load
from scripts.config import MODEL_PATH


def make_predictions(df: pd.DataFrame) -> np.ndarray:

    df = preprocess(df)
    model = load(MODEL_PATH)
    return model.predict(df)
