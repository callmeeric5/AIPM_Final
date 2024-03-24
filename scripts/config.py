CATEGORY_FEATURES = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "PreferedOrderCat",
    "MaritalStatus",
    "Gender",
]
CONTINUOUS_FEATURES = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "NumberOfAddress",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
    "CityTier",
    "SatisfactionScore",
    "Complain",
]
ENCODER_PATH = "models/Encoder.joblib"
SCALER_PATH = "models/Scaler.joblib"
MODEL_PATH = "models/Model.joblib"
MLFLOW_PATH = "models/mlruns"
