import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PowerTransformer
)

# =========================================================
# 4.2 DATA TRANSFORMATION
# =========================================================

# Get the folder where this Python file is located
folder = Path(__file__).resolve().parent

# Load cleaned dataset from the same folder
df = pd.read_csv(folder / "hotel_bookings_cleaned.csv")

print("=" * 65)
print("4.2 DATA TRANSFORMATION")
print("=" * 65)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)


# =========================================================
# 4.2.1 STANDARDISATION
# =========================================================

print("\n" + "=" * 65)
print("4.2.1 STANDARDISATION")
print("=" * 65)

standard_features = [
    'lead_time',
    'adr',
    'stays_in_week_nights',
    'total_of_special_requests'
]

print("\nOriginal Values:")
print(df[standard_features].head())

# Create a separate copy
df_standardised = df.copy()

# Create StandardScaler
standard_scaler = StandardScaler()

# Apply standardisation
df_standardised[standard_features] = (
    standard_scaler.fit_transform(
        df[standard_features]
    )
)

print("\nStandardised Values:")
print(df_standardised[standard_features].head())

print("\nMean After Standardisation:")
print(
    df_standardised[standard_features]
    .mean()
    .round(3)
)

print("\nStandard Deviation After Standardisation:")
print(
    df_standardised[standard_features]
    .std()
    .round(3)
)


# =========================================================
# 4.2.2 NORMALISATION
# =========================================================

print("\n" + "=" * 65)
print("4.2.2 NORMALISATION")
print("=" * 65)

normal_features = [
    'lead_time',
    'adr',
    'stays_in_week_nights',
    'total_of_special_requests'
]

print("\nOriginal Minimum Values:")
print(df[normal_features].min())

print("\nOriginal Maximum Values:")
print(df[normal_features].max())

# Create copy
df_normalised = df.copy()

# Create MinMaxScaler
minmax_scaler = MinMaxScaler()

# Apply normalisation
df_normalised[normal_features] = (
    minmax_scaler.fit_transform(
        df[normal_features]
    )
)

print("\nNormalised Sample:")
print(df_normalised[normal_features].head())

print("\nMinimum Values After Normalisation:")
print(
    df_normalised[normal_features]
    .min()
    .round(3)
)

print("\nMaximum Values After Normalisation:")
print(
    df_normalised[normal_features]
    .max()
    .round(3)
)


# =========================================================
# 4.2.3 LOG TRANSFORMATION
# =========================================================

print("\n" + "=" * 65)
print("4.2.3 LOG TRANSFORMATION")
print("=" * 65)

# Use only non-negative variables
log_features = [
    'lead_time',
    'days_in_waiting_list',
    'previous_cancellations'
]

print("\nSkewness Before Log Transformation:")
print(
    df[log_features]
    .skew()
    .round(3)
)

# Create copy
df_log = df.copy()

# Apply log(1 + x)
for column in log_features:
    df_log[column] = np.log1p(
        df_log[column]
    )

print("\nSample Values After Log Transformation:")
print(df_log[log_features].head())

print("\nSkewness After Log Transformation:")
print(
    df_log[log_features]
    .skew()
    .round(3)
)


# =========================================================
# 4.2.4 POWER TRANSFORMATION
# =========================================================

print("\n" + "=" * 65)
print("4.2.4 POWER TRANSFORMATION - YEO-JOHNSON")
print("=" * 65)

power_features = [
    'lead_time',
    'adr',
    'days_in_waiting_list'
]

print("\nSkewness Before Power Transformation:")
print(
    df[power_features]
    .skew()
    .round(3)
)

# Create copy
df_power = df.copy()

# Yeo-Johnson can handle zero and negative values
power_transformer = PowerTransformer(
    method='yeo-johnson',
    standardize=True
)

df_power[power_features] = (
    power_transformer.fit_transform(
        df[power_features]
    )
)

print("\nSample Values After Power Transformation:")
print(df_power[power_features].head())

print("\nSkewness After Power Transformation:")
print(
    df_power[power_features]
    .skew()
    .round(3)
)


# =========================================================
#  SUMMARY
# =========================================================

print("\n" + "=" * 65)
print("4.2 TRANSFORMATION SUMMARY")
print("=" * 65)

print("""
Standardisation:
- Changes features to approximately mean = 0
  and standard deviation = 1.
- Useful for Logistic Regression and K-means.

Normalisation:
- Rescales values between 0 and 1.
- Sensitive to extreme values.

Log Transformation:
- Reduces positive skewness.
- Suitable for non-negative skewed variables.

Power Transformation:
- Uses Yeo-Johnson transformation.
- Can handle positive, zero and negative values.
""")


# =========================================================
# SAVE TRANSFORMED DATASETS
# =========================================================

df_standardised.to_csv(
    folder / "hotel_bookings_standardised.csv",
    index=False
)

df_normalised.to_csv(
    folder / "hotel_bookings_normalised.csv",
    index=False
)

df_log.to_csv(
    folder / "hotel_bookings_log_transformed.csv",
    index=False
)

df_power.to_csv(
    folder / "hotel_bookings_power_transformed.csv",
    index=False
)

print("\nTransformation files saved successfully:")
print("- hotel_bookings_standardised.csv")
print("- hotel_bookings_normalised.csv")
print("- hotel_bookings_log_transformed.csv")
print("- hotel_bookings_power_transformed.csv")