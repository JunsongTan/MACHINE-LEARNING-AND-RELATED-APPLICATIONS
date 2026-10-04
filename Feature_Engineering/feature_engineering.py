import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# 4.3 FEATURE ENGINEERING
# =========================================================

# Get current Python file folder
folder = Path(__file__).resolve().parent

# Load cleaned dataset from Section 4.1
df = pd.read_csv(folder / "hotel_bookings_cleaned.csv")

print("=" * 70)
print("4.3 FEATURE ENGINEERING")
print("=" * 70)

print("\nDataset loaded successfully.")
print("Original dataset shape:", df.shape)


# =========================================================
# 4.3.1 ENCODING CATEGORICAL VARIABLES
# =========================================================

print("\n" + "=" * 70)
print("4.3.1 ENCODING CATEGORICAL VARIABLES")
print("=" * 70)

# Use mainly low-cardinality categorical variables
categorical_features = [
    'hotel',
    'meal',
    'market_segment',
    'distribution_channel',
    'deposit_type',
    'customer_type'
]

print("\nCategorical Features:")
print(categorical_features)

print("\nNumber of Unique Categories:")
for col in categorical_features:
    print(f"{col}: {df[col].nunique()}")

# Create encoder
encoder = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

# Apply one-hot encoding
encoded_array = encoder.fit_transform(
    df[categorical_features]
)

# Get new column names
encoded_columns = encoder.get_feature_names_out(
    categorical_features
)

# Convert encoded result to DataFrame
encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoded_columns,
    index=df.index
)

print("\nOriginal categorical variables:",
      len(categorical_features))

print("Encoded variables:",
      encoded_df.shape[1])

print("\nSample Encoded Data:")
print(encoded_df.head())

# Create dataset with categorical columns replaced by encoded columns
df_encoded = pd.concat(
    [
        df.drop(columns=categorical_features),
        encoded_df
    ],
    axis=1
)

print("\nDataset shape after encoding:",
      df_encoded.shape)


# =========================================================
# 4.3.2 FEATURE SELECTION
# =========================================================

print("\n" + "=" * 70)
print("4.3.2 FEATURE SELECTION")
print("=" * 70)

print("\nFeatures before selection:",
      df.shape[1])

# Remove target-leakage and low-value attributes
drop_features = [
    'reservation_status',
    'reservation_status_date',
    'company'
]

print("\nFeatures selected for removal:")
for feature in drop_features:
    print("-", feature)

# Remove selected attributes
df_selected = df.drop(
    columns=drop_features,
    errors='ignore'
).copy()

print("\nFeatures after selection:",
      df_selected.shape[1])

print("\nRemoved features:")
print(
    [
        col for col in drop_features
        if col not in df_selected.columns
    ]
)

print("\nRemaining dataset shape:",
      df_selected.shape)


# =========================================================
# 4.3.3 FEATURE EXTRACTION
# =========================================================

print("\n" + "=" * 70)
print("4.3.3 FEATURE EXTRACTION")
print("=" * 70)

# Start from selected dataset
df_engineered = df_selected.copy()

# Total number of nights
df_engineered['total_stay'] = (
    df_engineered['stays_in_week_nights'] +
    df_engineered['stays_in_weekend_nights']
)

# Total guests
df_engineered['total_guests'] = (
    df_engineered['adults'] +
    df_engineered['children'] +
    df_engineered['babies']
)

# Total previous booking history
df_engineered['previous_booking_total'] = (
    df_engineered['previous_cancellations'] +
    df_engineered['previous_bookings_not_canceled']
)

print("\nNew Features Created:")
print("- total_stay")
print("- total_guests")
print("- previous_booking_total")

print("\nSample Extracted Features:")
print(
    df_engineered[
        [
            'stays_in_week_nights',
            'stays_in_weekend_nights',
            'total_stay',
            'adults',
            'children',
            'babies',
            'total_guests',
            'previous_cancellations',
            'previous_bookings_not_canceled',
            'previous_booking_total'
        ]
    ].head(10)
)

print("\nDataset shape after feature extraction:",
      df_engineered.shape)


# =========================================================
# 4.3.4 DIMENSIONALITY REDUCTION USING PCA
# =========================================================

print("\n" + "=" * 70)
print("4.3.4 DIMENSIONALITY REDUCTION USING PCA")
print("=" * 70)

# Select numerical variables only
numeric_data = df_engineered.select_dtypes(
    include=[np.number]
).copy()

# Remove target variable from PCA
numeric_data = numeric_data.drop(
    columns=['is_canceled'],
    errors='ignore'
)

# Remove columns containing missing values if any remain
numeric_data = numeric_data.dropna(
    axis=1
)

print("\nNumerical features before PCA:",
      numeric_data.shape[1])

# Standardise before PCA
scaler = StandardScaler()

scaled_numeric = scaler.fit_transform(
    numeric_data
)

# Keep enough components to explain 95% variance
pca = PCA(
    n_components=0.95
)

pca_data = pca.fit_transform(
    scaled_numeric
)

print("Principal components retained:",
      pca_data.shape[1])

print(
    "Total explained variance:",
    round(
        pca.explained_variance_ratio_.sum(),
        4
    )
)

print("\nExplained Variance by Component:")

for i, variance in enumerate(
    pca.explained_variance_ratio_,
    start=1
):
    print(
        f"PC{i}: {variance:.4f}"
    )

# Create PCA DataFrame
pca_columns = [
    f'PC{i+1}'
    for i in range(pca_data.shape[1])
]

pca_df = pd.DataFrame(
    pca_data,
    columns=pca_columns
)

print("\nSample PCA Data:")
print(pca_df.head())


# =========================================================
# 4.3 SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("4.3 FEATURE ENGINEERING SUMMARY")
print("=" * 70)

print(f"""
Encoding:
- {len(categorical_features)} categorical features were selected.
- One-hot encoding produced {encoded_df.shape[1]} binary features.

Feature Selection:
- reservation_status removed because of target leakage.
- reservation_status_date removed because of target leakage.
- company removed because it is mainly an identifier and originally
  contained very high missingness.

Feature Extraction:
- total_stay created from weekday + weekend nights.
- total_guests created from adults + children + babies.
- previous_booking_total created from previous booking history.

PCA:
- Original numerical features: {numeric_data.shape[1]}
- Components retained: {pca_data.shape[1]}
- Explained variance retained:
  {pca.explained_variance_ratio_.sum():.4f}
""")


# =========================================================
# SAVE OUTPUT FILES
# =========================================================

df_encoded.to_csv(
    folder / "hotel_bookings_encoded.csv",
    index=False
)

df_selected.to_csv(
    folder / "hotel_bookings_selected.csv",
    index=False
)

df_engineered.to_csv(
    folder / "hotel_bookings_engineered.csv",
    index=False
)

pca_df.to_csv(
    folder / "hotel_bookings_pca.csv",
    index=False
)

print("\nFiles saved successfully:")
print("- hotel_bookings_encoded.csv")
print("- hotel_bookings_selected.csv")
print("- hotel_bookings_engineered.csv")
print("- hotel_bookings_pca.csv")