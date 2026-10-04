import pandas as pd
from pathlib import Path

# =========================================================
# 4.1 DATA CLEANING
# =========================================================

# Get folder where this Python file is located
folder = Path(__file__).resolve().parent

# Load dataset from same folder as this Python file
df = pd.read_csv(folder / "hotel_bookings.csv")

print("=" * 55)
print("4.1 DATA CLEANING")
print("=" * 55)

print("\nOriginal Dataset Shape:")
print(df.shape)


# =========================================================
# 4.1.1 MISSING VALUE IMPUTATION
# =========================================================

columns = ['children', 'country', 'agent', 'company']

print("\n4.1.1 Missing Values Before Imputation")
print("-" * 45)
print(df[columns].isnull().sum())

# children: replace missing values with median
df['children'] = df['children'].fillna(
    df['children'].median()
)

# country: replace missing values with Unknown
df['country'] = df['country'].fillna(
    'Unknown'
)

# agent and company: treat identifiers as categorical
df['agent'] = (
    df['agent']
    .astype('string')
    .fillna('No Agent')
)

df['company'] = (
    df['company']
    .astype('string')
    .fillna('No Company')
)

print("\nMissing Values After Imputation")
print("-" * 45)
print(df[columns].isnull().sum())

print("\nSample Data After Imputation")
print("-" * 45)
print(df[columns].head(10))


# =========================================================
# 4.1.2 DUPLICATE REMOVAL
# =========================================================

print("\n4.1.2 Duplicate Removal")
print("-" * 45)

duplicate_count = df.duplicated().sum()

print(
    "Duplicate rows before removal:",
    duplicate_count
)

# Remove exact duplicate rows
df = df.drop_duplicates().copy()

print(
    "Duplicate rows after removal:",
    df.duplicated().sum()
)

print(
    "Dataset shape after duplicate removal:",
    df.shape
)


# =========================================================
# 4.1.3 NOISE AND OUTLIER TREATMENT
# =========================================================

print("\n4.1.3 Noise and Outlier Treatment")
print("-" * 45)

# ---------------------------------------------------------
# Check zero-guest bookings
# ---------------------------------------------------------

zero_guest = (
    (df['adults'] == 0) &
    (df['children'] == 0) &
    (df['babies'] == 0)
)

print(
    "Bookings with zero guests:",
    zero_guest.sum()
)

# Remove zero-guest bookings
df = df.loc[~zero_guest].copy()

remaining_zero_guest = (
    (df['adults'] == 0) &
    (df['children'] == 0) &
    (df['babies'] == 0)
).sum()

print(
    "Zero-guest bookings after removal:",
    remaining_zero_guest
)


# ---------------------------------------------------------
# Outlier detection using IQR
# ---------------------------------------------------------

Q1 = df['adr'].quantile(0.25)
Q3 = df['adr'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

adr_outliers = df[
    (df['adr'] < lower_bound) |
    (df['adr'] > upper_bound)
]

print("\nADR Outlier Detection Using IQR")
print("-" * 45)

print("Q1:", round(Q1, 2))
print("Q3:", round(Q3, 2))
print("IQR:", round(IQR, 2))

print(
    "Lower Bound:",
    round(lower_bound, 2)
)

print(
    "Upper Bound:",
    round(upper_bound, 2)
)

print(
    "Potential ADR outliers:",
    len(adr_outliers)
)

print("\nExample ADR Outliers")
print("-" * 45)

print(
    adr_outliers[
        ['hotel', 'lead_time', 'adults', 'adr']
    ].head(10)
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 55)
print("FINAL DATA CLEANING SUMMARY")
print("=" * 55)

print(
    "Final Dataset Shape:",
    df.shape
)

print(
    "Total Missing Values:",
    df.isnull().sum().sum()
)

print(
    "Remaining Duplicate Rows:",
    df.duplicated().sum()
)

# Save cleaned dataset in same folder
output_file = (
    folder / "hotel_bookings_cleaned.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nCleaned dataset saved successfully:")
print(output_file)