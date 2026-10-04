import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# =========================================================
# 4.4 DATA BALANCING
# =========================================================

# Get folder containing this Python file
folder = Path(__file__).resolve().parent

# Load engineered dataset produced in Section 4.3
df = pd.read_csv(
    folder / "hotel_bookings_engineered.csv"
)

print("=" * 70)
print("4.4 DATA BALANCING")
print("=" * 70)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)


# =========================================================
# TARGET AND PREDICTORS
# =========================================================

# Target variable
y = df['is_canceled']

# Predictor variables
X = df.drop(
    columns=['is_canceled']
)

print("\nOriginal Target Distribution")
print("-" * 45)

print(y.value_counts())

print("\nOriginal Target Percentage")
print(
    (y.value_counts(normalize=True) * 100)
    .round(2)
)


# =========================================================
# TRAIN-TEST SPLIT
# =========================================================

# IMPORTANT:
# Split the data BEFORE balancing to avoid data leakage.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

print("\nTraining Target Distribution Before Balancing")
print("-" * 45)
print(y_train.value_counts())

print("\nTraining Target Percentage Before Balancing")
print(
    (y_train.value_counts(normalize=True) * 100)
    .round(2)
)


# =========================================================
# PREPARE CATEGORICAL AND NUMERICAL FEATURES
# =========================================================

categorical_features = (
    X_train
    .select_dtypes(
        include=['object', 'string']
    )
    .columns
    .tolist()
)

numerical_features = (
    X_train
    .select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)

print("\nCategorical features:",
      len(categorical_features))

print("Numerical features:",
      len(numerical_features))


# =========================================================
# PREPROCESSING
# =========================================================

# One-hot encode categorical variables
# Standardise numerical variables
preprocessor = ColumnTransformer(
    transformers=[
        (
            'categorical',
            OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=True
            ),
            categorical_features
        ),
        (
            'numerical',
            StandardScaler(),
            numerical_features
        )
    ]
)

# Fit ONLY on training data
X_train_processed = (
    preprocessor.fit_transform(X_train)
)

# Apply same transformation to test data
X_test_processed = (
    preprocessor.transform(X_test)
)

print("\nProcessed training shape:",
      X_train_processed.shape)

print("Processed testing shape:",
      X_test_processed.shape)


# =========================================================
# 4.4.1 SMOTE
# =========================================================

print("\n" + "=" * 70)
print("4.4.1 SMOTE")
print("=" * 70)

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = (
    smote.fit_resample(
        X_train_processed,
        y_train
    )
)

print("\nClass Distribution After SMOTE")
print("-" * 45)
print(y_train_smote.value_counts())

print("\nClass Percentage After SMOTE")
print(
    (
        y_train_smote
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

print("\nTraining shape before SMOTE:",
      X_train_processed.shape)

print("Training shape after SMOTE:",
      X_train_smote.shape)


# =========================================================
# 4.4.2 RANDOM OVER-SAMPLING
# =========================================================

print("\n" + "=" * 70)
print("4.4.2 RANDOM OVER-SAMPLING")
print("=" * 70)

ros = RandomOverSampler(
    random_state=42
)

X_train_ros, y_train_ros = (
    ros.fit_resample(
        X_train_processed,
        y_train
    )
)

print("\nClass Distribution After Random Over-Sampling")
print("-" * 45)
print(y_train_ros.value_counts())

print("\nClass Percentage After Random Over-Sampling")
print(
    (
        y_train_ros
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

print("\nTraining shape before Over-Sampling:",
      X_train_processed.shape)

print("Training shape after Over-Sampling:",
      X_train_ros.shape)


# =========================================================
# 4.4.3 RANDOM UNDER-SAMPLING
# =========================================================

print("\n" + "=" * 70)
print("4.4.3 RANDOM UNDER-SAMPLING")
print("=" * 70)

rus = RandomUnderSampler(
    random_state=42
)

X_train_rus, y_train_rus = (
    rus.fit_resample(
        X_train_processed,
        y_train
    )
)

print("\nClass Distribution After Random Under-Sampling")
print("-" * 45)
print(y_train_rus.value_counts())

print("\nClass Percentage After Random Under-Sampling")
print(
    (
        y_train_rus
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

print("\nTraining shape before Under-Sampling:",
      X_train_processed.shape)

print("Training shape after Under-Sampling:",
      X_train_rus.shape)


# =========================================================
# 4.4 COMPARISON SUMMARY
# =========================================================

summary = pd.DataFrame({
    'Method': [
        'Original Training Data',
        'SMOTE',
        'Random Over-Sampling',
        'Random Under-Sampling'
    ],

    'Class 0': [
        (y_train == 0).sum(),
        (y_train_smote == 0).sum(),
        (y_train_ros == 0).sum(),
        (y_train_rus == 0).sum()
    ],

    'Class 1': [
        (y_train == 1).sum(),
        (y_train_smote == 1).sum(),
        (y_train_ros == 1).sum(),
        (y_train_rus == 1).sum()
    ],

    'Total Training Records': [
        len(y_train),
        len(y_train_smote),
        len(y_train_ros),
        len(y_train_rus)
    ]
})

print("\n" + "=" * 70)
print("4.4 DATA BALANCING COMPARISON")
print("=" * 70)

print(summary.to_string(index=False))


# =========================================================
# SUMMARY
# =========================================================

summary.to_csv(
    folder / "data_balancing_summary.csv",
    index=False
)

print("\nSummary saved as:")
print("data_balancing_summary.csv")


