"""
Shared preprocessing pipeline for the Loan Default Detection project.

Contains the feature-engineering transform, column definitions, and the
preprocessor builder. This module is importable (unlike a `__main__` script),
so the pickled preprocessor can be re-loaded by any script (app, tests, etc.).
"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

NUMERICAL_COLS = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
]

CATEGORICAL_COLS = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]

ENGINEERED_COLS = [
    "MonthlyIncome",
    "LoanToIncome",
    "LoanToCreditScore",
    "InterestRatePerCreditPoint",
    "DTIInterestProduct",
    "CreditBurdenRatio",
    "YearsEmployed",
    "IncomePerCreditLine",
]


def engineer_numeric_features(df):
    """Derive risk-relevant financial features from the raw numeric columns."""
    eps = 1e-6
    out = df.copy()
    income = df["Income"]
    loan = df["LoanAmount"]
    cs = df["CreditScore"]
    ir = df["InterestRate"]
    dti = df["DTIRatio"]
    months = df["MonthsEmployed"]
    num_cl = df["NumCreditLines"]

    out["MonthlyIncome"] = income / 12.0
    out["LoanToIncome"] = loan / (income + eps)
    out["LoanToCreditScore"] = loan / (cs + eps)
    out["InterestRatePerCreditPoint"] = ir / (cs / 100.0)
    out["DTIInterestProduct"] = dti * ir
    out["CreditBurdenRatio"] = num_cl * dti
    out["YearsEmployed"] = months / 12.0
    out["IncomePerCreditLine"] = income / (num_cl + eps)
    return out


def engineered_feature_names(_self, _input_features):
    """Return the output names of the feature-engineering transform."""
    return NUMERICAL_COLS + ENGINEERED_COLS


def build_preprocessor():
    feat_eng = FunctionTransformer(
        engineer_numeric_features,
        feature_names_out=engineered_feature_names,
        validate=False,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("feat_eng", feat_eng),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERICAL_COLS,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
    )
    return preprocessor