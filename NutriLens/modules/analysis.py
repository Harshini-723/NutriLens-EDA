"""Reusable, defensive data analysis helpers for NutriLens."""
import numpy as np
import pandas as pd

def safe_numeric_columns(df):
    result = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # A column is numeric if its dtype is numeric, or if most non-empty values parse.
        non_empty = df[col].notna().sum()
        if pd.api.types.is_numeric_dtype(df[col]) or (non_empty and converted.notna().sum() / non_empty >= 0.8):
            result.append(col)
    return result

def safe_categorical_columns(df):
    nums = set(safe_numeric_columns(df))
    return [c for c in df.columns if c not in nums]

def profile_dataset(df):
    numeric = safe_numeric_columns(df)
    categorical = safe_categorical_columns(df)
    info = pd.DataFrame({
        "Column": list(df.columns),
        "Data type": [str(x) for x in df.dtypes],
        "Non-missing": [int(df[c].notna().sum()) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
        "Role": ["Numeric" if c in numeric else "Categorical" for c in df.columns]
    })
    return {"numeric": numeric, "categorical": categorical, "column_info": info}

def missing_summary(df):
    count = df.isna().sum()
    pct = (count / max(len(df), 1) * 100).round(2)
    return pd.DataFrame({"Column": count.index, "Missing values": count.values,
                         "Missing (%)": pct.values,
                         "Non-missing": df.notna().sum().values})

def numeric_summary(df):
    records = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        mode = s.mode()
        records.append({
            "Column": col, "Count": int(s.count()), "Mean": s.mean(),
            "Median": s.median(), "Mode": mode.iloc[0] if len(mode) else np.nan,
            "Variance": s.var(), "Std. deviation": s.std(),
            "Minimum": s.min(), "Q1 (25%)": s.quantile(.25),
            "Q2 (50%)": s.quantile(.50), "Q3 (75%)": s.quantile(.75),
            "Maximum": s.max(), "Range": s.max()-s.min()
        })
    return pd.DataFrame(records).set_index("Column") if records else pd.DataFrame()

def explain_distribution(values, column):
    s = pd.Series(values).dropna()
    if s.empty:
        return f"There are no usable values in {column} to describe."
    mean, median, std = s.mean(), s.median(), s.std()
    skew = s.skew() if len(s) > 2 else 0
    if abs(skew) < .5:
        shape = "fairly balanced around its centre"
    elif skew > 0:
        shape = "stretched more toward larger values"
    else:
        shape = "stretched more toward smaller values"
    spread = "quite spread out" if std > abs(mean) * .5 and mean != 0 else "relatively close together"
    return (f"For <b>{column}</b>, the average is {mean:,.2f} and the middle value is {median:,.2f}. "
            f"The values look {shape}. They are {spread} (standard deviation: {std:,.2f}). "
            f"The chart describes this dataset only; it does not explain why the values occur.")

def clean_dataframe(df, remove_duplicates=False, missing_action="Leave them unchanged", fill_value="Not provided"):
    result = df.copy()
    if remove_duplicates:
        result = result.drop_duplicates()
    if missing_action == "Drop rows containing any missing value":
        result = result.dropna()
    elif missing_action == "Fill numeric gaps with column median":
        for col in result.select_dtypes(include=np.number).columns:
            result[col] = result[col].fillna(result[col].median())
    elif missing_action == "Fill numeric gaps with column mean":
        for col in result.select_dtypes(include=np.number).columns:
            result[col] = result[col].fillna(result[col].mean())
    elif missing_action == "Fill all gaps with a fixed value":
        result = result.fillna(fill_value)
    return result

def scale_numeric(df, columns, method):
    result = df.copy()
    for col in columns:
        values = pd.to_numeric(result[col], errors="coerce")
        if method.startswith("Min"):
            minimum, maximum = values.min(), values.max()
            result[col] = 0.0 if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum else (values-minimum)/(maximum-minimum)
        else:
            mean, std = values.mean(), values.std()
            result[col] = 0.0 if pd.isna(std) or std == 0 else (values-mean)/std
    return result
