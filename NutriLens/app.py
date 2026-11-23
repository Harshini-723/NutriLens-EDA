import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from modules.analysis import (
    profile_dataset,
    missing_summary,
    numeric_summary,
    explain_distribution,
    clean_dataframe,
    scale_numeric,
    safe_numeric_columns,
    safe_categorical_columns
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NutriLens | Food & Nutrition Insight Explorer",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SAMPLE DATASET LOADER - FIXED
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def load_sample_dataset():
    possible_paths = [
        BASE_DIR / "data" / "sample_food_data.csv",
        BASE_DIR / "Data" / "sample_food_data.csv",
        BASE_DIR.parent / "data" / "sample_food_data.csv",
        BASE_DIR / "sample_food_data.csv",
        Path.cwd() / "data" / "sample_food_data.csv",
    ]

    for file_path in possible_paths:
        if file_path.is_file():
            df = pd.read_csv(file_path)

            if df.empty:
                raise ValueError("The sample dataset is empty.")

            return df

    checked = "\n".join(str(path) for path in possible_paths)

    raise FileNotFoundError(
        "sample_food_data.csv was not found. Checked:\n" + checked
    )


# ============================================================
# VISUAL SYSTEM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --ink: #182b28;
    --muted: #667873;
    --green: #176b52;
    --mint: #e9f5ef;
    --lime: #d7f36b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(180deg,#f7faf7 0%,#f3f7f4 100%);
    color: var(--ink);
}

[data-testid="stSidebar"] {
    background: #102e27;
}

[data-testid="stSidebar"] * {
    color: #eef8f1 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 7px 10px;
    border-radius: 9px;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #17352d;
    letter-spacing: -.03em;
}

h1 {
    font-weight: 800 !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1440px;
}

.hero {
    background: linear-gradient(120deg,#123d32,#1b7055);
    color: white;
    border-radius: 24px;
    padding: 34px 38px;
    margin-bottom: 22px;
    overflow: hidden;
}

.hero h1 {
    color: white !important;
    font-size: 2.45rem;
    margin: 5px 0 10px;
}

.hero p {
    color: #d7eee2;
    max-width: 720px;
    font-size: 1.03rem;
    line-height: 1.7;
}

.eyebrow {
    color: #d7f36b;
    text-transform: uppercase;
    letter-spacing: .15em;
    font-size: .72rem;
    font-weight: 800;
}

.metric-card {
    background: white;
    border: 1px solid #e3ece6;
    border-radius: 17px;
    padding: 18px 20px;
    box-shadow: 0 5px 18px rgba(27,65,49,.045);
    min-height: 112px;
}

.metric-label {
    color: #71817a;
    font-size: .78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .06em;
}

.metric-value {
    color: #173d31;
    font-size: 1.85rem;
    font-weight: 800;
    margin-top: 7px;
}

.metric-note {
    color: #8a9891;
    font-size: .78rem;
    margin-top: 2px;
}

.insight {
    background: #edf6ef;
    border-left: 4px solid #3d9b70;
    border-radius: 10px;
    padding: 14px 17px;
    color: #244a3a;
    margin: 10px 0 18px;
}

.section-kicker {
    color: #398364;
    text-transform: uppercase;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .13em;
}

.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    font-weight: 700;
}

[data-testid="stDataFrame"] {
    border: 1px solid #e5ece7;
    border-radius: 12px;
}

hr {
    border-color: #e2ebe5;
}

.small-muted {
    color: #73827b;
    font-size: .88rem;
}
</style>
""", unsafe_allow_html=True)

sns.set_theme(
    style="whitegrid",
    palette=["#26765a", "#a6c95b", "#e5a65c", "#6c9ca0"]
)


# ============================================================
# NAVIGATION
# ============================================================

PAGES = [
    "Home",
    "Dataset Upload",
    "Dataset Overview",
    "Data Quality",
    "Data Cleaning",
    "Statistical Analysis",
    "Distribution Analysis",
    "Category Analysis",
    "Scaling & Standardization",
    "Export Data"
]


# ============================================================
# SESSION STATE
# ============================================================

if "df_original" not in st.session_state:
    st.session_state.df_original = None

if "df_working" not in st.session_state:
    st.session_state.df_working = None

if "source_name" not in st.session_state:
    st.session_state.source_name = "No dataset selected"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def set_dataset(df, name):
    st.session_state.df_original = df.copy()
    st.session_state.df_working = df.copy()
    st.session_state.source_name = name


def current_df():
    return st.session_state.df_working


def require_data():
    df = current_df()

    if df is None:
        st.warning(
            "No dataset is loaded yet. Open **Dataset Upload** "
            "and choose the sample dataset or upload a CSV."
        )
        return None

    if df.empty:
        st.error("This dataset has no rows.")
        return None

    return df


def insight(message):
    st.markdown(
        f'<div class="insight"><b>💡 What does this mean?</b><br>'
        f'{message}</div>',
        unsafe_allow_html=True
    )


def metric(label, value, note=""):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def plot_to_streamlit(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🥗 NutriLens")
    st.caption("🍎 Food & Nutrition Insight Explorer")

    st.markdown("---")

    page = st.radio(
        "WORKSPACE",
        PAGES,
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown("**🍽️ ACTIVE DATASET**")
    st.caption(st.session_state.source_name)

    if current_df() is not None:
        st.caption(
            f"📊 {len(current_df()):,} rows · "
            f"{len(current_df().columns):,} columns"
        )

    st.markdown("---")
    st.caption("🥗 Eat Smart • Explore Data • Discover Insights")


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    home_hero = """
<div class="hero">
<div class="eyebrow">🥗 Explore • Understand • Improve</div>
<h1>Welcome to NutriLens! 🥑</h1>
<p>
Turn food and nutrition data into meaningful insights.
Explore nutrients, discover patterns and understand your dataset
with simple visualizations and explanations.
</p>
</div>
"""

    st.markdown(home_hero, unsafe_allow_html=True)

    st.markdown("### 🥕 Your data, explained simply")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric(
            "🥗 Explore",
            "01",
            "Understand rows, columns and data types"
        )

    with c2:
        metric(
            "🍎 Improve",
            "02",
            "Find missing values and clean your data"
        )

    with c3:
        metric(
            "🥑 Discover",
            "03",
            "Explore patterns with statistics and charts"
        )

    st.markdown("---")

    a, b = st.columns([1.15, .85])

    with a:

        st.markdown("#### 🍽️ Get started")

        if st.button(
            "🥗 Explore the built-in food dataset",
            type="primary",
            use_container_width=True
        ):

            try:
                sample = load_sample_dataset()

                set_dataset(
                    sample,
                    "Built-in food & nutrition sample"
                )

                st.success("🥑 Sample dataset loaded successfully!")

                st.rerun()

            except Exception as e:
                st.error(f"Could not load the sample dataset: {e}")

        st.caption("📁 Or choose Dataset Upload to use your own CSV.")


# ============================================================
# DATASET UPLOAD
# ============================================================

elif page == "Dataset Upload":

    st.markdown(
        '<div class="section-kicker">🥗 Start here</div>',
        unsafe_allow_html=True
    )

    st.title("📁 Choose your dataset")

    st.write(
        "Use the sample data to learn, or upload a CSV file from your device."
    )

    left, right = st.columns(2)

    with left:

        st.markdown("#### 🥗 Built-in sample")

        st.write(
            "A food and nutrition table with food names, categories "
            "and nutrient values."
        )

        if st.button(
            "🍎 Load sample dataset",
            type="primary",
            use_container_width=True
        ):

            try:
                sample = load_sample_dataset()

                set_dataset(
                    sample,
                    "Built-in food & nutrition sample"
                )

                st.success(
                    f"Loaded {len(sample):,} rows and "
                    f"{len(sample.columns)} columns."
                )

                st.rerun()

            except Exception as e:
                st.error(f"Sample data could not be loaded: {e}")

    with right:

        st.markdown("#### 📂 Upload your own CSV")

        uploaded = st.file_uploader(
            "Choose a .csv file",
            type=["csv"],
            help="UTF-8 CSV files are recommended."
        )

        if uploaded is not None:

            try:

                if uploaded.size == 0:

                    st.error("The selected file is empty.")

                else:

                    df = pd.read_csv(uploaded)

                    if len(df.columns) == 0:

                        st.error("No columns were found.")

                    elif df.empty:

                        st.error(
                            "The CSV has column headings but no data rows."
                        )

                    else:

                        set_dataset(df, uploaded.name)

                        st.success(
                            f"Successfully loaded {len(df):,} rows "
                            f"and {len(df.columns)} columns."
                        )

            except UnicodeDecodeError:

                st.error(
                    "The file encoding could not be read. "
                    "Save it as UTF-8 CSV and try again."
                )

            except pd.errors.ParserError as e:

                st.error(f"CSV structure error: {e}")

            except Exception as e:

                st.error(f"Unable to read this CSV: {e}")

    if current_df() is not None:

        st.markdown("---")

        st.markdown("#### 👀 Current dataset preview")

        st.dataframe(
            current_df().head(10),
            use_container_width=True
        )

        st.caption(
            "Loading a new dataset replaces the active working copy. "
            "Your original file is not changed."
        )


# ============================================================
# DATASET OVERVIEW
# ============================================================

elif page == "Dataset Overview":

    st.title("📊 Dataset overview")

    st.write("Get a quick map of what is inside your data.")

    df = require_data()

    if df is not None:

        profile = profile_dataset(df)

        cols = st.columns(4)

        with cols[0]:

            metric(
                "📋 Rows",
                f"{len(df):,}",
                "Records in dataset"
            )

        with cols[1]:

            metric(
                "📑 Columns",
                f"{len(df.columns):,}",
                "Data fields"
            )

        with cols[2]:

            metric(
                "🔢 Numeric fields",
                str(len(profile["numeric"])),
                "Numeric variables"
            )

        with cols[3]:

            metric(
                "🏷️ Categorical fields",
                str(len(profile["categorical"])),
                "Categorical variables"
            )

        st.markdown("### 👀 First look")

        st.dataframe(
            df.head(12),
            use_container_width=True
        )

        st.markdown("### 📝 Column guide")

        st.dataframe(
            profile["column_info"],
            use_container_width=True,
            hide_index=True
        )

        insight(
            f"Your dataset contains <b>{len(df):,} rows</b> and "
            f"<b>{len(df.columns)} columns</b>. "
            f"NutriLens detected {len(profile['numeric'])} numeric and "
            f"{len(profile['categorical'])} categorical columns."
        )

        st.markdown("### 📈 Quick visual map")

        if profile["numeric"]:

            selected = st.selectbox(
                "Choose a numeric column",
                profile["numeric"]
            )

            values = pd.to_numeric(
                df[selected],
                errors="coerce"
            ).dropna()

            if not values.empty:

                fig, ax = plt.subplots(figsize=(8, 3.4))

                sns.histplot(
                    values,
                    kde=len(values) > 2,
                    ax=ax,
                    color="#26765a"
                )

                ax.set_title(f"Values in {selected}")
                ax.set_xlabel(selected)

                plot_to_streamlit(fig)

            else:

                st.info("No valid numeric values in this column.")

        else:

            st.info("No numeric columns were detected.")


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.title("🔍 Data quality check")

    st.write(
        "Find gaps, repeated records and possible data-type issues."
    )

    df = require_data()

    if df is not None:

        miss = missing_summary(df)

        dup = int(df.duplicated().sum())

        total_missing = int(df.isna().sum().sum())

        a, b, c = st.columns(3)

        with a:

            metric(
                "⚠️ Missing cells",
                f"{total_missing:,}",
                "Blank or unavailable values"
            )

        with b:

            metric(
                "🔁 Duplicate rows",
                f"{dup:,}",
                "Rows repeated exactly"
            )

        with c:

            metric(
                "✅ Complete cells",
                f"{df.size - total_missing:,}",
                "Cells containing a value"
            )

        st.markdown("### 📋 Missing values by column")

        st.dataframe(
            miss,
            use_container_width=True,
            hide_index=True
        )

        if total_missing:

            ms = df.isna().sum().sort_values(ascending=False)

            ms = ms[ms > 0]

            fig, ax = plt.subplots(figsize=(8, 3.2))

            sns.barplot(
                x=ms.values,
                y=ms.index,
                ax=ax
            )

            ax.set_xlabel("Missing cells")
            ax.set_ylabel("")

            plot_to_streamlit(fig)

            insight(
                f"There are <b>{total_missing}</b> empty cells. "
                "Consider removing affected rows or filling gaps "
                "with a suitable value."
            )

        else:

            insight("Great — no missing cells were found.")

        st.markdown("### 🔁 Duplicate records")

        if dup:

            st.warning(f"{dup} exact duplicate rows were found.")

            st.dataframe(
                df[df.duplicated(keep=False)].head(20),
                use_container_width=True
            )

        else:

            st.success("No exact duplicate rows were found.")

        insight(
            f"NutriLens found {dup} duplicate rows. "
            "Check whether these are accidental repeats before removing them."
        )


# ============================================================
# DATA CLEANING
# ============================================================

elif page == "Data Cleaning":

    st.title("🧹 Clean your dataset")

    st.write(
        "Make changes to a separate working copy. "
        "Your original dataset stays unchanged."
    )

    df = require_data()

    if df is not None:

        st.markdown("### 🛠️ Choose cleaning actions")

        remove_dupes = st.checkbox(
            "🔁 Remove exact duplicate rows",
            value=False
        )

        missing_action = st.selectbox(
            "How should missing values be handled?",
            [
                "Leave them unchanged",
                "Drop rows containing any missing value",
                "Fill numeric gaps with column median",
                "Fill numeric gaps with column mean",
                "Fill all gaps with a fixed value"
            ]
        )

        fill_value = st.text_input(
            "Fixed fill value (used only for the last option)",
            value="Not provided"
        )

        keep_cols = st.multiselect(
            "Keep selected columns (leave empty to keep all)",
            list(df.columns),
            default=list(df.columns)
        )

        if st.button("🧹 Apply selected cleaning", type="primary"):

            if not keep_cols:

                st.error("Select at least one column to keep.")

            else:

                cleaned = df.loc[:, keep_cols].copy()

                cleaned = clean_dataframe(
                    cleaned,
                    remove_duplicates=remove_dupes,
                    missing_action=missing_action,
                    fill_value=fill_value
                )

                st.session_state.df_working = cleaned

                st.success(
                    f"Cleaning applied. Working dataset now has "
                    f"{len(cleaned):,} rows and "
                    f"{len(cleaned.columns)} columns."
                )

                st.rerun()

        st.markdown("### 👀 Current working data")

        st.dataframe(
            df.head(15),
            use_container_width=True
        )

        st.caption(
            f"{len(df):,} rows × {len(df.columns):,} columns"
        )

        if st.button("↩️ Restore original dataset"):

            st.session_state.df_working = (
                st.session_state.df_original.copy()
            )

            st.success("Working copy restored to the original dataset.")

            st.rerun()

        insight(
            "Cleaning prepares data for analysis. Removing duplicates "
            "can prevent repeated records from being counted twice. "
            "Filling gaps requires careful consideration."
        )


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

elif page == "Statistical Analysis":

    st.title("📊 Statistical analysis")

    st.write(
        "Understand typical values and how much your numbers vary."
    )

    df = require_data()

    if df is not None:

        nums = safe_numeric_columns(df)

        if not nums:

            st.info("No usable numeric columns were detected.")

        else:

            selected = st.multiselect(
                "Choose numeric columns",
                nums,
                default=nums[:min(3, len(nums))]
            )

            if selected:

                summary = numeric_summary(df[selected])

                st.dataframe(
                    summary.style.format(precision=2),
                    use_container_width=True
                )

                insight(
                    "Mean is the arithmetic average. Median is the "
                    "middle value. Mode is the most common value. "
                    "Standard deviation describes the spread of values."
                )

                col = st.selectbox(
                    "Explore one column in more detail",
                    selected
                )

                values = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).dropna()

                if not values.empty:

                    q1, median, q3 = values.quantile([.25, .5, .75])

                    st.markdown(
                        f"**{col}:** average **{values.mean():,.2f}**, "
                        f"median **{median:,.2f}**, "
                        f"range **{values.min():,.2f} to "
                        f"{values.max():,.2f}**."
                    )

                    insight(
                        f"The middle half of {col} values lies between "
                        f"{q1:,.2f} and {q3:,.2f}."
                    )

                else:

                    st.info("This column has no valid numeric values.")

            else:

                st.info(
                    "Choose at least one numeric column to see its summary."
                )


# ============================================================
# DISTRIBUTION ANALYSIS
# ============================================================

elif page == "Distribution Analysis":

    st.title("📈 Distribution analysis")

    st.write(
        "See how one numeric variable is spread across its possible values."
    )

    df = require_data()

    if df is not None:

        nums = safe_numeric_columns(df)

        if not nums:

            st.info("No usable numeric columns were detected.")

        else:

            col = st.selectbox("Select a numeric column", nums)

            values = pd.to_numeric(
                df[col],
                errors="coerce"
            ).dropna()

            if values.empty:

                st.warning("This column has no valid numeric values to plot.")

            else:

                bins = st.slider(
                    "Histogram detail (number of bins)",
                    5,
                    50,
                    18
                )

                fig, ax = plt.subplots(figsize=(8, 3.6))

                sns.histplot(
                    values,
                    bins=bins,
                    kde=len(values) > 2,
                    ax=ax,
                    color="#26765a"
                )

                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)

                plot_to_streamlit(fig)

                fig, ax = plt.subplots(figsize=(8, 2.3))

                sns.boxplot(
                    x=values,
                    ax=ax,
                    color="#c6df9b"
                )

                ax.set_title(
                    f"Spread and possible unusual values: {col}"
                )

                plot_to_streamlit(fig)

                st.write(explain_distribution(values, col))

                insight(
                    "A histogram groups values into ranges; taller bars "
                    "show ranges with more observations. A boxplot shows "
                    "the median, middle half and potential unusual values. "
                    "Potential outliers should be investigated, not "
                    "automatically treated as errors."
                )


# ============================================================
# CATEGORY ANALYSIS
# ============================================================

elif page == "Category Analysis":

    st.title("🍎 Category analysis")

    st.write(
        "Compare groups and see which categories appear most often."
    )

    df = require_data()

    if df is not None:

        cats = safe_categorical_columns(df)

        if not cats:

            st.info("No categorical columns were detected.")

        else:

            cat = st.selectbox("Choose a category column", cats)

            st.markdown("### 🏷️ Category frequency")

            counts = (
                df[cat]
                .fillna("(Missing)")
                .astype(str)
                .value_counts()
                .head(20)
            )

            fig, ax = plt.subplots(
                figsize=(8, max(3, .35 * len(counts)))
            )

            sns.barplot(
                x=counts.values,
                y=counts.index,
                ax=ax
            )

            ax.set_xlabel("Number of rows")
            ax.set_ylabel("")
            ax.set_title(f"Most common values in {cat}")

            plot_to_streamlit(fig)

            st.dataframe(
                counts.rename("Rows")
                .rename_axis(cat)
                .reset_index(),
                use_container_width=True,
                hide_index=True
            )

            insight(
                f"The most common value in <b>{cat}</b> is "
                f"<b>{counts.index[0]}</b>, appearing "
                f"{int(counts.iloc[0])} times."
            )

            nums = safe_numeric_columns(df)

            if nums:

                st.markdown(
                    "### 📊 Compare a numeric measure across groups"
                )

                num = st.selectbox("Numeric measure", nums)

                grouped = (
                    df.assign(
                        _measure=pd.to_numeric(
                            df[num],
                            errors="coerce"
                        )
                    )
                    .groupby(cat, dropna=False)["_measure"]
                    .agg(["count", "mean", "median"])
                    .sort_values("mean", ascending=False)
                )

                st.dataframe(
                    grouped,
                    use_container_width=True
                )

                insight(
                    f"This table compares the average and median of "
                    f"<b>{num}</b> inside each <b>{cat}</b> group."
                )


# ============================================================
# SCALING & STANDARDIZATION
# ============================================================

elif page == "Scaling & Standardization":

    st.title("⚖️ Scaling & standardization")

    st.write(
        "Put numeric columns onto comparable scales."
    )

    df = require_data()

    if df is not None:

        nums = safe_numeric_columns(df)

        if not nums:

            st.info("Scaling needs at least one numeric column.")

        else:

            selected = st.multiselect(
                "Select columns to transform",
                nums,
                default=nums[:min(3, len(nums))]
            )

            method = st.radio(
                "Transformation",
                [
                    "Min–max scaling (0 to 1)",
                    "Standardization (z-score)"
                ],
                horizontal=True
            )

            if selected:

                transformed = scale_numeric(
                    df,
                    selected,
                    method
                )

                st.markdown("### 🔬 Transformed preview")

                st.dataframe(
                    transformed[selected].head(15),
                    use_container_width=True
                )

                if method.startswith("Min"):

                    insight(
                        "Min–max scaling changes selected columns so "
                        "their minimum becomes 0 and maximum becomes 1."
                    )

                else:

                    insight(
                        "Standardization expresses values in "
                        "standard-deviation units from the mean."
                    )

                st.download_button(
                    "⬇️ Download transformed columns as CSV",
                    transformed.to_csv(index=False).encode("utf-8"),
                    "nutrilens_scaled_data.csv",
                    "text/csv"
                )

                if st.button(
                    "💾 Save transformed data as working dataset"
                ):

                    st.session_state.df_working = transformed

                    st.success(
                        "Transformed data saved as the current working dataset."
                    )

                    st.rerun()


# ============================================================
# EXPORT DATA
# ============================================================

elif page == "Export Data":

    st.title("📥 Export your data")

    st.write(
        "Download the current working dataset as a CSV file."
    )

    df = require_data()

    if df is not None:

        st.markdown("### 👀 Export preview")

        st.dataframe(
            df.head(12),
            use_container_width=True
        )

        csv_bytes = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download current dataset (CSV)",
            csv_bytes,
            "nutrilens_cleaned_data.csv",
            "text/csv",
            type="primary",
            use_container_width=True
        )

        st.caption(
            f"Ready to export {len(df):,} rows and "
            f"{len(df.columns)} columns."
        )

        insight(
            "The downloaded file contains the current working version. "
            "If you applied cleaning or saved scaled values, those changes "
            "are included. Your original file is not overwritten."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="small-muted">'
    '🥗 NutriLens · Food & Nutrition Insight Explorer · '
    'Built with Python, Streamlit and Pandas 🍎'
    '</div>',
    unsafe_allow_html=True
)
