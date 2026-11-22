# NutriLens – Food & Nutrition Insight Explorer

NutriLens is a beginner-friendly Python web application for exploratory data analysis (EDA). It turns a food and nutrition sample dataset or a user's own CSV file into an interactive dashboard with clear explanations.

## Features

- Load the included food and nutrition dataset or upload a CSV.
- Automatically inspect dataset dimensions, data types, numeric/categorical columns and unique values.
- Check missing values and duplicate rows.
- Preview, select and clean columns; remove duplicates; handle missing values.
- Calculate mean, median, mode, variance, standard deviation, minimum, maximum, quartiles and range.
- Explore numeric distributions with histograms and boxplots.
- Compare categories with frequency charts and grouped numeric summaries.
- Apply min–max scaling or z-score standardization.
- Download the current working dataset or transformed data as CSV.
- Simple-English “What does this mean?” explanations throughout.

## EDA syllabus coverage

| Unit | Coverage |
|---|---|
| Unit 1 | EDA fundamentals, dataset structure, exploration, transformations, reshaping-ready Pandas workflow and visual aids |
| Unit 2 | Pandas operations, selection, missing-data handling, cleaning, duplicate handling, grouping, aggregation and category comparisons |
| Unit 3 | Univariate analysis, numerical summaries, distribution analysis, quartiles, scaling and standardization |

## Technologies

Python, Streamlit, Pandas, NumPy, Matplotlib and Seaborn.

## Project structure

```text
NutriLens/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── sample_food_data.csv
└── modules/
    ├── __init__.py
    └── analysis.py
```

## Installation

1. Install Python 3.10 or newer.
2. Download and unzip `NutriLens.zip`.
3. Open the extracted `NutriLens` folder in VS Code.
4. Create and activate a virtual environment:

   **Windows (PowerShell)**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run locally

From the project root folder:

```bash
streamlit run app.py
```

Streamlit will show a local URL (usually `http://localhost:8501`) to open in your browser.

## Use your own CSV

1. Open the **Dataset Upload** page from the sidebar.
2. Select a `.csv` file.
3. NutriLens reads the column names and detects numeric and categorical columns automatically.
4. Visit the analysis pages. If a dataset does not contain a suitable column type, the page explains why that analysis is unavailable.
5. Use **Data Cleaning** to make changes to the working copy. The original loaded dataset remains available through **Restore original dataset**.
6. Download the result from **Export Data**.

UTF-8 CSV files are recommended. Files should have a header row. Very large files may use more memory and take longer to process.

## GitHub instructions

Create a new empty repository on GitHub, then run these commands inside the project folder:

```bash
git init
git add .
git commit -m "Add NutriLens EDA application"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/NutriLens.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Do not commit private datasets or secrets.

## Deploy with Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Choose **Create app** and connect the repository.
4. Set the main file path to `app.py`.
5. Deploy. Streamlit installs packages from `requirements.txt`.
6. Keep `data/sample_food_data.csv` in the repository so the sample-data option works.

## Notes and responsible interpretation

- The application is a general-purpose EDA explorer, not a nutrition or medical advice tool.
- Statistical summaries describe the data provided; they do not establish cause and effect.
- Automatic numeric detection is heuristic for text columns containing numbers. Review detected types when analysing unusual datasets.
- Filling missing values changes the working copy and may affect statistical results. Choose a method deliberately.
- Standardization uses the sample standard deviation (Pandas default, `ddof=1`).

## Testing

The package includes a lightweight local test script at `tests/test_analysis.py`. Run it with:

```bash
python -m unittest discover -s tests -v
```

The app itself can be launched using `streamlit run app.py`. Browser-based interaction tests require a running browser session.
