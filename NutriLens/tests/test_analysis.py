import unittest
import pandas as pd
from modules.analysis import (
    profile_dataset, missing_summary, numeric_summary, clean_dataframe,
    scale_numeric, safe_numeric_columns, safe_categorical_columns
)

class NutriLensAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "group": ["A", "B", "A", "B"],
            "amount": [1.0, 2.0, None, 4.0],
            "count": [10, 20, 30, 40]
        })

    def test_profile(self):
        p = profile_dataset(self.df)
        self.assertIn("amount", p["numeric"])
        self.assertIn("group", p["categorical"])
        self.assertEqual(len(p["column_info"]), 3)

    def test_missing(self):
        m = missing_summary(self.df)
        self.assertEqual(int(m.loc[m["Column"] == "amount", "Missing values"].iloc[0]), 1)

    def test_summary(self):
        s = numeric_summary(self.df[["count"]])
        self.assertAlmostEqual(s.loc["count", "Mean"], 25)

    def test_cleaning(self):
        dup = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        out = clean_dataframe(dup, remove_duplicates=True)
        self.assertEqual(len(out), 4)
        filled = clean_dataframe(self.df, missing_action="Fill numeric gaps with column median")
        self.assertFalse(filled["amount"].isna().any())

    def test_scaling(self):
        out = scale_numeric(self.df, ["count"], "Min–max scaling (0 to 1)")
        self.assertAlmostEqual(out["count"].min(), 0)
        self.assertAlmostEqual(out["count"].max(), 1)

    def test_column_detection(self):
        self.assertIn("count", safe_numeric_columns(self.df))
        self.assertIn("group", safe_categorical_columns(self.df))

if __name__ == "__main__":
    unittest.main()
