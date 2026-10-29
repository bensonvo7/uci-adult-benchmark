# UCI Adult Benchmark

Baseline attribute-inference benchmark for the GMU CYSE 492/493 Group 32 project,
*Degrading AI-Assisted Intelligence Target Package Accuracy* (sponsor: NALEJ).

> **Important:** UCI Adult is real (de-identified) 1994 U.S. Census data. It is used
> only as an **external benchmark** to validate the baseline profiling method. It is
> **not** part of the project's synthetic persona dataset.

## What the script does

`inspect_adult.py`:

- Verifies the dataset (row/column counts, missing values, duplicates)
- Profiles every attribute (distributions, cardinality, skew)
- Measures dependencies between attributes (normalized mutual information)
- Measures quasi-identifier uniqueness (how many records are singled out by combinations of attributes)
- Runs a baseline attribute-inference test: hides one attribute (income, sex, marital status, occupation)
  and predicts it from the rest, comparing a majority-class guess, logistic regression, and gradient boosting
- Reports accuracy, macro precision/recall/F1, and confidence measures (mean confidence,
  expected calibration error, accuracy when confidence ≥ 0.9)

## The data

The data is included in the [`data/`](data/) folder:

- **Just want to look at it?** Open [`data/adult.csv`](data/adult.csv). GitHub shows it as a
  searchable table (48,842 rows, with column names).
- The original UCI files are in [`data/original/`](data/original/).
- See [`data/DATA_NOTE.md`](data/DATA_NOTE.md) for the source, license, and how the CSV was made.

Official source: https://archive.ics.uci.edu/dataset/2/adult

## Running it

```bash
pip install pandas scikit-learn
python3 inspect_adult.py data/original results/
```

Results are printed and saved to `results/adult_inspection.json`.

## Citation

Licensed under CC BY 4.0, which requires attribution:

Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5XW20
