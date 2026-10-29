# UCI Adult Benchmark

This is part of our CYSE 492/493 senior design project (Group 32, sponsored by NALEJ) on degrading AI-assisted target package accuracy.

I'm using the UCI Adult dataset to test our baseline profiling approach before we run it on the team's synthetic personas.

**Heads up:** this is real 1994 Census data (no names or IDs). We're only using it as a benchmark. It's not part of our synthetic persona dataset and shouldn't get mixed into it.

## What's in here

- `data/adult.csv` - the full dataset (48,842 rows) with column names added so it's easy to read
- `data/original/` - the original files straight from UCI
- `data/DATA_NOTE.md` - where the data came from and the license
- `inspect_adult.py` - the script I used to go through the data

## What the script does

It basically does two things:

1. **Checks the data.** Missing values, duplicates, how each column is spread out, and which columns are closely tied to each other.
2. **Runs a quick baseline test.** It hides one attribute (income, sex, marital status, or occupation) and tries to guess it from the rest. It compares just guessing the most common answer vs. two models (logistic regression and gradient boosting), and reports accuracy, precision, recall, F1, and how confident the model is when it's right vs. wrong.

It also checks how many people can be singled out just by combining a few attributes like age, sex, race, and education.

## How to run it

```bash
pip install pandas scikit-learn
python3 inspect_adult.py data/original results/
```

Results print out and also save to `results/adult_inspection.json`.

## Citation

The dataset is under a CC BY 4.0 license, so it has to be cited:

Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

Official page: https://archive.ics.uci.edu/dataset/2/adult
