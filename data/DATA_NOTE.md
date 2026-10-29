# About this data

**Source:** UCI Machine Learning Repository, "Adult" dataset
https://archive.ics.uci.edu/dataset/2/adult

**Citation (required by the CC BY 4.0 license):**
Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5XW20

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

## Important

This is **real, de-identified 1994 U.S. Census data**. In this project it is used
**only as an external benchmark** to validate the baseline profiling method. It is
**not** part of the project's synthetic persona dataset, and it must not be mixed
into it.

## Files

- `adult.csv`: all 48,842 records (train + test combined) with column names added.
  The `split` column shows which original file each row came from. Missing values
  are kept as `?`, as in the original. The trailing "." on test-set income labels
  was removed so both sets use the same labels.
- `original/`: UCI's untouched files (`adult.data`, `adult.test`, `adult.names`),
  for verification against the official source.
