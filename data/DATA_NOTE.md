# About the data

This is the "Adult" dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/2/adult

It's under a CC BY 4.0 license, so if you use it, cite it like this:

Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

**Reminder:** this is real 1994 Census data (de-identified). We're only using it as a benchmark to test our baseline. Don't add it to the synthetic persona dataset.

## Files

- `adult.csv` - all 48,842 rows (train and test combined) with column names. The `split` column tells you which original file a row came from. Missing values are still `?` like in the original. I also removed the extra "." at the end of the income labels in the test file so both sets match.
- `original/` - the untouched files from UCI, in case you want to check them against the official download.
