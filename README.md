# nuwe-ds-challenge
Repository for the Nuwe's "Product manager - Data science challenge manager" position challange.

## Model benchmarking tables (showing F1 score)

### models - datasets (trained on BCE Loss)

| Model Name          | raw     | cleaned | balanced |
|---------------------|---------|---------|----------|
| baseline            | 0.8950 | 0.8857 | 0.7186 |
| baseline_fine-tuned | 0.9063 | ...  | 0.7206 |
| ...                 | ...     | ...     | ...      |



### models - datasets (trained on [Smoothed F1 Score Loss](https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric/notebook))

| Model Name          | raw     | cleaned | balanced |
|---------------------|---------|---------|----------|
| baseline            | 0.8579  | 0.8568 | 0.7372 |
| baseline_fine-tuned | ...| ...  | ... |
| ...                 | ...     | ...     | ...      |
