# Data Download Instructions

This project uses the **H&M Personalized Fashion Recommendations** dataset from Kaggle.

## Prerequisites

1. A [Kaggle account](https://www.kaggle.com/)
2. Accept the [competition rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules)
3. Kaggle API credentials configured (`~/.kaggle/kaggle.json`)

## Download steps

```bash
# Install the Kaggle CLI if you haven't
pip install kaggle

# Download the dataset (≈ 3.5 GB)
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p data/

# Unzip
cd data/
unzip h-and-m-personalized-fashion-recommendations.zip
cd ..
```

You should now have three CSV files in `data/`:

| File                        | Description                    |
|-----------------------------|--------------------------------|
| `transactions_train.csv`    | Purchase transactions          |
| `customers.csv`             | Customer demographics          |
| `articles.csv`              | Article (product) metadata     |

## Data-use terms

Please review and comply with the competition's data-use terms before proceeding.
The images directory (`data/images/`) is **not required** for this analysis.

## Note

Raw data files are **not committed** to version control (see `.gitignore`).
This is a public dataset and is **not** data from any employer.
