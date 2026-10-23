# Credit Default Risk Prediction

An educational machine-learning project using the UCI Default of Credit Card Clients dataset.

## What it does
- Explores the dataset
- Trains Logistic Regression, Decision Tree, and Random Forest
- Compares Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Saves the best model selected by F1-score
- Creates confusion-matrix and ROC-curve plots
- Provides a Streamlit interface for trying sample predictions

## Dataset
Place `default of credit card clients.xls` in the `data/` folder. The target is `default payment next month`.

## Setup
```bash
python -m pip install -r requirements.txt
```

## Run exploratory analysis
```bash
python eda.py
```

## Train models
```bash
python train.py
```

## Launch the web app
```bash
python -m streamlit run app.py
```

## Important limitation
This dataset predicts next-month credit-card payment default. It does not measure every aspect of creditworthiness and this educational demo should not be used for actual lending or credit-approval decisions.
