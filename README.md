# Project Deliverable 2: ML Engine — Ridge Regression

**Course:** ICS 3202 – Artificial Intelligence  
**Project:** Skill Exchange and Service Evaluation App  
**Group Members:** Alicia Maryanne, Charlotte Mujuri, Lyon Agina

## 1. Overview

This project develops a machine learning engine to estimate the monetary value of services exchanged through the application. It uses supervised regression to predict service prices from categories, sub-categories, titles, ratings, and review counts.

The predictions help assess whether two services have approximately equal value and determine when an exchange may require a cash top-up or an additional service.

## 2. Repository Structure

| File | Description |
|---|---|
| `Ridge_Base.ipynb` | Data preparation, analysis, model training, and evaluation |
| `fiverr.csv` | Dataset containing 6,183 original service listings |
| `README.md` | Project documentation |

## 3. Setup and Execution

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

1. Place `fiverr.csv` in the same directory as the notebook.
2. Open `Ridge_Base.ipynb` in Jupyter Notebook or Google Colab.
3. Run all cells in order. In Colab, upload the dataset through the Files panel.

## 4. Models Used

- **Ridge Regression:** Base model that uses regularisation to reduce overfitting.
- **Lasso Regression:** Comparison model that can reduce less useful feature coefficients to zero.
- **Random Forest Regression:** Comparison model that captures non-linear relationships.
- **Dummy Regressor:** Baseline that predicts the mean target value.

The dataset is split into 80% training data and 20% testing data. Five-fold cross-validation is used to select model parameters and assess performance. Feature preparation includes one-hot encoding, TF-IDF text vectorisation, imputation, and scaling.

## 5. Results

The selected Ridge model uses `alpha = 5`.

| Metric | Result |
|---|---:|
| Test RMSE (log price) | 0.941 |
| R² | 0.339 |
| Mean Absolute Error | €44.30 |
| Median Absolute Error | €13.41 |

Ridge outperforms the baseline and performs comparably to Lasso and Random Forest. Its predictions are approximate and intended for relative service valuation.

## 6. Integration with the App

The intended workflow is:

1. Users submit their service details.
2. The backend passes the details to the trained model.
3. The model predicts each service's value.
4. The application compares the predictions.
5. Exchanges within a 15% tolerance are considered approximately balanced. Otherwise, a cash top-up or additional service may be suggested.

The notebook currently demonstrates this valuation workflow. Full integration requires connecting the saved model and preprocessing pipeline to the application's backend.

## 7. Limitations and Future Work

The dataset contains Fiverr starting prices in euros and may not reflect Kenyan market rates. Prediction accuracy is also limited by the available features.

Future work includes validating the model using Kenyan service prices, integrating it into the application backend, and developing a recommendation system for matching users by skills, location, and ratings.
