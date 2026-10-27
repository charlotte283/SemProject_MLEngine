# SemProject_MLEngine: Ridge_Base.ipynb explained

**Course:** ICS 3202 Artificial Intelligence, Project Deliverable 2 (CAT 2)
**Project:** Skill Exchange and Service Evaluation App


This README walks through `Ridge_Base.ipynb` cell by cell. For each code cell it shows what the code does, line by line, in plain words, then the real output from our run.

---

## Files in this repo

| File | What it is |
|---|---|
| `Ridge_Base.ipynb` | The notebook (with outputs already saved) |
| `fiverr.csv` | The dataset (6,183 rows, 7 columns) |
| `README.md` | This file |

**To run it:** keep `fiverr.csv` in the same folder as the notebook (in Colab, upload it in the Files panel), then Runtime > Run all.

---

## The big picture

```
 fiverr.csv ──► Clean ──► Explore (EDA) ──► Train Ridge ──► Check with ──► Use in the app
 (raw text)    (numbers)   (6 findings)     (pick alpha      unseen test    (fair-swap check)
                                             by 5-fold CV)    data
                                                  │
                                                  └──► Compare with Lasso + Random Forest (Q4)
```

**The problem:** our app lets two people swap services. For a swap to be fair, the app must know roughly how much each service is worth.

**The ML task:** regression. The model reads a service (category, sub-category, title, rating, number of reviews) and predicts its **price**.

**Base model:** Ridge Regression, which is linear regression plus a penalty (`alpha`) that stops it from overfitting. This covers the "regularisation for regression" requirement in Question 3.

---

## Cell 1: Imports

```python
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

| Line | What it does |
|---|---|
| `import re` | Python's tool for pattern matching in text |
| `import numpy as np` | Maths on arrays of numbers (we use `np.log1p`, `np.expm1`, `np.where`) |
| `import pandas as pd` | Tables (DataFrames). Almost all our data work uses this |
| `import matplotlib.pyplot as plt` | Drawing charts |
| `import seaborn as sns` | Nicer-looking charts built on matplotlib |

```python
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
```

| Imported tool | What we use it for |
|---|---|
| `train_test_split` | Split the data into training and test parts |
| `GridSearchCV` | Try many `alpha` values using cross-validation and pick the best |
| `cross_validate` | Get cross-validation scores for a model with nothing to tune (Random Forest) |
| `KFold` | Defines how the 5 cross-validation folds are cut |
| `Pipeline` | Chains steps (prepare data, then model) so they always run in order |
| `ColumnTransformer` | Applies a different preparation to each type of column |
| `OneHotEncoder` | Turns category names into 0/1 columns |
| `StandardScaler` | Puts numeric columns on the same scale |
| `TfidfVectorizer` | Turns the service title text into numbers (word scores) |
| `SimpleImputer` | Fills missing values |
| `Ridge`, `Lasso` | The two linear models with a penalty |
| `RandomForestRegressor` | A tree-based model that can learn non-linear patterns |
| `DummyRegressor` | The "always guess the average" baseline |
| the `metrics` imports | Ways to measure how wrong the predictions are |

```python
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
```

- `sns.set_theme(...)` sets one chart style for the whole notebook.
- `RANDOM_STATE = 42` is a fixed "seed". Splits and random choices come out the same every run, so everyone gets the same numbers.

---

## Section 1: Load and explore the data

### Cell 3: Load the file

```python
DATA_PATH = "fiverr.csv"
df = pd.read_csv(DATA_PATH)

rows, columns = df.shape
print("Rows:", rows, "| Columns:", columns)
df.head()
```

- `DATA_PATH` holds the file name. Change it if your CSV is somewhere else.
- `pd.read_csv` reads the CSV into a table called `df`.
- `df.shape` gives (number of rows, number of columns). We unpack it into `rows` and `columns`.
- `df.head()` shows the first 5 rows.

**Output:** `Rows: 6183 | Columns: 7`

The 7 columns: `Category`, `Category-href`, `Subcat`, `Subcat-href`, `name`, `stars`, `price`.

### Cell 4: Check types, missing values, duplicates

```python
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nFully duplicated rows:", df.duplicated().sum())
print("Distinct categories:", df["Category"].nunique(), "| Distinct sub-categories:", df["Subcat"].nunique())
```

- `df.dtypes` shows what type each column is.
- `df.isnull().sum()` counts empty cells in each column.
- `df.duplicated().sum()` counts rows that are exact copies of another row.
- `.nunique()` counts distinct values.

**What we found:**

| Check | Result |
|---|---|
| Every column's type | `object` (text), including `price` and `stars`, which should be numbers |
| Missing values | only `stars` has any: **493** |
| Duplicate rows | **1** |
| Categories / sub-categories | **7** / **129** |

### Cell 5: Problems found in the raw data (written note)

| Problem | Example |
|---|---|
| `price` is text | `Starting at€137.76` |
| `stars` is text with rating and review count stuck together | `5.0(10)`; `1k+` means 1000 or more; some are missing |
| Leftover tags on sub-category names | `Podcast EditingNew`, `Local PhotographyBeta` |
| Link columns are not useful | `Category-href`, `Subcat-href` |
| Duplicate rows | 1 found |

---

## Section 2: Data cleaning

### Cell 7: The nine-step cleaning

```python
clean = df.copy()
```
Makes a copy so the original `df` stays untouched. All cleaning happens on `clean`.

**Step 1: Drop the link columns**
```python
clean = clean.drop(columns=["Category-href", "Subcat-href"])
```
Web links do not describe the service, so the model cannot learn from them.

**Step 2: Turn the price text into a number**
```python
clean["price_eur"] = (
    clean["price"].str.extract(r"€\s*([\d,]+\.?\d*)")[0]
    .str.replace(",", "", regex=False)
    .astype(float)
)
```

| Piece | Meaning |
|---|---|
| `.str.extract(r"€\s*([\d,]+\.?\d*)")` | Find the `€` sign, skip any spaces, then capture the number after it. `[\d,]+` = digits (and commas), `\.?` = an optional decimal point, `\d*` = more digits |
| `[0]` | Take the captured number (the first group) |
| `.str.replace(",", "")` | Remove thousands commas so `1,200` becomes `1200` |
| `.astype(float)` | Convert from text to a real number |

Result: `Starting at€137.76` becomes `137.76`.

**Step 3: Split `stars` into a rating and a review count**
```python
parts = clean["stars"].str.extract(r"^([\d.]+)\(([\d.]+)(k?)\+?\)$")
clean["rating"] = parts[0].astype(float)
clean["reviews"] = parts[1].astype(float) * np.where(parts[2] == "k", 1000, 1)
```

The pattern reads `5.0(1k+)` from left to right:

| Part of pattern | Matches | Captured as |
|---|---|---|
| `^([\d.]+)` | the rating at the start, e.g. `5.0` | `parts[0]` |
| `\(` | the opening bracket | |
| `([\d.]+)` | the review number, e.g. `1` or `68` | `parts[1]` |
| `(k?)` | an optional letter `k` | `parts[2]` |
| `\+?\)$` | an optional `+`, then the closing bracket | |

- `rating` = `parts[0]` as a number.
- `reviews` = the review number, **times 1000 if there was a `k`**, otherwise times 1. So `1k+` becomes 1000.

```python
unparsed = clean["stars"].notna() & clean["rating"].isna()
print("Stars values that could not be parsed:", unparsed.sum())
```
A safety check: it counts rows where `stars` had a value but we failed to read a rating from it. **Output: 0**, so every value was understood.

**Step 4: Handle missing stars**
```python
clean["reviews"] = clean["reviews"].fillna(0)
clean["has_reviews"] = clean["stars"].notna().astype(int)
```
- A missing `stars` means the seller has no reviews yet, so we set `reviews` to 0.
- `has_reviews` is a new 0/1 column: 1 if the service has reviews, 0 if not.
- We **do not** fill the missing `rating` here. That happens later inside the model pipeline (see "Data leakage" below).

**Step 5: Remove the leftover "New" / "Beta" tags**
```python
clean["Subcat"] = clean["Subcat"].str.replace(r"(?<=[a-z])(New|Beta)$", "", regex=True).str.strip()
```
- `(?<=[a-z])` means "only if the character just before is a lowercase letter".
- `(New|Beta)$` means "the word New or Beta at the very end".
- So `Podcast EditingNew` becomes `Podcast Editing`, and `Local PhotographyBeta` becomes `Local Photography`.
- `.str.strip()` trims stray spaces.

**Step 6: Tidy the service title**
```python
clean["service"] = (
    clean["name"].str.lower()
    .str.replace(r"^i will\s+", "", regex=True)
    .str.strip()
)
```
Makes the title lower case and removes the "I will " that starts almost every Fiverr listing, because it carries no information. `I will write gdpr privacy policy...` becomes `write gdpr privacy policy...`.

**Step 7: Drop the old text columns**
```python
clean = clean.drop(columns=["name", "stars", "price"])
```
We have replaced these with `service`, `rating`, `reviews`, `price_eur`, so the old ones are removed.

**Step 8: Remove duplicates**
```python
before = len(clean)
clean = clean.drop_duplicates().reset_index(drop=True)
print("Duplicate rows removed:", before - len(clean))
```
Counts rows before, drops exact copies, renumbers the rows, and prints how many were removed. **Output: 1.**

### Cell 8: Outliers, log transform, final check

```python
q1, q3 = clean["price_eur"].quantile([0.25, 0.75])
upper_fence = q3 + 1.5 * (q3 - q1)
n_out = (clean["price_eur"] > upper_fence).sum()
print(f"Prices above the IQR fence (EUR {upper_fence:.2f}): {n_out} rows ({n_out/len(clean):.1%})")
```

This is the **IQR rule** for spotting outliers:
- `q1` and `q3` are the 25th and 75th percentile prices.
- The "fence" is `q3 + 1.5 × (q3 − q1)`. Anything above it counts as an unusually high price.
- **Output:** fence = EUR 131.09, with **708 rows (11.5%)** above it.

**Decision: we keep the outliers.** They are real, expensive services (consulting, for example), not mistakes. Instead, we reduce their pull on the model by predicting the **log** of the price:

```python
clean["log_price"] = np.log1p(clean["price_eur"])
clean["log_reviews"] = np.log1p(clean["reviews"])
```
- `np.log1p(x)` = log(1 + x). The "+1" lets it handle zero safely (needed for `reviews = 0`).
- Logging squashes huge values so a handful of expensive services do not dominate.

```python
print(clean.isnull().sum())
print("\nFinal shape:", clean.shape)
clean.dtypes
```
Final checks. **Output:** the only missing values left are the **493** in `rating`. Final shape is **(6182, 9)**.

**Final columns:** `Category`, `Subcat`, `price_eur`, `rating`, `reviews`, `has_reviews`, `service`, `log_price`, `log_reviews`.

### Data leakage (the note in cell 9)

If we filled the missing ratings now using the average of the whole dataset, the test data would influence the fill value. That is "data leakage" and it makes results look better than they really are. So we fill inside the model pipeline, where the fill value is learned **from training data only**.

---

## Section 3: Exploratory Data Analysis (6 observations)

Every number printed is calculated from the data, so the text always matches the data.

### Cell 11: Observation 1, prices are heavily skewed
```python
mean_p, median_p = clean["price_eur"].mean(), clean["price_eur"].median()
skew = clean["price_eur"].skew()
```
Calculates the mean, the median, and the skewness (how lopsided the distribution is).

```python
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
sns.histplot(clean["price_eur"], bins=50, ax=ax[0]) ...
sns.histplot(clean["log_price"], bins=50, ax=ax[1], color="seagreen") ...
```
Draws two histograms side by side: raw price (left) and log price (right).

**Output:** mean = **EUR 65.99**, median = **EUR 22.22**, skewness = **14.5**.
**Meaning:** most services are cheap and a few are very expensive, so the average is pulled far above the typical price. The log chart is much more balanced, which is why we model `log(price)`.

### Cell 12: Observation 2, prices come in fixed tiers
```python
n_unique = clean["price_eur"].nunique()
top5 = clean["price_eur"].value_counts(normalize=True).head(5)
```
- `.nunique()` counts the distinct prices.
- `.value_counts(normalize=True)` gives each price's share of all rows. `.head(5)` keeps the top 5.

**Output:** only **142** different prices exist in 6,182 rows. The 5 most common cover **52%**:

| Price (EUR) | Share |
|---|---|
| 4.44 | 15.9% |
| 8.89 | 14.3% |
| 13.33 | 8.3% |
| 17.78 | 7.7% |
| 22.22 | 5.5% |

**Meaning:** sellers choose from set price steps, so the model can only give an estimate, not an exact tier.

### Cell 13: Observation 3, categories differ in price
```python
cat_stats = clean.groupby("Category")["price_eur"].agg(["count", "median", "mean"]).sort_values("median", ascending=False)
```
Groups rows by category and calculates the count, median and mean price of each, sorted by median (highest first). A bar chart of the medians follows.

**Output:**

| Category | Services | Median (EUR) | Mean (EUR) |
|---|---|---|---|
| Video & Animation | 1,248 | 35.55 | 100.50 |
| Programming & Tech | 807 | 31.11 | 115.26 |
| Business | 720 | 26.66 | 62.66 |
| Writing & Translation | 1,440 | 26.66 | 57.36 |
| Digital Marketing | 336 | 26.66 | 46.44 |
| Music & Audio | 768 | 22.22 | 41.43 |
| Lifestyle | 863 | 8.89 | 16.64 |

**Meaning:** Lifestyle is far cheaper than the rest, so category is a useful feature.

### Cell 14: Observation 4, ratings are almost all high
```python
rated = clean["rating"].dropna()
share_high = (rated >= 4.8).mean()
```
- `.dropna()` removes missing ratings first.
- `(rated >= 4.8)` gives True/False per row, and `.mean()` turns that into a share.

**Output:** **96%** of rated services score 4.8 or more (median 5.0).
**Meaning:** almost everyone is rated highly, so rating cannot separate cheap from expensive services.

### Cell 15: Observation 5, missing reviews and weak link to price
```python
missing_share = (1 - clean.groupby("Category")["has_reviews"].mean()).sort_values(ascending=False)
```
`has_reviews` averaged per category gives the share **with** reviews. `1 − that` gives the share **without**.

**Output (5a):** share of services with no reviews yet: Programming & Tech 10.4%, Digital Marketing 8.9%, Video & Animation 8.7%, Writing & Translation 8.5%, Lifestyle 7.2%, Business 6.8%, Music & Audio 4.9%.

```python
corr = clean[["price_eur", "reviews", "rating"]].corr(method="spearman")
```
Spearman correlation measures how much two columns rise and fall together, without being thrown off by outliers. 1 = move together perfectly, 0 = no link.

**Output (5b):** correlation with price: `reviews` = **0.108**, `rating` = **−0.005**.
**Meaning:** review count has only a weak link with price, and rating has almost none.

### Cell 16: Observation 6, categories are unbalanced
```python
print(clean["Category"].value_counts().to_string())
print(... clean["Subcat"].value_counts().min(), ... .max())
```
Counts services per category and finds the smallest and largest sub-category.

**Output:** Writing & Translation 1,440, Video & Animation 1,248, Lifestyle 863, Programming & Tech 807, Music & Audio 768, Business 720, Digital Marketing 336. Sub-categories range from **39 to 48** services each.
**Meaning:** sub-categories are about equal in size, but categories are not. That is one reason we use cross-validation and not one lucky split.

### Cell 17: Summary of EDA findings

| # | Finding | What we do about it |
|---|---|---|
| 1 | Price is strongly right-skewed | Predict `log(price)` |
| 2 | Prices sit on fixed tiers | Treat predictions as estimates |
| 3 | Category matters (Lifestyle is cheap) | Use category and sub-category as features |
| 4 | Ratings are nearly all high | Expect rating to add little |
| 5 | Some sellers have no reviews; reviews weakly linked to price | Add a "has reviews" flag |
| 6 | Categories are uneven | Judge with cross-validation |

---

## Section 4: Modelling with Ridge Regression

### Cell 18: Why Ridge? (written note)

Ridge is linear regression with a penalty. We have thousands of text and category columns but only about 6,000 rows, so an unpenalised model would **memorise** the training data (overfit). `alpha` sets the penalty strength:

```
alpha too small ──► overfitting (great on training data, worse on new data)
alpha too large ──► underfitting (too simple to learn anything)
alpha just right ──► best score on unseen data
```

**Features the model sees:**

| Feature | Meaning in our app |
|---|---|
| Category, Sub-category | what kind of service it is |
| Service title (as TF-IDF word scores) | stand-in for expertise and complexity |
| Rating | quality |
| log(reviews), has-reviews flag | stand-in for demand and trust |

**Target:** `log(price)`

### Cell 19: Choose columns, split, build the preprocessor

```python
num_cols = ["rating", "log_reviews", "has_reviews"]
cat_cols = ["Category", "Subcat"]
text_col = "service"
```
Sorts the input columns by type, because each type needs different preparation.

```python
X = clean[num_cols + cat_cols + [text_col]]
y = clean["log_price"]
```
- `X` = the inputs (features). `y` = the answer we want to predict.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
```
Randomly splits the data: **80% for training, 20% kept aside** as a final exam the model never sees. **Output: 4,945 training rows, 1,237 test rows.**

```python
def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000, stop_words="english"), text_col),
    ])
```

This function builds the "data translator", with one branch per column type:

| Branch | Applied to | What it does |
|---|---|---|
| `num` | rating, log_reviews, has_reviews | First `SimpleImputer(strategy="median")` fills missing ratings with the median **learned from training data only**. Then `StandardScaler()` rescales so no number dominates just because it is larger |
| `cat` | Category, Subcat | `OneHotEncoder` makes one 0/1 column per category name. `handle_unknown="ignore"` means a category not seen in training will not crash the model |
| `txt` | service title | `TfidfVectorizer` turns text into word scores (see below) |

**TF-IDF settings explained:**

| Setting | Meaning |
|---|---|
| TF-IDF | Scores a word higher if it appears often in one title but rarely across all titles, so common filler words count less |
| `ngram_range=(1, 2)` | Use single words ("website") and two-word pairs ("video course") |
| `min_df=2` | Ignore words/pairs that appear in fewer than 2 titles |
| `max_features=3000` | Keep only the 3,000 most useful words/pairs |
| `stop_words="english"` | Drop filler words like "the", "and", "your" |

It is a **function** (`make_preprocessor()`) so every model in the notebook gets a fresh, identical copy of the same preparation. That keeps the comparison fair.

### Cell 20: Scoring helper and baseline

```python
def evaluate(model, X_data, y_log, label):
    pred_log = model.predict(X_data)
    return {
        "Model": label,
        "RMSE (log)": np.sqrt(mean_squared_error(y_log, pred_log)),
        "R2": r2_score(y_log, pred_log),
        "MAE (EUR)": mean_absolute_error(np.expm1(y_log), np.expm1(pred_log)),
        "Median error (EUR)": median_absolute_error(np.expm1(y_log), np.expm1(pred_log)),
    }
```

`evaluate` takes a trained model, makes predictions, and returns four scores. `np.expm1` undoes the log, converting back to euros.

| Metric | Plain meaning | Better is |
|---|---|---|
| **RMSE (log)** | Typical size of the error, measured on log price | lower |
| **R²** | Share of price variation the model explains. 0 = no better than guessing the average, 1 = perfect | higher |
| **MAE (EUR)** | Average error in euros (pulled up by a few expensive services) | lower |
| **Median error (EUR)** | The middle error in euros, a fairer "typical mistake" | lower |

```python
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
baseline_scores = evaluate(baseline, X_test, y_test, "Baseline (always the average)")
```
The **baseline** always predicts the average log price. It is the minimum bar: a real model must beat it.

**Baseline output:** RMSE (log) 1.158, R² about 0, MAE EUR 51.40, median error EUR 16.82.

### Cell 21: Ridge with 5-fold cross-validation

```python
ridge_pipe = Pipeline([("prep", make_preprocessor()), ("model", Ridge())])
```
Chains the preprocessor and the Ridge model into one object, so the data is always prepared first, then modelled.

```python
param_grid = {"model__alpha": [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]}
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
```
- `param_grid` lists the 10 `alpha` values to try. (`model__alpha` means "the `alpha` of the step named `model`".)
- `KFold(n_splits=5, shuffle=True)` cuts the training data into **5 shuffled parts**.

**How cross-validation works:**

```
Round 1: [TEST] [train] [train] [train] [train]
Round 2: [train] [TEST] [train] [train] [train]
Round 3: [train] [train] [TEST] [train] [train]
Round 4: [train] [train] [train] [TEST] [train]
Round 5: [train] [train] [train] [train] [TEST]
                  average the 5 test scores
```
Every part gets a turn as the test, so the score does not depend on one lucky split.

```python
grid = GridSearchCV(ridge_pipe, param_grid, cv=cv,
                    scoring="neg_root_mean_squared_error", return_train_score=True)
grid.fit(X_train, y_train)
```
- `GridSearchCV` runs the 5-fold cross-validation for **every** `alpha` and keeps the best.
- `scoring="neg_root_mean_squared_error"`: scikit-learn always treats "higher is better", so errors are negated. That is why we later flip the sign.
- `return_train_score=True` also records the training error, which we plot next.

```python
best_alpha = grid.best_params_["model__alpha"]
best_idx = grid.best_index_
```
Picks out the winning `alpha` and its row number in the results table.

**Output:** best alpha = **5**. Cross-validated RMSE (log) = **0.961** (± 0.019 across the 5 folds).

### Cell 22: Regularisation plot (training vs validation error)

```python
res = pd.DataFrame(grid.cv_results_)
alphas = res["param_model__alpha"].astype(float)
plt.plot(alphas, -res["mean_train_score"], marker="o", label="Training error")
plt.plot(alphas, -res["mean_test_score"], marker="o", label="Validation error (cross-validation)")
plt.axvline(best_alpha, color="grey", linestyle="--", ...)
plt.xscale("log") ...
```
- `grid.cv_results_` holds every score from the search, turned into a table.
- The minus signs flip the negated scores back to normal errors.
- Two lines are drawn: training error and validation error, for each `alpha`.
- A dashed vertical line marks the chosen `alpha`.
- `plt.xscale("log")` spaces the alphas evenly on the axis, since they range from 0.01 to 100.

**How to read it:** with a tiny alpha the training error is very low but the validation error is higher, which means overfitting. As alpha grows, the gap closes. We pick the alpha where the validation error is lowest.

### Cell 24: Final test on unseen data

```python
ridge_best = grid.best_estimator_
results = pd.DataFrame([baseline_scores, evaluate(ridge_best, X_test, y_test, f"Ridge (alpha={best_alpha})")])
results.round(3)
```
- `grid.best_estimator_` is the pipeline already retrained on all the training data with the best `alpha`.
- We score it on the 20% test set, the first time the model sees it, and put it next to the baseline.

**Output:**

| Model | RMSE (log) | R² | MAE (EUR) | Median error (EUR) |
|---|---|---|---|---|
| Baseline (always the average) | 1.158 | −0.001 | 51.40 | 16.82 |
| **Ridge (alpha=5)** | **0.941** | **0.339** | **44.30** | **13.41** |

### Cell 25: What the results mean (in simple words)

- Ridge explains about **a third** (R² 0.34) of the differences in price. It clearly beats guessing, but it is not precise.
- The average euro error looks large because a few very expensive services dominate it. The **median error (about EUR 13)** is the fairer measure.
- The ceiling is low because sellers set their own prices, and our data only has title, category, rating and review count.
- For the app, this is a **rough guide** for comparing two services (which is worth more, roughly by how much), not an exact price.

### Cell 26: Predicted vs actual, and residuals

```python
pred_eur = np.expm1(ridge_best.predict(X_test))
actual_eur = np.expm1(y_test)
resid = np.log1p(actual_eur) - np.log1p(pred_eur)
```
- `pred_eur`: the model's predictions on the test set, converted back to euros.
- `actual_eur`: the real prices.
- `resid`: the errors (actual minus predicted, in log space).

```python
ax[0].scatter(actual_eur, pred_eur, alpha=0.3)
ax[0].plot([0, lim], [0, lim], color="red", linestyle="--")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
sns.histplot(resid, bins=40, ax=ax[1], color="purple")
ax[1].axvline(0, color="red", linestyle="--")
```
- **Left chart:** each dot is one test service. The red dashed line is "perfect prediction". Dots near the line are good predictions. Both axes use log scale so cheap and expensive services are visible.
- **Right chart:** a histogram of the errors. A healthy model has errors centred around 0 (red line), so it does not systematically over- or under-predict.

### Cell 27: Which features matter most

```python
names = ridge_best.named_steps["prep"].get_feature_names_out()
coefs = ridge_best.named_steps["model"].coef_
coef_df = pd.Series(coefs, index=names).sort_values()
```
- `get_feature_names_out()` lists every column the preprocessor created (categories, words, numbers).
- `.coef_` holds the weight Ridge gave each one. Positive pushes the predicted price up, negative pushes it down.
- We pair names with weights and sort from lowest to highest.

```python
print(coef_df.head(8) ...)   # 8 most negative
print(coef_df.tail(8) ...)   # 8 most positive
```

**Output:**

| Lowers predicted price most | Weight | Raises predicted price most | Weight |
|---|---|---|---|
| Sub-category: Article to Video | −0.997 | Word: studio | 1.434 |
| Sub-category: User Testing | −0.976 | Sub-category: Cybersecurity & Data Protection | 1.427 |
| Sub-category: Subtitles & Captions | −0.934 | Words: video course | 1.026 |
| Sub-category: Flyer Distribution | −0.876 | Sub-category: E-Commerce Development | 1.008 |
| Word: nature | −0.817 | Sub-category: Local Photography | 0.984 |
| Word: java | −0.788 | Word: produce | 0.903 |
| Sub-category: Transcripts | −0.786 | Sub-category: Resume Writing | 0.845 |
| Sub-category: Screencasting Videos | −0.760 | Sub-category: Whiteboard & Animated Explainers | 0.819 |

The prefixes `cat__` and `txt__` in the notebook output tell you whether the feature came from the category branch or the text branch.

---

## Section 5: The fair-exchange check (app workflow demo)

This is the app logic from our slides, running on top of the trained model:

```
Estimate the value of each service
            │
            ▼
Gap within 15% of the bigger value? ──yes──► FAIR EXCHANGE
            │ no
            ▼
Option 1: the person offering less adds CASH equal to the gap
Option 2: the person offering less adds an EXTRA SERVICE worth about the gap
```

**Important:** the dataset price is Fiverr's *starting price* in euros. These are estimates for comparing services against each other, not exact Kenyan prices.

### Cell 29: The functions

```python
def estimate_value(service, category, subcat, rating=np.nan, reviews=0):
    row = pd.DataFrame([{
        "rating": rating,
        "log_reviews": np.log1p(reviews),
        "has_reviews": int(reviews > 0),
        "Category": category,
        "Subcat": subcat,
        "service": service.lower(),
    }])
    return float(np.expm1(ridge_best.predict(row[X.columns]))[0])
```

`estimate_value` predicts the price (EUR) of **one** service:
1. Builds a one-row table with the same columns the model was trained on. If no rating or reviews are given, rating is missing (the pipeline fills it) and reviews are 0.
2. `row[X.columns]` puts the columns in the same order as in training.
3. `ridge_best.predict(...)` gives a log price, and `np.expm1` converts it to euros.

```python
def check_exchange(service_a, service_b, tolerance=0.15):
    value_a = estimate_value(**service_a)
    value_b = estimate_value(**service_b)
    gap = abs(value_a - value_b)
    gap_share = gap / max(value_a, value_b)
```
- Estimates both values (`**service_a` unpacks the dictionary into the function's inputs).
- `gap` is the difference in euros. `gap_share` is that gap as a fraction of the bigger value.
- `tolerance=0.15` means a gap of up to **15%** counts as fair. This number is adjustable.

```python
    if gap_share <= tolerance:
        print(f"\nFAIR EXCHANGE (gap is only {gap_share:.0%}).")
        return
```
If the gap is within tolerance, say it is fair and stop.

```python
    lower = "A" if value_a < value_b else "B"
    print(f"Option 1: Person {lower} adds a cash top-up of about EUR {gap:.2f}.")
```
Otherwise, work out who offers less and suggest they add the gap in **cash**.

```python
    closest = clean.iloc[(clean["price_eur"] - gap).abs().argsort()[:3]]
```
Option 2: look through the real listings in our data, find the **3 whose price is closest to the gap**, and suggest them as an extra service. (`.abs().argsort()` ranks listings by how close their price is to the gap.)

### Cell 29 demo output: legal writing vs website design

| Service | Estimated value |
|---|---|
| A: write a GDPR privacy policy and terms | EUR 26.94 |
| B: design and redesign a Wix website | EUR 102.79 |

Result: **NOT EQUAL**, gap EUR 75.85 (74%). Person A is offering less. Option 1: add a cash top-up of about EUR 75.85. Option 2: add an extra service worth about EUR 75.55 (for example, a PDF book design service).

### Cell 30 demo output: transcription vs podcast editing

| Service | Estimated value |
|---|---|
| A: transcribe English audio files | EUR 9.25 |
| B: edit and master your podcast | EUR 18.87 |

Result: **NOT EQUAL**, gap EUR 9.62 (51%). Person A is offering less. Option 1: add a cash top-up of about EUR 9.62. Option 2: add an extra service worth about EUR 8.89.

> The suggested extra services come from the closest *price*, not the closest *skill*, so some are unrelated (for example "impersonation" in Demo 2). Matching by skill is the job of the recommender, which is the next stage.

---

## Section 6: Validation plan (Question 4)

One score does not prove much. So we train other models using **exactly the same process** (same cleaning, same features, same 5-fold cross-validation, same test set) and compare.

| Model | Why we try it |
|---|---|
| Baseline (average) | The minimum bar any model must beat |
| **Ridge** (base model) | Keeps all features, shrinks their weights |
| Lasso | Also regularises, but sets unhelpful features to exactly zero, which shows which features really matter |
| Random Forest | Handles non-linear patterns (for example a word that matters only inside one category) that a linear model misses |

### Cell 32: The comparison loop

```python
models = {
    "Ridge (base)": (Ridge(), {"model__alpha": [0.1, 1, 5, 10, 20]}),
    "Lasso": (Lasso(max_iter=10000), {"model__alpha": [0.0005, 0.001, 0.005, 0.01, 0.05]}),
    "Random Forest": (RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1), None),
}
```
A dictionary listing each model and the `alpha` values to search. Random Forest has `None` because we do not tune it here.
- `max_iter=10000` gives Lasso enough steps to finish.
- `n_estimators=100` means 100 trees. `max_features="sqrt"` lets each split look at only a random subset of features. `n_jobs=-1` uses all CPU cores.

```python
rows = [baseline_scores]
for name, (estimator, grid_params) in models.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("model", estimator)])
```
Starts the results list with the baseline, then loops over each model, building the **same** preprocessor + model pipeline for each.

```python
    if grid_params:
        search = GridSearchCV(pipe, grid_params, cv=cv, scoring="neg_root_mean_squared_error")
        search.fit(X_train, y_train)
        fitted = search.best_estimator_
        label = f"{name} {search.best_params_}"
        cv_rmse = -search.best_score_
```
For Ridge and Lasso: search for the best `alpha` with the same 5-fold cross-validation, keep the best model, and record its cross-validated error.

```python
    else:
        fitted = pipe.fit(X_train, y_train)
        label = name
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
        cv_rmse = -cv_scores["test_score"].mean()
```
For Random Forest (nothing to tune): train it, then run the same 5-fold cross-validation to get a comparable score.

```python
    scores = evaluate(fitted, X_test, y_test, label)
    scores["CV RMSE (log)"] = cv_rmse
    rows.append(scores)

comparison = pd.DataFrame(rows).set_index("Model")
comparison.round(3)
```
Scores each finished model on the same unseen test set, adds its cross-validation error, and combines everything into one table.

**Output:**

| Model | Test RMSE (log) | R² | MAE (EUR) | Median error (EUR) | CV RMSE (log) |
|---|---|---|---|---|---|
| Baseline | 1.158 | −0.001 | 51.40 | 16.82 | n/a |
| Ridge (alpha=5) | **0.941** | **0.339** | 44.30 | 13.41 | **0.961** |
| Lasso (alpha=0.0005) | 0.946 | 0.331 | 44.63 | 13.77 | 0.971 |
| Random Forest | 0.944 | 0.333 | 44.53 | 13.40 | 0.970 |

### Cell 33: Comparison chart
```python
comparison["RMSE (log)"].sort_values().plot(kind="barh", color="teal")
```
A horizontal bar chart of test error per model (shorter bar = better).

### Cell 34: What the comparison shows

Ridge, Lasso and Random Forest score almost the same (test RMSE 0.94 to 0.95, R² 0.33 to 0.34). The gaps between them are smaller than the fold-to-fold variation in cross-validation (about 0.02), so **no model is truly better**.

This is a useful result: the limit comes from the **data** (what a listing tells us), not from the algorithm. We keep Ridge as the base model because it is the simplest and easiest to explain.

---

## Limitations and next steps

| Limitation | Detail |
|---|---|
| Starting price, not average price | Each row is the cheapest package in a listing, so estimates are lower-bound style values |
| Modest accuracy | R² of about 0.34, so use estimates as a rough guide |
| Missing value factors | Time, expertise, complexity and demand are not columns in this dataset. We use the title and review count as proxies |
| Euro prices from a global platform | Needs calibration with Kenyan service prices (for example, data from our own app users) |

**Next:**
1. Complete the model comparison and error analysis for the final project.
2. Build the second AI feature, the **recommender** (suggest providers and swap partners using skills, location and ratings).

---

## Quick glossary (for the presentation)

| Term | Simple meaning |
|---|---|
| Regression | Predicting a number (here, price) |
| Ridge | Linear regression with a penalty that stops overfitting |
| alpha | How strong that penalty is |
| Overfitting | Memorising the training data, then doing worse on new data |
| Cross-validation | Testing the model 5 times on different slices of the training data for a fairer score |
| Data leakage | Test data accidentally influencing training, which gives falsely good results |
| Log transform | Squashing huge values so a few expensive items do not dominate |
| TF-IDF | Turning text into word scores |
| One-hot encoding | Turning category names into 0/1 columns |
| Baseline | The "always guess the average" model every real model must beat |
| R² | Share of price variation explained (0 to 1) |
| Residual | Actual value minus predicted value |
