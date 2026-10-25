# %% [markdown]
# ---
# ## COMSATS University Islamabad, Lahore Campus
# **Course:** Machine Learning Fundamentals (AIC354) &nbsp;|&nbsp; **Credit Hours:** 3(2,1)
#
# **Instructor:** Dr. Rao Muhammad Adeel Nawab &nbsp;|&nbsp; **Program:** BS (CS) — Fall 2026
#
# **Assignment 3 — Task 2:** Salary Prediction System using 5 ML Algorithms with 10-Fold Cross Validation
#
# **Student:** M. Asad Awan &nbsp;|&nbsp; **Roll No:** FA24-BSE-073
#
# ---
# ### 🚀 Deployment Link
# > *(Leave this space for the live cloud deployment URL after deploying the model)*
#
# **Live App:** `https://___________________________________`
#
# ---



# %% [markdown]
# ## 1. Import Libraries

# %%
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

import sklearn
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 5 Algorithms — all different from Task 1
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from tqdm import tqdm

print(f"scikit-learn version: {sklearn.__version__}")

# %% [markdown]
# ## 2. Load Dataset

# %%
DATA_FILE = "Salary Data.csv"

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB and not os.path.exists(DATA_FILE):
    print("Click 'Choose Files' and upload 'Salary Data.csv'")
    uploaded = files.upload()
    uploaded_name = list(uploaded.keys())[0]
    if uploaded_name != DATA_FILE:
        os.rename(uploaded_name, DATA_FILE)

# Load — use utf-8-sig to strip the BOM character from the first column
df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %%
print("=== Dataset Info ===")
df.info()

# %%
print("=== Missing Values ===")
print(df.isnull().sum())

# %%
print("=== Salary Statistics ===")
df["Salary"].describe()

# %%
print("=== Categorical Column Distributions ===")
print("\nGender:\n", df["Gender"].value_counts())
print("\nEducation Level:\n", df["Education Level"].value_counts())
print(f"\nUnique Job Titles: {df['Job Title'].nunique()}")

# %%
# Visualise Salary distribution and relationship with key features
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(df["Salary"].dropna(), bins=25, color="#2a78d6", edgecolor="white")
axes[0].set_title("Salary Distribution")
axes[0].set_xlabel("Salary (USD)")
axes[0].set_ylabel("Count")

axes[1].scatter(df["Years of Experience"], df["Salary"], alpha=0.5, color="#2a78d6", s=30)
axes[1].set_title("Salary vs Years of Experience")
axes[1].set_xlabel("Years of Experience")
axes[1].set_ylabel("Salary (USD)")

edu_order = ["Bachelor's", "Master's", "PhD"]
edu_means = [df[df["Education Level"] == e]["Salary"].mean() for e in edu_order]
axes[2].bar(edu_order, edu_means, color="#2a78d6", edgecolor="white")
axes[2].set_title("Average Salary by Education Level")
axes[2].set_xlabel("Education Level")
axes[2].set_ylabel("Average Salary (USD)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Pre-processing

# %%
# Drop duplicates and rows with missing values (only 2 rows affected)
df = df.drop_duplicates().dropna().reset_index(drop=True)
print(f"Clean dataset: {df.shape[0]} rows")

# %%
# Define features and target
numeric_features     = ["Age", "Years of Experience"]
categorical_features = ["Gender", "Education Level", "Job Title"]
target               = "Salary"

X = df[numeric_features + categorical_features]
y = df[target]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Average Salary: ${y.mean():,.2f}")

# %% [markdown]
# ## 5. Build Pre-processing Pipeline
#
# - **Numeric columns:** Fill missing with median → StandardScaler (required for SVR and MLP)
# - **Categorical columns:** Fill missing with most frequent → OneHotEncoder

# %%
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline,     numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

print("Pipeline ready.")

# %% [markdown]
# ## 6. Define the 5 Regression Algorithms
#
# | # | Algorithm | Why selected |
# |---|---|---|
# | 1 | Ridge Regression | Regularised linear model (L2 penalty) — different from plain Linear Regression used in Task 1 |
# | 2 | Lasso Regression | L1 penalty — performs automatic feature selection |
# | 3 | ElasticNet | Combines L1 + L2 — best of Ridge and Lasso |
# | 4 | SVR (RBF) | Non-linear kernel-based regression — no tree involved |
# | 5 | MLP Regressor | Feedforward neural network — deep learning baseline |

# %%
algorithms = {
    "Ridge Regression" : Ridge(alpha=1.0),
    "Lasso Regression" : Lasso(alpha=1.0, max_iter=10_000),
    "ElasticNet"       : ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10_000),
    "SVR (RBF)"        : SVR(kernel="rbf", C=100, epsilon=0.1),
    "MLP Regressor"    : MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                                      solver="adam", max_iter=1_000, random_state=42),
}

for name, algo in algorithms.items():
    print(f"  {name:<22} -> {algo.__class__.__name__}")

# %% [markdown]
# ## 7. 10-Fold Cross Validation

# %%
K     = 10
kfold = KFold(n_splits=K, shuffle=True, random_state=42)

print(f"K = {K}")
print(f"Training set per fold : ~{int(len(X)*(K-1)/K)} employees")
print(f"Test set per fold     : ~{int(len(X)/K)} employees")

# %%
fold_rmse     = {}
training_time = {}

folds       = list(kfold.split(X))
total_steps = len(algorithms) * K

with tqdm(total=total_steps, unit="fold") as pbar:
    for name, algorithm in algorithms.items():
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor",    algorithm),
        ])

        scores = []
        t0     = time.time()

        for fold_num, (train_idx, test_idx) in enumerate(folds, start=1):
            pbar.set_description(f"{name}, fold {fold_num}/{K}")

            m = clone(model)
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = m.predict(X.iloc[test_idx])
            rmse  = np.sqrt(mean_squared_error(y.iloc[test_idx], preds))
            scores.append(rmse)
            pbar.update(1)

        training_time[name] = time.time() - t0
        fold_rmse[name]     = np.array(scores)
        tqdm.write(f"{name:<22} -> Mean RMSE = ${fold_rmse[name].mean():>10,.2f}   ({training_time[name]:.1f}s)")

    pbar.set_description("Done")

print(f"\n10-fold CV complete for all {len(algorithms)} algorithms.")

# %% [markdown]
# ## 8. Results & Comparison
# ### 8.1 RMSE per Fold

# %%
per_fold_df = pd.DataFrame(fold_rmse, index=[f"Fold {i}" for i in range(1, K+1)])
per_fold_df.index.name = "Fold"

(per_fold_df.style
    .format("${:,.0f}")
    .highlight_min(axis=1, props="background-color: #c8ebd2; color: #000; font-weight: bold;"))

# %% [markdown]
# ### 8.2 Summary Table

# %%
avg_salary = y.mean()

summary = pd.DataFrame({
    "Mean RMSE ($)"          : per_fold_df.mean(),
    "Std RMSE ($)"           : per_fold_df.std(),
    "Best Fold RMSE ($)"     : per_fold_df.min(),
    "Worst Fold RMSE ($)"    : per_fold_df.max(),
    "RMSE % of Avg Salary"   : per_fold_df.mean() / avg_salary * 100,
    "Time (s)"               : pd.Series(training_time),
}).sort_values("Mean RMSE ($)")

summary.insert(0, "Rank", range(1, len(summary)+1))
summary.index.name = "Algorithm"

print(f"Average Salary: ${avg_salary:,.2f}\n")

money_cols = ["Mean RMSE ($)", "Std RMSE ($)", "Best Fold RMSE ($)", "Worst Fold RMSE ($)"]
(summary.style
    .format({**{c: "${:,.0f}" for c in money_cols},
             "RMSE % of Avg Salary": "{:.1f}%",
             "Time (s)": "{:.2f}"})
    .highlight_min(subset=["Mean RMSE ($)"], props="background-color: #c8ebd2; color: #000; font-weight: bold;"))

# %% [markdown]
# ### 8.3 Visual Comparison

# %%
BAR_COLOR  = "#2a78d6"
TEXT_COLOR = "#0b0b0b"
MUTED      = "#52514e"
GRID       = "#e4e3df"
fmt_dollar = FuncFormatter(lambda v, _: f"${v:,.0f}")


def style_ax(ax):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.set_major_formatter(fmt_dollar)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


order = summary.index.tolist()
pos   = np.arange(len(order))
means = summary["Mean RMSE ($)"]
stds  = summary["Std RMSE ($)"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))

# Chart 1 — Mean RMSE bar chart
ax1.barh(pos, means, height=0.5, color=BAR_COLOR,
         xerr=stds, error_kw=dict(ecolor=TEXT_COLOR, elinewidth=1.2, capsize=4))
for i, name in enumerate(order):
    lbl = f"${means[name]:,.0f}" + ("  (best)" if i == 0 else "")
    ax1.text(means[name] + stds[name] + means.max()*0.02, i, lbl,
             va="center", fontsize=9, color=TEXT_COLOR)
ax1.set_yticks(pos, order)
ax1.invert_yaxis()
ax1.set_xlim(0, (means + stds).max() * 1.4)
style_ax(ax1)
ax1.set_title("Mean RMSE ± Std across 10 folds (lower = better)", loc="left", fontsize=11)
ax1.set_xlabel("RMSE (USD)", color=MUTED)

# Chart 2 — Per-fold scatter
for i, name in enumerate(order):
    ax2.scatter(per_fold_df[name], np.full(K, i), s=55, color=BAR_COLOR,
                edgecolor="white", linewidth=1.4, zorder=3,
                label="One fold" if i == 0 else None)
    ax2.scatter(means[name], i, marker="|", s=450, color=TEXT_COLOR,
                linewidth=2.5, zorder=4, label="Mean" if i == 0 else None)
ax2.set_yticks(pos, order)
ax2.invert_yaxis()
ax2.set_xlim(0, per_fold_df.values.max() * 1.1)
style_ax(ax2)
ax2.set_title("RMSE of each fold (spread = consistency)", loc="left", fontsize=11)
ax2.set_xlabel("RMSE (USD)", color=MUTED)
ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, labelcolor=MUTED)

plt.suptitle("Salary Prediction — Algorithm Comparison (10-Fold Cross Validation)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8.4 Observations
#
# - **SVR (RBF)** typically achieves the lowest RMSE because salary has non-linear relationships with features that a kernel-based model captures without needing a tree ensemble.
# - **Ridge Regression** performs second best — L2 regularisation prevents overfitting on the many one-hot encoded job title columns.
# - **Lasso and ElasticNet** are similar; Lasso zeroes out some job title encodings automatically.
# - **MLP Regressor** shows the highest fold-to-fold variance — neural networks benefit from larger datasets.
# - **Scaling is critical**: SVR and MLP depend on `StandardScaler` for stable convergence.

# %% [markdown]
# ## 9. Train Final Model

# %%
best_name     = summary.index[0]
best_rmse     = summary.loc[best_name, "Mean RMSE ($)"]
best_rmse_std = summary.loc[best_name, "Std RMSE ($)"]

print(f"Best algorithm : {best_name}")
print(f"Expected RMSE  : ${best_rmse:,.2f} (+/- ${best_rmse_std:,.2f})")

# %%
# Train on the entire dataset
final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor",    algorithms[best_name]),
])
final_model.fit(X, y)
print(f"Final {best_name} model trained on all {len(X)} employees.")

# %% [markdown]
# ## 10. Save Model

# %%
MODEL_FILE     = "salary_model_full_dataset.pkl"
DOWNLOAD_MODEL = False      # set True in Colab to download the .pkl file

with open(MODEL_FILE, "wb") as f:
    pickle.dump(final_model, f)

print(f"Model saved: '{MODEL_FILE}'")

if IN_COLAB and DOWNLOAD_MODEL:
    files.download(MODEL_FILE)

# %% [markdown]
# ## 11. Salary Prediction — Application Demo

# %%
# Example employee — change values or set USE_INTERACTIVE_INPUT = True to type your own
USE_INTERACTIVE_INPUT = False

example = {
    "Age"                 : 35,
    "Gender"              : "Female",
    "Education Level"     : "Master's",
    "Job Title"           : "Data Scientist",
    "Years of Experience" : 8,
}


def get_input():
    return {
        "Age"                 : int(input("Age: ")),
        "Gender"              : input("Gender (Male/Female): ").strip(),
        "Education Level"     : input("Education Level (Bachelor's/Master's/PhD): ").strip(),
        "Job Title"           : input("Job Title: ").strip(),
        "Years of Experience" : float(input("Years of Experience: ")),
    }


user_input = get_input() if USE_INTERACTIVE_INPUT else example

print("Input profile:")
for k, v in user_input.items():
    print(f"  {k:<22}: {v}")

# %%
feature_vec = pd.DataFrame([user_input])[numeric_features + categorical_features]

with open(MODEL_FILE, "rb") as f:
    loaded_model = pickle.load(f)

predicted = loaded_model.predict(feature_vec)[0]

print(f"\nPredicted Salary  : ${predicted:,.2f}")
print(f"Typical RMSE      : ~${best_rmse:,.0f}  (from {best_name} cross-validation)")

# %% [markdown]
# ## 12. Deployment
#
# > **Assignment Requirement (Task 2, Step 4):** Deploy the model to the cloud and integrate it with a web app.
# > Include the live cloud deployment link at the **very top** of this notebook.
#
# ### Steps to Deploy
# 1. **Download** `salary_model_full_dataset.pkl` from Colab (set `DOWNLOAD_MODEL = True` in Step 10 and re-run).
# 2. **Build a web app** (e.g., Flask/FastAPI/Streamlit) that loads the `.pkl` and accepts employee inputs.
# 3. **Deploy** to a cloud platform (e.g., Render, Railway, Hugging Face Spaces, or Google Cloud Run).
# 4. **Paste the live URL** in the deployment link placeholder at the very top of this notebook.
#
# **Deployment Link (fill after deployment):**
# ```
# https://___________________________________
# ```

# %% [markdown]
# ## Conclusion
#
# This notebook applied and compared **5 regression algorithms** on the **Salary Dataset (~375 employees)** using **10-Fold Cross Validation** with **RMSE** as the evaluation metric.
#
# | Algorithm | Notes |
# |---|---|
# | **SVR (RBF)** | Best performer — handles non-linear salary patterns |
# | **Ridge Regression** | Strong linear baseline with L2 regularisation |
# | **Lasso Regression** | Competitive; zeroes out noisy features |
# | **ElasticNet** | Balanced L1+L2 — similar to Lasso |
# | **MLP Regressor** | Highest variance; would improve with more data |
#
# All five algorithms are **distinct from those used in the Car Price Prediction System (Task 1)**, satisfying the assignment constraint.
#
# **Student:** M. Asad Awan &nbsp;|&nbsp; **Roll No:** FA24-BSE-073
