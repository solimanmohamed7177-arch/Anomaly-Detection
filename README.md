# Anomaly-Detection
Task 1 – Anomaly Detection (Boston Housing, Isolation Forest) — What I Did and Why

Goal (in plain English):
Build an unsupervised model that flags unusual homes (outliers/anomalies) in the Boston Housing dataset, and show that the same model can also flag anomalies for new inputs.


---

1) Get the data

What I did: Downloaded the dataset from Kaggle (Boston House Price) and loaded it into a pandas DataFrame df.
Why it matters: We need a clean table of numeric features to train any anomaly detector. I also verified that the CSV parsed correctly (some sources can load as a single column if the delimiter is misread).
Expected output: A DataFrame with columns like CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV.


---

2) Basic data health checks (EDA)

What I did:

Checked missing values with df.isnull().sum() to confirm the dataset is complete.

Printed a statistical summary with df.describe() to see ranges (min/max), central tendency (mean/median), and spread (std).


Why it matters:

Missing values or impossible ranges can look like anomalies, but they’re just data quality issues. We want to detect real outliers, so we confirm data quality first.


Expected output:

Missing values ideally all zeros.

Summary shows realistic ranges (e.g., CRIM right-skewed, RM around ~5–8, MEDV capped in many versions at 50.0).



---

3) Visual hints for outliers

What I did:

Plotted a boxplot for CRIM, RM, and MEDV to see extreme points.

Drew a scatter plot: CRIM (x) vs MEDV (y) to visualize the relationship between crime rate and price.


Why it matters:

Boxplots reveal potential outliers fast.

Scatter helps spot points that don’t follow the typical pattern (e.g., very high crime with very high price).


Expected output:

Boxplot dots outside the whiskers indicate possible outliers.

Scatter shows a general negative trend (higher CRIM → lower MEDV), with a few points deviating strongly.



---

4) Choose features and scale them

What I did:

Used all columns as features for detection (including MEDV) by copying df into X.

Applied StandardScaler to get X_scaled (mean 0, std 1 per feature).


Why it matters:

Including MEDV means we also detect price-based anomalies (e.g., a price that doesn’t match the rest of the attributes).

Scaling prevents big-magnitude features (like TAX) from dominating small ones (like binary CHAS).


Expected output:

A scaled numeric array with the same number of rows as df and all features standardized.



---

5) Train the anomaly detector (Isolation Forest)

What I did:

Initialized IsolationForest with n_estimators=100, contamination='auto', and a fixed random_state for reproducibility.

Trained it on X_scaled without any labels (unsupervised).


Why it matters (plain English):

Isolation Forest builds many random trees that split the data. Unusual points get isolated quickly (fewer splits), so they receive a more “anomalous” score and are labeled as anomalies.


Expected output:

A fitted model ready to score each row as Normal (1) or Anomaly (-1).



---

6) Flag anomalies in the dataset

What I did:

Ran predictions on the whole dataset: 1 = Normal, -1 = Anomaly.

Counted how many anomalies were found.

Added a new column Anomaly to df with values "Normal" / "Anomaly".

Printed a few example rows that were flagged.


Why it matters:

This shows the detector’s decisions and how common/rare anomalies are in this dataset.


Expected output:

A message like: Number of detected anomalies: X out of 506.

A small sample (e.g., first 5) of rows labeled "Anomaly" so we can inspect their values.



---

7) Visualize and analyze the flagged points

What I did:

Plotted CRIM vs MEDV again, but colored points by the model’s label (blue = Normal, red = Anomaly).

Calculated a statistical summary only for the anomalies.

Compared mean values of key features (CRIM, RM, MEDV) between Normal vs Anomaly groups.


Why it matters:

Visualization helps verify that red points are indeed unusual.

Comparing means shows whether anomalies differ in sensible ways (e.g., much higher CRIM, extreme RM, or unexpected MEDV given the other features).


Expected output:

A scatter with clearly highlighted red points.

Descriptive stats for anomalies showing notable differences.

Mean comparison where anomalies often have more extreme averages (e.g., higher CRIM or atypical MEDV).



---

8) Test on new input data

What I did:

Created a new row with the same feature order as training.

Applied the same scaler and used the model to predict "Normal" or "Anomaly".


Why it matters:

Proves the model can score unseen inputs, which is required by the task.


Expected output:

A message like: New data prediction: Normal (or Anomaly), depending on how unusual the new row is.



---

9) How Isolation Forest detects outliers (no math)

It builds many random trees that split features at random thresholds.

Rare/far points need fewer splits to be isolated from the rest of the data.

The model aggregates across trees: if a point is consistently isolated quickly, it gets labeled as an anomaly.

This works without labels and scales well to tabular data.



---

10) Interpreting the results

An anomaly is not necessarily a mistake; it can be a valid but rare combination (e.g., unusually high ZN with high PTRATIO and large DIS but mid-range MEDV).

Use the flagged rows to double-check data quality or to investigate interesting edge cases.

If too many or too few points are flagged, adjust contamination (e.g., 0.05 for ~5% anomalies).
