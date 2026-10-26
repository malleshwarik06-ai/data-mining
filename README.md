🎵 Spotify Track Popularity

*Preprocessing \& ML Pipelines*

===

**CS4168 Data Mining · MSc Software Engineering · University of Limerick (2025/26)**

What makes a song popular? This group project analyses 2,000 Spotify tracks across five genres (pop, indie-pop, synth-pop, hip-hop, R\&B) to understand and predict track popularity, using exploratory analysis, clustering, classification and regression.

This repository contains **my part of the project: data preprocessing and the scikit-learn pipelines** that fed every model in the team.

\---

## 🔧 What I did

**Data audit**

* Checked all 17 columns for missing values, duplicates, data types, valid ranges and skewness
* Found 32 duplicate tracks and removed them (2,000 → 1,968 rows)
* Showed that the \~2% of missing values came as one block of rows (the same tracks were missing several audio features), which justified median imputation

**Feature engineering**

* Wrote a custom `Log1pTransformer` (scikit-learn `BaseEstimator` + `TransformerMixin`) for heavily right-skewed features: speechiness, acousticness, instrumentalness and liveness
* Treated musical `key` as categorical (pitch class, no natural order) and one-hot encoded it together with `track\_genre`
* Converted `explicit` from boolean to 0/1

**Three task-specific pipelines** built with `ColumnTransformer`

|Task|Features|Notes|
|-|-|-|
|Clustering (k-means, DBSCAN)|13|Genre and popularity excluded; all features standardised|
|Classification (high vs low popularity)|30|Binary target from the median popularity (45); stratified split|
|Regression (popularity score 0–100)|30|Same preprocessing, continuous target|

**Preventing data leakage**

* Split train/test (80/20) *before* fitting any preprocessing
* Wrapped preprocessing and model in a single `Pipeline`, so cross-validation (5-fold) re-fits the preprocessing inside each fold
* Exported the preprocessed arrays so teammates could load them straight into their modelling notebooks

\---

## 📊 Team results

* **Random Forest** was the best model for both classification (highest F1 and ROC-AUC) and regression (lowest error, highest R²)
* **Audio features** such as energy, loudness and danceability explained popularity better than genre labels
* Predicting **high vs low popularity** worked much better than predicting the exact score, because external factors (trends, artist fame) aren't in the audio data
* Consistent scaling and proper pipelines were identified as a decision that **improved model stability**

\---

## 🗂️ Files

|File|Description|
|-|-|
|`preprocessing\_pipelines.ipynb`|Data audit, EDA, cleaning and pipeline construction|
|`tracks2026.csv`|Dataset (excerpt of the Spotify Tracks Dataset, provided by the module)|
|`presentation.pdf`|Team presentation slides|

## ▶️ How to run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook preprocessing\_pipelines.ipynb
```

Keep `tracks2026.csv` in the same folder as the notebook.

\---

## 👥 Team

|Member|Responsibility|
|-|-|
|**Malleswari Kurva**|Data preprocessing, feature scaling, encoding and pipeline setup|
|Neha Vinod|EDA, visualisation and pattern analysis|
|Ganga Bhavani Chintalapalli|Clustering (k-means \& DBSCAN)|
|Lekha Murthy|Classification modelling|
|Yaswitha Vasipalli|Regression modelling and analysis|

## 🛠️ Tech

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter

