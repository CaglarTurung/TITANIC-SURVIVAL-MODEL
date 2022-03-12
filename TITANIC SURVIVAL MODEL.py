#TITANIC SURVIVAL MODEL

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from helpers.eda import *
from helpers.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)



#DESCRIPTIVE STATISTICS
df.describe().T
df.shape
check_df(df)

# FEATURE ENGINEERING
df = pd.read_pickle(r"E:\CAGLAR\titanic_prep.pkl")

check_df(df)

# MODELING
y = df["SURVIVED"]
X = df.drop(["SURVIVED"], axis=1)
log_model = LogisticRegression(max_iter=100000).fit(X, y)
log_model.intercept_ #---> B = -0.24
log_model.coef_   #---> w =

cv_results = cross_validate(log_model,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# Accuracy: 0.8249063670411985

cv_results['test_precision'].mean()
# Precision: 0.7925152411317675

cv_results['test_recall'].mean()
# Recall:  0.7395798319327731

cv_results['test_f1'].mean()
# F1-score: 0.7614244440185146

cv_results['test_roc_auc'].mean()
# AUC: 0.8671542879778175

# Let's predict a random user's survival.
random_user = X.sample(1, random_state=42)

log_model.predict(random_user)  # 1
