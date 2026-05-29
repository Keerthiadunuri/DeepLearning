import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("attrition.csv")

# =====================================================
# ENCODE ATTRITION
# =====================================================

df['Attrition'] = df['Attrition'].map({
    'Yes': 1,
    'No': 0
})

# =====================================================
# SELECT FEATURES
# =====================================================

X = df[
    [
        'Age',
        'MonthlyIncome',
        'JobSatisfaction',
        'DistanceFromHome',
        'TotalWorkingYears',
        'YearsAtCompany'
    ]
]

y = df['Attrition']

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# TRAIN MODEL
# =====================================================

model = RandomForestClassifier()

model.fit(X_train, y_train)

# =====================================================
# SAVE MODEL
# =====================================================

pickle.dump(model, open("random_forest.pkl", "wb"))

print("Model Saved Successfully!")