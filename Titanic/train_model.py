import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# =====================================================
# LOAD DATASET
# =====================================================

data = pd.read_csv("Titanic-Dataset.csv")

# =====================================================
# SELECT FEATURES
# =====================================================

X = data[["Pclass", "Age", "Fare"]]

# Fill missing Age values
X["Age"] = X["Age"].fillna(X["Age"].mean())

y = data["Survived"]

# =====================================================
# NORMALIZATION
# =====================================================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# BUILD ANN MODEL
# =====================================================

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# TRAIN MODEL
# =====================================================

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32
)

# =====================================================
# SAVE MODEL
# =====================================================

model.save("titanic_ann_model.h5")

print("Model Saved Successfully")