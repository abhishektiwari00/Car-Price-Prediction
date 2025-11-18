# train.py
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers

# 1. Load data
df = pd.read_csv("CarPrice_Assignment.csv")

# 2. Numeric columns (including target 'price')
NUMERIC_COLUMNS = [
    "symboling",
    "wheelbase",
    "carlength",
    "carwidth",
    "carheight",
    "curbweight",
    "enginesize",
    "boreratio",
    "stroke",
    "compressionratio",
    "horsepower",
    "peakrpm",
    "citympg",
    "highwaympg",
    "price",
]

df = df[NUMERIC_COLUMNS].dropna()

# 3. Split features/target
X = df.drop("price", axis=1).values
y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

input_dim = X_train.shape[1]

# 4. Define Keras model
model = keras.Sequential([
    layers.Input(shape=(input_dim,), name="input_layer"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, name="output_layer"),
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 5. Train
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1,
)

# 6. Evaluate
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE: {mae:.2f}")

# 7. Save using modern Keras format (.keras)
model.save("model.keras")
print("Saved model to model.keras")
