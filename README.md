# EXPERIMENT
# Linear SVM: Decision Boundary and Margin

# Step 1: Import Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# Step 2: Load the Iris Dataset

iris = load_iris()

# Select Petal Length and Petal Width
X = iris.data[:, [2, 3]]
y = iris.target

print("Feature names:")
print(iris.feature_names)

print("\nDataset shape:")
print(X.shape)


# Step 3: Convert the Dataset into Binary Classification

# Setosa = 1
# Non-Setosa = 0

y_binary = np.where(y == 0, 1, 0)

print("\nClass distribution:")
print(pd.Series(y_binary).value_counts())


# Step 4: Split the Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Step 5: Standardize the Features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 6: Implement Linear SVM

svm_model = SVC(
    kernel="linear",
    C=1.0
)

svm_model.fit(X_train, y_train)


# Step 7: Make Predictions

y_pred = svm_model.predict(X_test)

print("\nPredicted labels:")
print(y_pred)


# Step 8: Evaluate the Model

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("\nModel Performance")
print("------------------")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-Score :", f1)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-Setosa",
            "Setosa"
        ]
    )
)


# Step 9: Obtain Model Parameters

w = svm_model.coef_[0]
b = svm_model.intercept_[0]

print("Weight Vector:")
print(w)

print("\nBias:")
print(b)


# Step 10: Calculate the Decision Boundary

x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1

xx = np.linspace(
    x_min,
    x_max,
    500
)

# Decision boundary:
# w0*x + w1*y + b = 0

decision_boundary = (
    -(w[0] * xx + b) / w[1]
)


# Step 11: Calculate the Margin Boundaries

# Positive margin:
# w0*x + w1*y + b = +1

margin_positive = (
    -(w[0] * xx + b - 1) / w[1]
)


# Negative margin:
# w0*x + w1*y + b = -1

margin_negative = (
    -(w[0] * xx + b + 1) / w[1]
)


# Step 12: Identify Support Vectors

support_vectors = svm_model.support_vectors_

print("\nNumber of Support Vectors:")
print(len(support_vectors))

print("\nSupport Vectors:")
print(support_vectors)


# Step 13: Visualize Decision Boundary and Margin

plt.figure(figsize=(10, 7))


# Setosa samples

plt.scatter(
    X_train[y_train == 1, 0],
    X_train[y_train == 1, 1],
    label="Setosa",
    marker="o"
)


# Non-Setosa samples

plt.scatter(
    X_train[y_train == 0, 0],
    X_train[y_train == 0, 1],
    label="Non-Setosa",
    marker="s"
)


# Decision Boundary

plt.plot(
    xx,
    decision_boundary,
    label="Decision Boundary"
)


# Positive Margin

plt.plot(
    xx,
    margin_positive,
    "--",
    label="Margin"
)


# Negative Margin

plt.plot(
    xx,
    margin_negative,
    "--"
)


# Support Vectors

plt.scatter(
    support_vectors[:, 0],
    support_vectors[:, 1],
    s=120,
    facecolors="none",
    edgecolors="black",
    linewidths=1.5,
    label="Support Vectors"
)


# Labels

plt.xlabel(
    "Standardized Petal Length"
)

plt.ylabel(
    "Standardized Petal Width"
)


# Graph Title

plt.title(
    "Linear SVM: Decision Boundary and Margin"
)

plt.legend()

plt.grid(True)

plt.show()


# Step 14: Calculate the Margin Width

margin_width = (
    2 / np.linalg.norm(w)
)

print("\nMargin Width:")
print(round(margin_width, 4))
