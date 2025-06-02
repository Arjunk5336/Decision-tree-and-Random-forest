import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import graphviz
import seaborn as sns

df = pd.read_csv(r"C:\Users\AL SharQ\Downloads\intern\heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

dt_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_pruned.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_pred_pruned = dt_pruned.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importances from Random Forest")
plt.tight_layout()
plt.show()

cv_scores_dt = cross_val_score(dt_pruned, X, y, cv=5)
cv_scores_rf = cross_val_score(rf, X, y, cv=5)

print("Decision Tree Accuracy:", accuracy_dt)
print("Pruned Tree Accuracy (max_depth=4):", accuracy_pruned)
print("Random Forest Accuracy:", accuracy_rf)
print("Decision Tree CV Score (mean):", cv_scores_dt.mean())
print("Random Forest CV Score (mean):", cv_scores_rf.mean())