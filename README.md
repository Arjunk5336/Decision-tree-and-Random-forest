# Heart Disease Prediction using Decision Tree and Random Forest

This project demonstrates how to build and visualize decision tree and random forest classifiers to predict the presence of heart disease using a medical dataset.

Dataset:

The dataset used is a CSV file named heart.csv, which contains several medical attributes related to heart disease. The target column (target) indicates the presence (1) or absence (0) of heart disease.

Features:

Data Preprocessing: Separates features and labels from the dataset.

Train/Test Split: Splits the dataset into training (80%) and testing (20%) sets.

Decision Tree Classifier:

Trains a full decision tree.

Visualizes the tree using plot_tree.

Also creates a pruned version of the tree (with max_depth=4) for better generalization.


Evaluation:

Accuracy Score

Confusion Matrix

Classification Report


Random Forest Classifier (Optional but imported): Can be integrated for better performance.

Visualization: Includes tree visualization with Matplotlib and optional Graphviz export.
