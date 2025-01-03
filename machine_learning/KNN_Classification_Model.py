"""
This program trains k-nn models that aim to
predict the percentage of low income household.
Author: Huiming Zhang
Date: 03/10/2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ======================
# Step 1: Data Loading and Preprocessing
# ======================

# Read CSV file
df = pd.read_csv('communities_modified.csv')

# Define the bins and labels for categorizing income
income_bins = [0, 20, 50, 100]  # Define bin edges
income_labels = ['Low', 'Medium', 'High']  # Labels for the bins

# Categorize the income percentage into "Low", "Medium", and "High"
df['income_category'] = pd.cut(df['Equivalent household income <$600/week, %'], bins=income_bins, labels=income_labels)

# Features selected for the model (independent variables)
features = ['Requires assistance with core activities, %',
            'Did not complete year 12, %',
            'Holds degree or higher, %',
            'ARIA+ (avg)',
            '2012 ERP age 70+, %']

# Target variable (income category)
target = df['income_category']

# Normalize the feature data to a range between 0 and 1
scaler = MinMaxScaler()
X = scaler.fit_transform(df[features])

# ======================
# Step 2: Data Splitting
# ======================

# Split data into training/validation (90%) and test (10%) sets
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.1, random_state=123, stratify=target)

# ======================
# Step 3: Model Training and Hyperparameter Tuning
# ======================

best_k = None
best_accuracy = 0
k_values = list(range(1, 21))  # Range of k values to try for k-NN

# Find the best k by training on the training set and validating on the validation set
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)

    # Use StratifiedKFold to split
    stratified_kfold = StratifiedKFold(n_splits=9, shuffle=True, random_state=123)

    # Apply cross validation to evaluate
    cv_scores = cross_val_score(model, X_train, y_train, cv=stratified_kfold, scoring='accuracy')

    # Calculate mean accuracy over the 10 folds
    mean_cv_accuracy = np.mean(cv_scores)

    # Store the best k based on validation accuracy
    if mean_cv_accuracy > best_accuracy:
        best_accuracy = mean_cv_accuracy
        best_k = k

print(f'Best k found: {best_k} with Validation Accuracy={best_accuracy:.4f}')

# ======================
# Step 4: Model Evaluation on Test Set
# ======================

# Train model with best k
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)

# Predict on the test set and evaluate the model's accuracy
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.4f}')

precision = precision_score(y_test, y_test_pred, average=None)
recall = recall_score(y_test, y_test_pred, average=None)
f1 = f1_score(y_test, y_test_pred, average=None)

# information regarding model performance on individual classes
data = {
    'Class': ['High', 'Low', 'Medium'],
    'Recall': recall,
    'Precision': precision,
    'F1 Score': f1
}
accuracy_df = pd.DataFrame(data)
print(accuracy_df)

# ======================
# Step 5: Confusion Matrix Visualization
# ======================

# Generate confusion matrix for test set predictions
cm = confusion_matrix(y_test, y_test_pred, labels=income_labels)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=income_labels)
disp.plot()

# Add titles and labels
plt.title("Confusion Matrix for Income Category Classification")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

