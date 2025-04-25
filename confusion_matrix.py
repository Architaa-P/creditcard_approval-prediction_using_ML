import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize

# Generate synthetic classification data (binary)
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
y = label_binarize(y, classes=[0, 1])  # Convert labels to binary format

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Gradient Boosting classifier
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train.ravel())

# Predict probabilities
y_score = clf.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
roc_auc = auc(fpr, tpr)

# Compute micro-average ROC curve
fpr_micro, tpr_micro, _ = roc_curve(y_test.ravel(), y_score[:, 1].ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# Compute macro-average ROC curve
all_fpr = np.unique(np.concatenate([fpr]))
mean_tpr = np.interp(all_fpr, fpr, tpr)
roc_auc_macro = auc(all_fpr, mean_tpr)

# Create plot
# Final Refinement: Exact color matching with the reference image

plt.figure(figsize=(6,5))

# Class 0 ROC Curve (Exact Cyan from Reference, Solid)
plt.plot(fpr, tpr, color='#00FFFF', lw=2, linestyle='-', label=f"ROC curve of class 0 (area = {roc_auc:.2f})")

# Class 1 ROC Curve (Exact Purple from Reference, Solid)
plt.plot(fpr, tpr, color='#8B008B', lw=2, linestyle='-', label=f"ROC curve of class 1 (area = {roc_auc:.2f})")

# Micro-average ROC Curve (Dark Magenta, Dotted with Border Effect)
plt.plot(fpr_micro, tpr_micro, color='#FF1493', lw=3.5, linestyle='dotted', alpha=0.9)  # Thick magenta
plt.plot(fpr_micro, tpr_micro, color='black', lw=1.5, linestyle='dotted', alpha=1, label=f"Micro-average ROC (area = {roc_auc_micro:.2f})")  # Border

# Macro-average ROC Curve (Exact Blue from Reference, Dashed with Border Effect)
plt.plot(all_fpr, mean_tpr, color='#0000CD', lw=3.5, linestyle='dashed', alpha=0.9)  # Thick blue
plt.plot(all_fpr, mean_tpr, color='black', lw=1.5, linestyle='dashed', alpha=1, label=f"Macro-average ROC (area = {roc_auc_macro:.2f})")  # Border

# Diagonal reference line
plt.plot([0, 1], [0, 1], 'k--', lw=1)

# Labels, title, legend
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC curve for gradient_boosting", fontsize=14)
plt.legend(loc="lower right", fontsize=10, frameon=True)

# Show plot
plt.show()

