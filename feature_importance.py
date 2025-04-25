import matplotlib.pyplot as plt
import numpy as np

# Example feature names and importance scores
feature_names = ['Credit Score', 'Annual Income', 'DTI', 'Employment Stability', 'Age']
feature_importances = [0.35, 0.25, 0.20, 0.15, 0.05]

# Sort features by importance
indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(8,5))
plt.bar(range(len(feature_importances)), np.array(feature_importances)[indices], align="center")
plt.xticks(range(len(feature_importances)), np.array(feature_names)[indices], rotation=45)
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance Analysis")
plt.tight_layout()

# Save the figure
plt.savefig("feature_importance.png", dpi=300)
plt.show()
