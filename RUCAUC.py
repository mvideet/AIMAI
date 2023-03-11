import joblib
import numpy as np
import openai
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

openai.api_key = "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"

model= joblib.load("svm_model_probability.pkl")
print("Loaded model")
df = pd.read_csv("combo11.csv", nrows=600)
print("Loaded dataset")
df["embedded_symptoms"] = df.embedded_symptoms.apply(eval).apply(np.array)

print("Finished Embedding")
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])
print("Finished embedding")

X_train, X_test, y_train, y_test = train_test_split(
        list(df.embedded_symptoms.values),
        df.label,
        test_size=0.8,
        random_state = 10
    )
y_probs = model.predict_proba(X_test)

# Define the names of the classes
class_names = list(encoder.classes_)

# Set up the subplot grid
fig, ax = plt.subplots(4, 2, figsize=(10, 10), sharey=True, sharex=True)
ax = ax.ravel()

# Plot the ROC curves for each class
for i in range(8):
    fpr, tpr, _ = roc_curve(y_test == i, y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax[i].plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})', marker='', markerfacecolor='blue', markersize=2, linestyle='-')
    ax[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[i].set_xlim([0.0, 1.0])
    ax[i].set_ylim([0.0, 1.05])
    ax[i].set_xlabel('False Positive Rate')
    ax[i].set_title(f'ROC Curve for {class_names[i]}', color = 'lightblue', fontsize = 10)
    if i == 0 or i == 4:
        ax[i].set_ylabel('True Positive Rate')



plt.tight_layout()
plt.show()
