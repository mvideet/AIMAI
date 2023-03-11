from joblib import dump, load
import joblib
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

model= joblib.load("svm_model.pkl")
df = pd.read_csv("combo11.csv")

test_size = 0.05
num_splits = 10
df["embedded_symptoms"] = df.embedded_symptoms.apply(eval).apply(np.array)
print("Finished Embedding")
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])
scores = []
class_names = encoder.classes_





X_train, X_test, y_train, y_test = train_test_split(
        list(df.embedded_symptoms.values),
        df.label,
        test_size=0.05,
        random_state = 10
    )
X_test = np.array(X_test)
y_test = np.array(y_test)
YPred = model.predict(X_test)

y_test_decoded = encoder.inverse_transform(y_test)
Y_pred_decoded = encoder.inverse_transform(YPred)
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test_decoded, Y_pred_decoded)
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size':20})
plt.xlabel('Predicted Imaging Order', fontsize=20)
plt.ylabel('Actual Imaging Order', fontsize=20)
plt.show()
print(y_test_decoded)
print(YPred)
from sklearn.metrics import classification_report

class_report = classification_report(y_test, YPred, target_names=class_names)
print(class_report)