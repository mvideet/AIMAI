import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn import svm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

model = joblib.load("svm_model.pkl")
print("loaded model and randomized data")
df = pd.read_csv("combo11.csv")
df = df.sample(n=900)
print("loaded model and randomized data")
df["embedded_symptoms"] = df.embedded_symptoms.apply(eval).apply(np.array)
print("Finished Embedding")
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])

X_train, X_test, y_train, y_test = train_test_split(
        list(df.embedded_symptoms.values),
        df.label,
        test_size=0.75,
        random_state = 10
    )
X_test = np.array(X_test)
y_test = np.array(y_test)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(np.vstack(df['embedded_symptoms'].values))
print("finished transforming")
predictions = model.predict(df['embedded_symptoms'].values.tolist())
print("finished predicting")
class_names = encoder.inverse_transform(np.unique(df['label']))

# Create a color map
colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
color_map = dict(zip(class_names, colors))

# Plot the scatter plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for name, x, y in zip(df['label'], pca_data[:, 0], pca_data[:, 1]):
    ax.scatter(x, y, c=color_map[encoder.inverse_transform([name])[0]])

# Add the legend
patches = [plt.plot([], [], marker="o", ls="", color=color_map[name],
                    label=name)[0] for name in class_names]
plt.legend(handles=patches, loc='best')
plt.show()