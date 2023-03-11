import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

model = joblib.load("svm_model.pkl")
print("loaded model and randomized data")
df = pd.read_csv("combo11.csv", nrows=900)

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
tsne = TSNE(n_components=3)
tsne_data = tsne.fit_transform(np.vstack(df['embedded_symptoms'].values))
print("finished transforming")
predictions = model.predict(df['embedded_symptoms'].values.tolist())
print("finished predicting")

class_names = encoder.inverse_transform(np.unique(df['label']))

# Create a color map
colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
color_map = dict(zip(class_names, colors))

# Plot the 3D t-SNE visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for name, x, y, z in zip(df['label'], tsne_data[:, 0], tsne_data[:, 1], tsne_data[:, 2]):
    ax.scatter(x, y, z, c=color_map[encoder.inverse_transform([name])[0]])
ax.set_xlim(-60,60)
ax.set_ylim(-20,20)
ax.set_zlim(-30,30)
# Add the legend
patches = [plt.plot([], [], marker="o", ls="", color=color_map[name],
                    label=name)[0] for name in class_names]
ax.legend(handles=patches, loc='upper left')
plt.show()