from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

# Load the iris dataset as an example
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features for visualization purpose
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a SVM model
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')

# Create a meshgrid of X1 and X2 features
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Obtain the decision boundary from the model
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary on top of the scatter plot
plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM Decision Boundary Visualization')
plt.show()
