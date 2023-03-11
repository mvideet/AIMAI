import matplotlib.pyplot as plt
import numpy as np

mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))

plt.plot(x, y, linewidth=5, color='black')


plt.axis('off')
plt.show()
