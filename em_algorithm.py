
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Generate sample data
np.random.seed(42)
n_samples = 300
X = np.concatenate([np.random.normal(0, 1, int(0.3 * n_samples)),
                    np.random.normal(5, 1, int(0.7 * n_samples))]).reshape(-1, 1)

# Create and fit the Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X)

# Generate points for plotting
x = np.linspace(-5, 10, 1000).reshape(-1, 1)
logprob = gmm.score_samples(x)
responsibilities = gmm.predict_proba(x)

# Plot the results
plt.figure(figsize=(10, 5))
plt.hist(X, bins=50, density=True, alpha=0.5)
plt.plot(x, np.exp(logprob), '-k')
plt.plot(x, responsibilities[:, 0] * np.exp(logprob), '--r')
plt.plot(x, responsibilities[:, 1] * np.exp(logprob), '--g')
plt.xlabel('Data')
plt.ylabel('Density')
plt.title('Gaussian Mixture Model and EM Algorithm')
plt.legend(['Estimated PDF', 'Component 1', 'Component 2'])
plt.savefig('em_algorithm_plot.png')
plt.show()

print("Means:", gmm.means_)
print("Covariances:", gmm.covariances_)
print("Weights:", gmm.weights_)
        