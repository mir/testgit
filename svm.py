import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

SEED = 45
rng = np.random.default_rng(SEED)

def get_data(means=None, n_samples = 100, n_centers=2, seed=SEED, rng=None):
    """
        Generates dataset with n_centers clusters
        with 2 features and 1 label.

        Each cluster has 2d normal distribution with
        random variance (from 0.1 to 0.5) and mean (from -1 to 1)
    """
    if not rng:
        rng = np.random.default_rng(seed)


    means = np.random.uniform(-1,1, size=(n_centers, 2)) if not means else means
    variances = np.random.uniform(0.01, 0.5/n_centers, size=(n_centers, 2))
    X = np.zeros((0,2))
    y = np.zeros((0,1))    
    for i in range(n_centers):
        X = np.concatenate((X, rng.multivariate_normal(means[i], np.diag(variances[i]), size=n_samples)))
        y = np.concatenate((y, np.ones((100,1))*i))
        
    return X, y

def shuffle_data(X, y, seed=SEED, rng=None):
    """
        Shuffles data and labels
    """
    if not rng:
        rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    return X[idx], y[idx]

X, y = shuffle_data(*get_data(n_centers=3,rng=rng),seed=SEED)

model = SVC(kernel='linear', C=1, gamma=1)
model.fit(X, y.ravel())

disp = DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,        
        xlabel='x',
        ylabel='y',
    )
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
# set ticks
plt.xticks(np.arange(-1, 1.5, 0.5))
# set xlabel
plt.xlabel('x')
plt.ylabel('y')
plt.show()