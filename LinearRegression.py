from sklearn import linear_model
from scipy import stats
import numpy as np


class LR(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    #def __init__(self, *args, **kwargs):
        #super(LR, self).__init__(*args, **kwargs)
    
    def __init__(self, fit_intercept=True):
        super(LR, self).__init__(fit_intercept=fit_intercept)

    def fit(self, X, y, n_jobs=1):
        self = super(LR, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])

        self.t = self.coef_ / se
        self.p = (2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))).flatten()
        return self