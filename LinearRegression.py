import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy import stats



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


class SubSampleRegression:
    s_name = None
    fit_done = False
    version = 1.001

    def __init__(self, p_value) -> None:
        self.p_margin = p_value

    def Fit(self, data: pd.DataFrame, X_names: list, y_name: str, s_name=None):  
        assert all([x in data.columns for x in X_names]), "SubSampleRegression: в данных нет переменных X_name"
        assert y_name in data.columns, "SubSampleRegression: в данных нет переменных y_name"

        self.X_names = X_names

        if s_name: 
            assert s_name in data.columns, "SubSampleRegression: в данных нет переменных s_name"
            self.X_names = X_names
            self.s_name = s_name
            self.fit_internal_split(data, y_name)
        else:
            self.X_names = X_names
            self.fit_internal_total(data, y_name)
        
        return self

    def fit_internal_split(self, data, y_name): 
        self.betas = pd.DataFrame(columns=self.X_names)
        self.p_values = pd.DataFrame(columns=self.X_names)

        for sample, data_chunk in data.groupby(self.s_name):
            reg = LR(fit_intercept=False) 
            reg.fit(data_chunk[self.X_names], data_chunk[y_name])
            self.betas.loc[sample, :] = pd.Series(reg.coef_, index=self.X_names)
            self.p_values.loc[sample, :] = pd.Series(reg.p, index=self.X_names)
        
        self.set_significant_betas_internal()
        self.fit_done = True

    def fit_internal_total(self, data, y_name): 
        #self.betas = pd.DataFrame(columns=self.X_names)
        #self.p_values = pd.DataFrame(columns=self.X_names)

        reg = LR(fit_intercept=False) 
        reg.fit(data[self.X_names], data[y_name])
        self.betas = pd.Series(reg.coef_, index=self.X_names)
        self.p_values = pd.Series(reg.p, index=self.X_names)
        self.set_significant_betas_internal()
        
        self.fit_done = True

    def set_significant_betas_internal(self): 
        self.betas_significant = self.betas.where(self.p_values < self.p_margin, 0)

    def Contributions(self, data): 
        assert self.fit_done, "Еще не обучена"
        if self.s_name:
            return self.contributions_internal_split(data)
        else: 
            return self.contributions_internal_total(data)

    def contributions_internal_split(self, data): 
        assert all([x in data.columns for x in self.X_names]), "SubSampleRegression: в данных нет переменных X_name"
        assert self.s_name in data.columns, "SubSampleRegression: в данных нет переменных s_name"
        return data[self.X_names].set_index([data.index, data[self.s_name]]).mul(self.betas_significant, level=1).set_index(data.index)
    
    def contributions_internal_total(self, data): 
        assert all([x in data.columns for x in self.X_names]), "SubSampleRegression: в данных нет переменных X_name"
        return data[self.X_names].mul(self.betas_significant)
    
    def Predict(self, data): 
        return self.Contributions(data).sum(axis=1)
    
    def Score(self, data: pd.DataFrame, y_name: str) -> float: 
        #y_predicted = self.Predict(data)
        #y_actual = data[y_name] 
        #return 1 - ((y_actual - y_predicted) ** 2).sum() / ((y_actual - y_actual.mean()) ** 2).sum()
        return r2_score(data[y_name], self.Predict(data))