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
    This class sets the intercept to 0 by default, since usually we include it in X.
    """

    #def __init__(self, *args, **kwargs):
        #super(LR, self).__init__(*args, **kwargs)
    
    def __init__(self, fit_intercept=True, positive=False):
        super(LR, self).__init__(fit_intercept=fit_intercept, positive=positive)

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

    def __init__(self, p_margin: float=0.05, second_run: bool=False, force_positive: bool=True):
        """
        p_margin - значение p выше которого считать бетту незначимой , 
        second_run == False - обычная регрессия, незначимые бетты обнуляются
        second_run == True - выкидываем переменные с незначимыми беттами и строим регрессию заново
        """
        self.p_margin = p_margin
        self.second_run = second_run
        self.positive = force_positive

    def Fit(self, data: pd.DataFrame, X_names: list, y_name: str, s_name=None):  
        assert all([x in data.columns for x in X_names]), "SubSampleRegression: в данных нет переменных X_name"
        assert y_name in data.columns, "SubSampleRegression: в данных нет переменных y_name"

        self.X_names = X_names

        if s_name: 
            assert s_name in data.columns, "SubSampleRegression: в данных нет переменных s_name"
            self.X_names = X_names
            self.s_name = s_name
            self.__FitSplit(data, y_name)
        else:
            self.X_names = X_names
            self.__FitTotal(data, y_name)
        
        return self

    def __FitSubSample(self, data, y_name): 
        betas = pd.Series(0,index=self.X_names)
        p_values = pd.Series(0, index=self.X_names)

        X_names_for_reg = list(self.X_names)
        reg = LR(fit_intercept=False, positive=self.positive).fit(data[X_names_for_reg], data[y_name])
        if self.second_run:
            X_names_for_reg = [name for i, name in enumerate(X_names_for_reg) if reg.p[i] < self.p_margin]
            reg = LR(fit_intercept=False, positive=self.positive).fit(data[X_names_for_reg], data[y_name])

        for i, name in enumerate(X_names_for_reg): 
            betas[name] = reg.coef_[i]
            p_values[name] = reg.p[i]
        
        return betas, p_values
    
    def __FitSplit(self, data, y_name): 
        """self.betas = pd.DataFrame(columns=self.X_names)
        self.p_values = pd.DataFrame(columns=self.X_names)

        for sample, data_chunk in data.groupby(self.s_name):
            reg = LR(fit_intercept=False) 
            reg.fit(data_chunk[self.X_names], data_chunk[y_name])
            self.betas.loc[sample, :] = pd.Series(reg.coef_, index=self.X_names)
            self.p_values.loc[sample, :] = pd.Series(reg.p, index=self.X_names)
        
        self.__SetSignificantBetas()
        self.fit_done = True"""

        self.betas = pd.DataFrame(columns=self.X_names)
        self.p_values = pd.DataFrame(columns=self.X_names)

        for sample, data_chunk in data.groupby(self.s_name):
            self.betas.loc[sample, :], self.p_values.loc[sample, :] = self.__FitSubSample(data_chunk, y_name)
        
        self.__SetSignificantBetas()
        self.fit_done = True

    def __FitTotal(self, data, y_name): 
        ### V2
        self.betas, self.p_values = self.__FitSubSample(data, y_name)
        
        self.__SetSignificantBetas()
        self.fit_done = True
        
        ### V1
        """self.betas = pd.Series(0,index=self.X_names)
        self.p_values = pd.Series(0, index=self.X_names)

        X_names_for_reg = list(self.X_names)
        reg = LR(fit_intercept=False).fit(data[X_names_for_reg], data[y_name])
        if self.second_run:
            X_names_for_reg = [name for i, name in enumerate(X_names_for_reg) if reg.p[i] < self.p_margin]
            reg = LR(fit_intercept=False).fit(data[X_names_for_reg], data[y_name])

        for i, name in enumerate(X_names_for_reg): 
            self.betas[name] = reg.coef_[i]
            self.p_values[name] = reg.p[i]
        
        self.__SetSignificantBetas()
        self.fit_done = True"""

        ### V0 
        """ reg = LR(fit_intercept=False).fit(data[self.X_names], data[y_name])
        self.betas = pd.Series(reg.coef_, index=self.X_names)
        self.p_values = pd.Series(reg.p, index=self.X_names)

        self.__SetSignificantBetas()
        self.fit_done = True

        if self.second_run:
            X_names_second_run = [name for name in self.X_names if self.p_values[name] < self.p_margin]
            reg_second = LR(fit_intercept=False).fit(data[X_names_second_run], data[y_name])
            self.betas[:] = 0 
            self.p_values[:] = 0
            for i, name in enumerate(X_names_second_run): 
                self.betas[name] = reg_second.coef_[i]
                self.betas_significant[name] = reg_second.coef_[i]
                self.p_values[name] = reg_second.p[i]"""



    def __SetSignificantBetas(self): 
        if self.second_run: 
            self.betas_significant = self.betas.copy()
        else: 
            self.betas_significant = self.betas.where(self.p_values < self.p_margin, 0)

    def Contributions(self, data: pd.DataFrame) -> pd.DataFrame: 
        assert self.fit_done, "Еще не обучена"
        if self.s_name:
            return self.__ContributionsSplit(data)
        else: 
            return self.__ContributionsTotal(data)

    def __ContributionsSplit(self, data): 
        assert all([x in data.columns for x in self.X_names]), "SubSampleRegression: в данных нет переменных X_name"
        assert self.s_name in data.columns, "SubSampleRegression: в данных нет переменных s_name"
        return data[self.X_names].set_index([data.index, data[self.s_name]]).mul(self.betas_significant, level=1).set_index(data.index)
    
    def __ContributionsTotal(self, data): 
        assert all([x in data.columns for x in self.X_names]), "SubSampleRegression: в данных нет переменных X_name"
        return data[self.X_names].mul(self.betas_significant)
    
    def Predict(self, data: pd.DataFrame) -> pd.DataFrame: 
        return self.Contributions(data).sum(axis=1)
    
    def Score(self, data: pd.DataFrame, y_name: str) -> float: 
        #y_predicted = self.Predict(data)
        #y_actual = data[y_name] 
        #return 1 - ((y_actual - y_predicted) ** 2).sum() / ((y_actual - y_actual.mean()) ** 2).sum()
        return r2_score(data[y_name], self.Predict(data))
    
    def GetBetas(self) -> pd.DataFrame:
        assert self.fit_done, "Еще не обучена"
        if isinstance(self.betas, pd.DataFrame): 
            return self.betas
        else:
            return pd.DataFrame([self.betas])
        
    def GetPValues(self) -> pd.DataFrame:
        assert self.fit_done, "Еще не обучена"
        if isinstance(self.p_values, pd.DataFrame): 
            return self.p_values
        else:
            return pd.DataFrame([self.p_values])
        
    def GetBetasAndPValues(self) -> pd.DataFrame:
        assert self.fit_done, "Еще не обучена"
        bts, pvs = self.GetBetas(), self.GetPValues()
        result = pd.concat([bts, pvs], axis=1)
        result.columns = [['Beta'] * len(bts.columns) + ['P-value'] * len(pvs.columns), result.columns]
        return result