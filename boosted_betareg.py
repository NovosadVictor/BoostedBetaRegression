import numpy as np
import pandas as pd
from scipy.special import loggamma, digamma, expit, logit
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class BoostedBetaReg:
    def __init__(self, fit_phi=True, max_features=None):
        self.fit_phi = fit_phi
        self.max_features = max_features
        self.coefs = None
        self.phi = None
        self.fitted = False

        self.eps = 1e-6
        self.epochs = 0
        self.history = {}

    def fit(self, X, y, epochs, sample_weight=None):
        print(f'y.mean() = {y.mean()}; y.var() = {y.var()}')
        self.phi = np.mean(y) * (1 - np.mean(y)) / np.var(y) - 1

        self.coefs = pd.Series(0, index=list(X.columns) + ['(Intercept)'])
        self.boost(X, y, epochs=epochs, sample_weight=sample_weight)

    def boost(self, X, y, epochs, sample_weight=None):
        assert (epochs > 0)
        if epochs > self.epochs:
            if not self.fitted:
                # zero epoch - find maximum of likelihood with constant prediction
                print('Fitting constant')
                grid = np.linspace(0 + self.eps, 1 - self.eps, 1000)
                init = grid[np.argmax([self._loglikelihood(mu_, y) for mu_ in grid])]
                self.coefs.loc['(Intercept)'] = logit(init)
                print('Fitted constant: ', init)

            base_learner = LinearRegression(fit_intercept=False)
            features = X.columns
            for epoch in range(self.epochs, epochs):
                print('Epoch :', epoch)
                # find best predicting column for negative ll gradient
                bl_scores = pd.DataFrame(columns=['MSE', 'Estimate'])
                preds = self.predict(X)
                y_ = self._llgradient(preds, y)
                grid = np.linspace(self.phi / 2, 2*self.phi, int(15*self.phi))
                if self.fit_phi:
                    self.phi = grid[np.argmax([self.__loglikelihood(mu=preds, phi=phi, y=y) for phi in grid])]
                    print(f'Found best phi: {self.phi}')
                print('Mean abs ll gradient: ', y_.abs().mean())
                for col in features:
                    X_ = X[[col]]
                    base_learner.fit(X_, y_, sample_weight=sample_weight)
                    bl_scores.loc[col] = [
                        mean_squared_error(base_learner.predict(X_), y_),
                        base_learner.coef_[0],
                    ]
                best_col = bl_scores['MSE'].idxmin()
                best_col_estimate = bl_scores.loc[best_col]['Estimate']
                # find best gamma (step size)
                grid = np.linspace(0 + self.eps, 1 - self.eps, 100)
                gamma_scores = pd.Series()
                for gamma in grid:
                    coefs_ = self.coefs.copy()
                    coefs_.loc[best_col] = self.coefs.loc[best_col] + gamma * best_col_estimate
                    gamma_scores.loc[gamma] = self._loglikelihood(self.predict_(X, coefs_), y)
                best_gamma = gamma_scores.idxmax()
                print(f'Found best feature: {best_gamma}*{best_col_estimate}*{best_col}')
                score = pearsonr(preds, y)[0]
                print(f'Likelihood: {gamma_scores.loc[best_gamma]}, BIC: {self.bic_(X, y)}, cor: {score}')

                self.coefs.loc[best_col] = self.coefs.loc[best_col] + best_gamma * best_col_estimate
                non_zero_features = self.coefs[self.coefs != 0].index[:-1]
                print(f'Non zero features: {len(non_zero_features)}')
                if self.max_features and len(non_zero_features) == self.max_features:
                    features = list(non_zero_features)

                self.history[epoch] = {'coefs': self.coefs.copy(), 'score': score}
                self.epochs = epoch
                if epoch > 100 and abs(score - self.history[epoch - 100]['score']) < 0.01:
                    break

            self.fitted = True
        else:
            self.coefs = self.history[epochs]['coefs']

    @staticmethod
    def fit_baselearner(X, y, sample_weight=None):
        base_learner = LinearRegression(fit_intercept=False)
        base_learner.fit(X, y, sample_weight=sample_weight)
        return [
            mean_squared_error(base_learner.predict(X), y),
            base_learner.coef_[0],
        ]

    def predict(self, X):
        return self.predict_(X, self.coefs)

    @staticmethod
    def predict_(X, coefs):
        # X - pd.DataFrame
        assert (coefs is not None and not coefs.empty)
        cols = set(X.columns) & set(coefs.index)
        pred = X[cols].mul(coefs.loc[cols], axis=1).sum(axis=1)
        if '(Intercept)' in coefs.index:
            pred += coefs.loc['(Intercept)']

        return expit(pred)

    def score(self, X, y):
        pass

    def _llgradient(self, mu, y, sample_weight=None):
        # derivative with respect to logit(mu) == linear combination
        llg = mu * (1 - mu) * self.phi * (
                digamma((1 - mu) * self.phi + self.eps)
                - digamma(mu * self.phi + self.eps)
                + logit(y + self.eps)
        )

        return llg

    def _loglikelihood(self, mu, y, sample_weight=None):
        return self.__loglikelihood(mu, self.phi, y, eps=self.eps)

    @staticmethod
    def __loglikelihood(mu, phi, y, eps=1e-6, sample_weight=None):
        return np.sum(
            loggamma(phi + eps)
            - loggamma(mu * phi + eps)
            - loggamma((1 - mu) * phi + eps)
            + (mu * phi - 1) * np.log(y + eps)
            + ((1 - mu) * phi - 1) * np.log(1 - y + eps)
        )

    def bic_(self, X, y):
        # number of parameters
        k = (self.coefs != 0).sum()
        # number of samples
        n = len(X)
        ll = self._loglikelihood(self.predict(X), y)

        return k * np.log(n) - 2 * ll
