# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class GSLSSVM(BaseEstimator, RegressorMixin):
    """
    Attributes:
        - gamma      : the hyper-parameter (float)
        - kernel     : the kernel used     (string)
        - kernel_    : the actual kernel function
        - x          : the data on which the LSSVM is trained (call it support vectors)
        - y          : the targets for the training data
        - coef_      : coefficents of the support vectors
        - intercept_ : intercept term
    """
    def __init__(self, gamma: float = 1.0, kernel: str = None, sigma: float = 1.0, threshold: float = 1, max_size: int = None):
        self.gamma = gamma
        self.c = 1000 # only for poly: not supported yet
        self.d = 20 # only for poly: not supported yet
        self.sigma = sigma
        self.threshold = threshold  # порог для остановки алгоритма
        self.max_size = max_size  # максимальный размер разреженного ядра
        
        if kernel is None:
            self.kernel = 'rbf'
        else:
            self.kernel = kernel

        params = dict()
        if kernel == 'poly':
            params['c'] = self.c # only for poly: not supported yet
            params['d'] = self.d # only for poly: not supported yet
        elif kernel == 'rbf':
            params['sigma'] = sigma

        self.kernel_ = GSLSSVM.__set_kernel(self.kernel, **params)

        # model parameters
        self.x = None
        self.y = None
        self.coef_ = None
        self.intercept_ = None
        self.support_vectors_ = []  # разреженное множество


    @staticmethod
    def __set_kernel(name: str, **params):
        """
        Выбор ядровой функции
        """
        def linear(xi, xj):
            return np.dot(xi, xj.T)

        def poly(xi, xj, c=params.get('c', 1.0), d=params.get('d', 2)):
            return ((np.dot(xi, xj.T))/c  + 1)**d

        def rbf(xi, xj, sigma=params.get('sigma', 1.0)):

            from scipy.spatial.distance import cdist
            if (xi.ndim == 2 and xi.ndim == xj.ndim): # both are 2D matrices
                return np.exp(-(cdist(xi, xj, metric='sqeuclidean'))/(2*(sigma**2)))
            elif ((xi.ndim < 2) and (xj.ndim < 3)):
                ax = len(xj.shape)-1 #compensate for python zero-base
                return np.exp(-(np.dot(xi, xi) + (xj**2).sum(axis=ax)
                                - 2*np.dot(xi, xj.T))/(2*(sigma**2)))
            else:
                message = "The rbf kernel is not suited for arrays with rank >2"
                raise Exception(message)

        kernels = {'linear': linear, 'poly': poly, 'rbf': rbf}
        if kernels.get(name) is not None:
            return kernels[name]
        else: #unknown kernel: crash and burn?
            message = "Kernel "+name+" is not implemented. Please choose from : "
            message += str(list(kernels.keys())).strip('[]')
            raise KeyError(message)

    def __UpdateSupportSet(self):
        """
        Итеративно добавляет элементы в разреженное множество, используя критерий снижения ошибки.
        """
        errors = []
        for i in range(len(self.x)):
            if i not in self.support_vectors_:
                S_new = self.support_vectors_ + [i]
                Omega_new = self.kernel_(self.x[S_new], self.x[S_new])
                Ones_new = np.ones((len(S_new), 1))
                H_new = np.block([
                    [Omega_new + self.gamma ** -1 * np.identity(len(S_new)), Ones_new],
                    [Ones_new.T, np.array([[0]])]
                ])
                c_new = np.concatenate((self.y[S_new], np.array([np.sum(self.y) / len(self.y)])))
                solution_new = np.linalg.solve(H_new, c_new)
                coef_new = solution_new[:-1]
                intercept_new = solution_new[-1]
                
                residuals = self.y - (self.kernel_(self.x, self.x[S_new]) @ coef_new + intercept_new)
                error = np.mean(residuals ** 2)
                errors.append((error, i))

        if (not errors):
            return False

        errors.sort()
        best_error, best_index = errors[0]
        
        if best_error < self.threshold and (self.max_size is None or len(self.support_vectors_) < self.max_size):
            self.support_vectors_.append(best_index)
            return True

        return False

    def __OptimizeParams(self):
        """
        Жадное обновление разреженного ядра и оптимизация коэффициентов.
        """
        if not self.support_vectors_:
            self.support_vectors_.append(0)
        
        S = self.support_vectors_
        Omega = self.kernel_(self.x[S], self.x[S])
        Ones = np.ones((len(S), 1))

        H = np.block([
            [Omega + self.gamma ** -1 * np.identity(len(S)), Ones],
            [Ones.T, np.array([[0]])]
        ])
        c = np.concatenate((self.y[S], np.array([np.sum(self.y) / len(self.y)])))

        solution = np.linalg.solve(H, c)
        self.coef_ = solution[:-1]
        self.intercept_ = solution[-1]


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучение модели и настройка гиперпараметров.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            Xloc = X.to_numpy()
        else:
            Xloc = X

        if isinstance(y, (pd.DataFrame, pd.Series)):
            yloc = y.to_numpy()
        else:
            yloc = y

        if (Xloc.ndim == 2) and (yloc.ndim == 1):
            self.x = Xloc
            self.y = yloc

            while self.__UpdateSupportSet():
                self.__OptimizeParams()
        else:
            message = "The fit procedure requires a 2D numpy array of features "\
                "and 1D array of targets"
            raise Exception(message)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает значения на основе обученной модели.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model is not trained yet. Call fit() before predict().")
        
        K = self.kernel_(X, self.x[self.support_vectors_])
        return K @ self.coef_ + self.intercept_
