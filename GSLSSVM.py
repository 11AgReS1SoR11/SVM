# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
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

    def get_op_vectors(self):
        return self.x[self.support_vectors_], self.y[self.support_vectors_]

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
            """
            RBF kernel, handles different input shapes.
            """
            if xi.ndim == 2 and xj.ndim == 2:  # both are 2D matrices
                return np.exp(-(cdist(xi, xj, metric='sqeuclidean')) / (2 * (sigma**2)))
            else:
                # Broadcasting approach for other cases (e.g., xi is vector, xj is matrix)
                return np.exp(-np.sum((xi - xj)**2, axis=-1) / (2 * (sigma**2)))

        kernels = {'linear': linear, 'poly': poly, 'rbf': rbf}
        if kernels.get(name) is not None:
            return kernels[name]
        else: #unknown kernel: crash and burn?
            message = "Kernel "+name+" is not implemented. Please choose from : "
            message += str(list(kernels.keys())).strip('[]')
            raise KeyError(message)


    def __solve(self, S_new, K_full, sum_K):
        L = len(self.x)  # Число всех точек
        len_S = len(S_new)  # Новая размерность S

        # Вычисляем подматрицу K и Omega
        # Векторизованное вычисление Omega_new
        x_S_new = self.x[S_new]  # Выбираем опорные векторы
        K_SS = self.kernel_(x_S_new[:, None], x_S_new[None, :]) # K(S_new, S_new) - матрица ядровых функций между опорными векторами

        # Вычисляем K(x, S_new) - матрица ядровых функций между всеми точками и опорными векторами
        K_XS = self.kernel_(self.x[:, None], x_S_new[None, :])

        # Вычисляем Omega_new с использованием матричного умножения
        Omega_new = (L / (2 * self.gamma)) * K_SS + K_XS.T @ K_XS

        # Вектор Phi
        Phi_new = sum_K[S_new]

        # Блочная матрица H
        H_new = np.block([
            [Omega_new, Phi_new.reshape(-1, 1)],
            [Phi_new.reshape(1, -1), np.array([[L]])]
        ])

        # Вектор правой части c
        c_new = np.concatenate((K_full[S_new] @ self.y, [np.sum(self.y)]))

        # Решаем систему H * [β; b] = c
        solution_new = np.linalg.solve(H_new, c_new)
        return solution_new

    def __UpdateSupportSet(self):
        """
        Итеративно добавляет элементы в разреженное множество, используя критерий снижения ошибки.
        Оптимизированная версия с векторизацией.
        """
        errors = []
        L = len(self.x)  # Число всех точек
        current_S = np.array(self.support_vectors_, dtype=int)  # Текущее множество опорных векторов

        # Вычисляем K заранее
        K_full = self.kernel_(self.x, self.x)  # L x L матрица ядра
        sum_K = np.sum(K_full, axis=1)         # L x 1 (суммы по столбцам)

        # Кандидаты для добавления (те, кто не в current_S)
        candidates = [i for i in range(L) if i not in current_S]

        for i in candidates:
            S_new = np.append(current_S, [i])     # Добавляем кандидата
            solution_new = self.__solve(S_new, K_full, sum_K)
            coef_new = solution_new[:-1]          # Вектор коэффициентов β
            intercept_new = solution_new[-1]      # Свободный член b

            # Ошибка предсказания
            predictions = K_full[:, S_new] @ coef_new + intercept_new
            residuals = self.y - predictions
            error = np.mean(residuals ** 2)
            
            errors.append((error, i, solution_new))

        if not errors:
            raise Exception("No valid candidates to add.")

        # Добавляем элемент с минимальной ошибкой
        errors.sort()
        return errors[0]

    def __PruneSupportSet(self):
        """
        Итеративное прореживание опорных векторов методом уменьшения количества точек
        на 5% за итерацию, пересчитывая параметры LS-SVM на каждом шаге.
        """
        prune_fraction = 0.05  # Доля удаляемых точек за итерацию
        remaining_vectors = list(range(len(self.x)))  # Начинаем со всех точек

        while len(remaining_vectors) > self.max_size:
            # Формируем подмножество точек S
            S = remaining_vectors
            l = len(S)

            # Строим матрицу ядра K
            K = self.kernel_(self.x[S], self.x[S])
            
            # Строим матрицу Omega
            Omega = K + (l / self.gamma) * np.identity(l)

            # Вектор единиц
            Ones = np.ones((l, 1))

            # Формируем матрицу H и вектор правой части
            H = np.block([
                [Omega, Ones],
                [Ones.T, np.array([[0]])]
            ])
            rhs = np.concatenate((self.y[S], np.array([0])))

            # Решаем систему линейных уравнений
            solution = np.linalg.solve(H, rhs)

            alpha = solution[:-1]  # Коэффициенты α
            b = solution[-1]       # Смещение b

            # Определяем, какие вектора удалить (по наименьшим |α_i|)
            num_to_remove = max(1, int(prune_fraction * len(S)))        # Число удаляемых точек
            remove_indices = np.argsort(np.abs(alpha))[:num_to_remove]  # Индексы наименьших α

            # Удаляем выбранные индексы
            remaining_vectors = [S[i] for i in range(len(S)) if i not in remove_indices]

        # Итоговое множество поддерживающих векторов
        print(f"len(remaining_vectors) = {len(remaining_vectors)}")
        self.support_vectors_ = remaining_vectors
        return self.support_vectors_

    def __OptimizeParams(self, isPrune):
        """
        Жадное обновление разреженного ядра и оптимизация коэффициентов.
        """
        if (isPrune):
            self.__PruneSupportSet()
            S = self.support_vectors_
            l = len(S)
            # Строим матрицу ядра K
            K = self.kernel_(self.x[S], self.x[S])
            # Строим матрицу Omega
            Omega = K + (l / self.gamma) * np.identity(l)
            # Вектор единиц
            Ones = np.ones((l, 1))
            # Формируем матрицу H и вектор правой части
            H = np.block([
                [Omega, Ones],
                [Ones.T, np.array([[0]])]
            ])
            rhs = np.concatenate((self.y[S], np.array([0])))
            # Решаем систему линейных уравнений
            solution = np.linalg.solve(H, rhs)
            self.coef_ = solution[:-1]     # Коэффициенты α
            self.intercept_ = solution[-1] # Смещение b

        else:
            while (self.max_size is None or len(self.support_vectors_) < self.max_size):
                best_error, best_index, solution = self.__UpdateSupportSet()
                self.support_vectors_.append(best_index)
                self.coef_ = solution[:-1]      # Вектор коэффициентов β
                self.intercept_ = solution[-1]  # Свободный член b

                if (best_error < self.threshold):
                    return


        if not self.support_vectors_:
            message = "No support vectors"
            raise Exception(message)


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
            
            self.__OptimizeParams(False) # bool parametr isPrune
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
