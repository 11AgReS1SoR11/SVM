import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from LSSVM import GSLSSVM


def sinc(X):
    return np.sinc(X).ravel()


def generate_data(n_samples, low, high, noise_std, func = sinc, seed=3):
    """
    Генерирует синтетические данные для регрессии (defult = sinc(x)).
    
    Аргументы:
    n_samples (int)   - Количество точек.
    low (float)       - Нижняя граница.
    high (float)      - Верхняя граница.
    noise_std (float) - Стандартное отклонение шума.
    func              - Истинная функция
    seed (int)        - Параметр для случайных чисел

    Возвращает:
    x (ndarray) - Массив входных значений.
    y (ndarray) - Соответствующие целевые значения.
    """
    np.random.seed(seed)
    x = np.random.uniform(low=low, high=high, size=(n_samples, 1))
    y = func(x) + np.random.normal(0, noise_std, size=n_samples)
    return x, y


def train_lssvm(x, y, gamma=100, sigma=1.0, max_size = 100, threshold = 1):
    """
    Обучает LSSVM-модель на данных.

    Аргументы:
    x (ndarray)   - Входные данные.
    y (ndarray)   - Целевые значения.
    gamma (float) - Параметр регуляризации.
    sigma (float) - Параметр RBF-ядра.

    Возвращает:
    model - Обученная модель LSSVM.
    """
    model = GSLSSVM(
            gamma=gamma,
            kernel='rbf',
            sigma=sigma,
            threshold=threshold,
            max_size=max_size)
    model.fit(x, y)
    return model


def plot_results(x_train, y_train, x_pred, y_exact, y_pred):
    """
    Визуализирует результаты предсказаний LSSVM.

    Аргументы:
    x_train (ndarray) - Исходные точки выборки.
    y_train (ndarray) - Истинные значения для этих точек.
    x_pred (ndarray)  - Массив предсказанных точек.
    y_exact (ndarray) - Истинные значения для предсказанных точек.
    y_pred (ndarray)  - Предсказанные моделью значения.
    """
    plt.scatter(x_train, y_train, color='black', label="Training Data")
    plt.plot(x_pred, y_exact, color='black', linestyle='dashed', label="True Function")
    plt.plot(x_pred, y_pred, color='red', label="LSSVM Prediction")
    
    plt.xlabel("Feature x")
    plt.ylabel("Target y")
    plt.legend()
    plt.grid()
    plt.show()


def evaluate_model(y_true, y_pred):
    """
    Вычисляет метрики качества модели.

    Аргументы:
    y_true (ndarray) - Истинные значения.
    y_pred (ndarray) - Предсказанные значения.

    Возвращает:
    mse (float)  - Среднеквадратическая ошибка.
    r2 (float)   - Коэффициент детерминации.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2


def test_support_vectors_impact(x_pred, a, b, num_vectors, noise_std, gamma, sigma, func = sinc):
    """
    Исследует влияние количества опорных векторов на среднеквадратическую ошибку.
    """
    
    y_exact = func(x_pred)
    errors = []

    for n in num_vectors:
        x, y = generate_data(n, a, b, noise_std, func)
        model = train_lssvm(x, y, gamma=gamma, sigma=sigma, max_size = n, threshold = 10000)

        y_pred = model.predict(x_pred)

        mse, _ = evaluate_model(y_exact, y_pred)
        errors.append(mse)

    plt.plot(num_vectors, errors, marker='o')
    plt.xlabel("Количество опорных векторов")
    plt.ylabel("Среднеквадратическая ошибка")
    plt.title("Зависимость среднеквадратической ошибки от количества опорных векторов")
    plt.grid()
    plt.show()


def e2e(n_samples, a, b, noise_std, gamma, sigma, x_pred, func = sinc):
    x_train, y_train = generate_data(n_samples, a, b, noise_std, func)
    model = train_lssvm(x_train, y_train, max_size = n_samples)

    y_exact = func(x_pred) + np.random.normal(0, noise_std, size=len(x_pred))
    y_pred = model.predict(x_pred)

    mse, r2 = evaluate_model(y_exact, y_pred)
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'R2 Score: {r2:.4f}')

    plot_results(x_train, y_train, x_pred, y_exact, y_pred)

    num_vectors = np.linspace(5, 50, 46, dtype=int)  # Количество опорных векторов
    test_support_vectors_impact(x_pred, a, b, num_vectors, noise_std, gamma, sigma, func)


if __name__ == "__main__":
    ########### Без шума
    """
    set hyperparams
    """
    n_samples_train = 10
    noise_std = 0.0
    gamma = 100
    sigma = 1.0

    """
    set sample
    """
    A = -5
    B = 5
    x_pred = np.linspace(A, B, 200).reshape(-1, 1)

    e2e(n_samples_train, A, B, noise_std, gamma, sigma, x_pred, sinc)


    ########### С шумом
    """
    set hyperparams
    """
    n_samples_train = 10
    noise_std = 0.1
    gamma = 100
    sigma = 1.0

    """
    set sample
    """
    A = -5
    B = 5
    x_pred = np.linspace(A, B, 200).reshape(-1, 1)

    e2e(n_samples_train, A, B, noise_std, gamma, sigma, x_pred, sinc)
