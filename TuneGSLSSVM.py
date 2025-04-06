import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from GSLSSVM import GSLSSVM 
import matplotlib.colors as mcolors


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


def tune_gslssvm_grid_search(X, Y, noise):
    """
    Подбирает гиперпараметры gamma и sigma для GSLSSVM с помощью Grid Search.

    Args:
        X: Все признаки (для финального обучения).
        Y: Все целевые переменные (для финального обучения).

    Returns:
        best_gamma: Лучшее значение gamma.
        best_sigma: Лучшее значение sigma.
        best_model: Лучшая обученная модель.
    """

    sigmas = np.arange(-1, 1.1, 0.1)
    gammas = [10**i for i in range(2, 10)]

    best_gamma = None
    best_sigma = None
    best_mse = float('inf')
    best_model = None

    mse_matrix = np.zeros((len(gammas), len(sigmas))) # for graphics

    n_opor_vectors = 10
    threshold = 1e-10

    for i, gamma in enumerate(gammas):
        ####
        print(f"{((i / len(gammas)) * 100):.2f}%")
        ####
        for j, sigma in enumerate(sigmas):
            try:
                model = train_lssvm(X, Y, gamma, sigma, max_size = n_opor_vectors, threshold = threshold)

                y_pred = model.predict(X)
                mse, _ = evaluate_model(Y, y_pred)

                mse_matrix[i, j] = mse

                if mse < best_mse:
                    best_mse = mse
                    best_gamma = gamma
                    best_sigma = sigma
                    best_model = model
            except Exception as e:
                print(f"Ошибка при Gamma: {gamma}, Sigma: {sigma}: {e}")
                mse_matrix[i, j] = float('inf')

    print(f"Best Gamma: {best_gamma}, Best Sigma: {best_sigma}, Best MSE: {best_mse}")

    plot_hyperparameter_tuning_results(gammas, sigmas, mse_matrix, noise)

    return best_gamma, best_sigma, best_model


def plot_hyperparameter_tuning_results(gammas, sigmas, mse_matrix, noise):
    """
    Строит графики зависимости MSE от gamma и sigma.
    """

    # Heatmap
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(sigmas, gammas)
    im = plt.pcolormesh(X, Y, mse_matrix, shading='gouraud', norm=mcolors.LogNorm(vmin=mse_matrix.min(), vmax=mse_matrix.max()))

    plt.colorbar(label='MSE', extend='both')
    plt.xlabel('Sigma')
    plt.ylabel('Gamma')
    plt.title('MSE Heatmap (Grid Search)')
    plt.yscale('log')
    plt.xticks(sigmas)
    plt.yticks(gammas)
    if noise > 0:
        plt.savefig('Heatmap_noise.png')
    else:
        plt.savefig('Heatmap.png')

    # График зависимости MSE от gamma (при фиксированном sigma)
    # Возьмем sigma, дающее наименьшее MSE
    best_sigma_index = np.argmin(np.mean(mse_matrix, axis=0)) # Индекс лучшего sigma
    plt.figure(figsize=(10, 6))
    plt.plot(gammas, mse_matrix[:, best_sigma_index], marker='o')
    plt.xlabel('Gamma')
    plt.ylabel('MSE')
    plt.title(f'MSE vs Gamma (Sigma = {sigmas[best_sigma_index]:.2f})')
    plt.xticks(gammas)
    plt.grid(True)
    plt.xscale('log')  # Добавляем логарифмический масштаб по оси x
    if noise > 0:
        plt.savefig('MSE_vs_Gamma_noise.png')
    else:
        plt.savefig('MSE_vs_Gamma.png')

    # График зависимости MSE от sigma (при фиксированном gamma)
    # Возьмем gamma, дающее наименьшее MSE
    best_gamma_index = np.argmin(np.mean(mse_matrix, axis=1)) # Индекс лучшего gamma
    plt.figure(figsize=(10, 6))
    plt.plot(sigmas, mse_matrix[best_gamma_index, :], marker='o')
    plt.xlabel('Sigma')
    plt.ylabel('MSE')
    plt.title(f'MSE vs Sigma (Gamma = {gammas[best_gamma_index]:2.2f})')
    plt.xticks(sigmas)
    plt.grid(True)
    if noise > 0:
        plt.savefig('MSE_vs_Sigma_noise.png')
    else:
        plt.savefig('MSE_vs_Sigma.png')


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


if __name__ == '__main__':

    """
    set hyperparams
    """
    n_samples_train = 200
    """
    set sample
    """
    noise_std = 0.0
    A = -5
    B = 5
    X, Y = generate_data(n_samples_train, A, B, noise_std)

    best_gamma, best_sigma, best_model = tune_gslssvm_grid_search(X, Y, noise_std)

    print("Tune without noise")
    print(f"Best Gamma (Final): {best_gamma}, Best Sigma (Final): {best_sigma}")

    """
    set sample
    """
    noise_std = 0.1
    X, Y = generate_data(n_samples_train, A, B, noise_std)

    best_gamma, best_sigma, best_model = tune_gslssvm_grid_search(X, Y, noise_std)

    print("Tune with noise")
    print(f"Best Gamma (Final): {best_gamma}, Best Sigma (Final): {best_sigma}")
