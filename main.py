import numpy as np
import matplotlib.pyplot as plt
from TuneGSLSSVM import evaluate_model, train_lssvm, generate_data, sinc


def plot_results(x_op, y_op, x_true, y_true, y_pred):
    """
    Визуализирует результаты предсказаний LSSVM.

    Аргументы:
    x_op (ndarray) - Опорные вектора x.
    y_op (ndarray) - Опорные вектора y.
    x_true (ndarray)  - Истинные значения x.
    y_true (ndarray)  - Истинные значения y.
    y_pred (ndarray)  - Предсказанные моделью значения.
    """

    plt.scatter(x_true, y_true, color='yellow', linestyle='dashed', label="True Function")
    plt.scatter(x_op, y_op, color='black', label="Oporny vectors")
    plt.scatter(x_true, y_pred, color='red', label="LSSVM Prediction")
    
    plt.xlabel("Feature x")
    plt.ylabel("Target y")
    plt.legend()
    plt.grid()
    plt.show()


def test_support_vectors_impact(x_pred, a, b, num_vectors, noise_std, gamma, sigma, func = sinc):
    """
    Исследует влияние количества опорных векторов на среднеквадратическую ошибку.
    """
    
    y_exact = func(x_pred)
    errors = []
    x, y = generate_data(200, a, b, noise_std, func)

    for n in num_vectors:
        model = train_lssvm(x, y, gamma=gamma, sigma=sigma, max_size = n, threshold = 1e-15) # there shouldn't use threshold

        y_pred = model.predict(x_pred)

        mse, _ = evaluate_model(y_exact, y_pred)
        errors.append(mse)

    plt.plot(num_vectors, errors, marker='o')
    plt.xlabel("Количество опорных векторов")
    plt.ylabel("Среднеквадратическая ошибка")
    plt.title("Зависимость среднеквадратической ошибки от количества опорных векторов")
    plt.grid()
    if noise_std > 0:
        plt.savefig('MSE_vs_n_vectors_noise.png')
    else:
        plt.savefig('MSE_vs_n_vectors.png')


def e2e(n_samples, n_opor_vectors, a, b, noise_std, gamma, sigma, threshold, func = sinc):
    x_train, y_train = generate_data(n_samples, a, b, noise_std, func)
    model = train_lssvm(x_train, y_train, gamma, sigma, max_size = n_opor_vectors, threshold = threshold)

    y_pred = model.predict(x_train)
    x_op, y_op = model.get_op_vectors()

    mse, r2 = evaluate_model(y_train, y_pred)
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'R2 Score: {r2:.4f}')

    plot_results(x_op, y_op, x_train, y_train, y_pred)

    # num_vectors = np.linspace(5, 15, 10, dtype=int)  # Количество опорных векторов
    # test_support_vectors_impact(x_train, a, b, num_vectors, noise_std, gamma, sigma, func)


if __name__ == "__main__":
    ########## Без шума
    """
    set hyperparams
    """
    n_samples_train = 200
    n_opor_vectors = 10
    threshold = 1e-6
    noise_std = 0.0
    gamma = 1500 #2**30
    sigma = 0.85 # 0.05

    """
    set sample
    """
    A = -5
    B = 5
    e2e(n_samples_train, n_opor_vectors, A, B, noise_std, gamma, sigma, threshold, sinc)


    ########### С шумом
    """
    set hyperparams
    """
    n_samples_train = 200
    n_opor_vectors = 10
    threshold = 1e-6
    noise_std = 0.1
    gamma = 1500 #2**30
    sigma = 0.85 # 0.05

    """
    set sample
    """
    A = -5
    B = 5

    e2e(n_samples_train, n_opor_vectors, A, B, noise_std, gamma, sigma, threshold, sinc)
