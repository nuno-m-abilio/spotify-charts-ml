import numpy as np

class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, num_iterations=1000, random_state=42):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        # np.clip evita overflow do exp()
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y, y_hat):
        m = len(y)
        # Epsilon para evitar log(0)
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        loss = - (1/m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Inicialização dos parâmetros
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []

        # Gradiente Descendente
        for i in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(linear_model)

            # Cálculo dos gradientes
            dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
            db = (1 / n_samples) * np.sum(y_hat - y)

            # Atualização dos pesos
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Registro da perda a cada 100 épocas
            if i % 100 == 0:
                loss = self._compute_loss(y, y_hat)
                self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        y_hat = self.predict_proba(X)
        return (y_hat >= threshold).astype(int)