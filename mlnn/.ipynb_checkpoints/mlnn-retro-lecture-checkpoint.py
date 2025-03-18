import numpy as np

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Datos de entrada y objetivos
x = np.array([1, 4, 5])  # Patrones de entrada: x_1, x_2, x_3
t = np.array([0.05, 0.01])  # Targets: t_1, t_2

# Inicialización de pesos y biases
np.random.seed(42)  # Para reproducibilidad
w_h = np.random.rand(3, 2)  # Pesos de la capa de entrada a la capa oculta (3x2)
b_h = np.random.rand(2)  # Bias de la capa oculta (2)
w_o = np.random.rand(2, 2)  # Pesos de la capa oculta a la capa de salida (2x2)
b_o = np.random.rand(2)  # Bias de la capa de salida (2)

# Imprimir valores iniciales de weights y biases
print("Valores iniciales de los weights y biases:")
print(f"Pesos capa oculta (w_h):\n{w_h}")
print(f"Bias capa oculta (b_h): {b_h}")
print(f"Pesos capa salida (w_o):\n{w_o}")
print(f"Bias capa salida (b_o): {b_o}\n")

# Hiperparámetros
learning_rate = 0.1
epochs = 10000
accuracy_threshold = 0.90  # Umbral de accuracy para detener el entrenamiento

# Entrenamiento de la red neuronal
for epoch in range(epochs):
    # Propagación hacia adelante
    z_h = np.dot(x, w_h) + b_h  # Suma ponderada en la capa oculta
    h = sigmoid(z_h)  # Salida de la capa oculta

    z_o = np.dot(h, w_o) + b_o  # Suma ponderada en la capa de salida
    o = sigmoid(z_o)  # Salida de la capa de salida

    # Cálculo del error
    error = t - o
    mse = np.mean(np.square(error))  # Error cuadrático medio

    # Cálculo del accuracy
    accuracy = 1 - np.mean(np.abs(error) / np.abs(t))  # Accuracy basado en el error relativo

    # Retropropagación del error
    d_o = error * sigmoid_derivative(o)  # Gradiente de la capa de salida
    error_h = np.dot(d_o, w_o.T)  # Propagación del error a la capa oculta
    d_h = error_h * sigmoid_derivative(h)  # Gradiente de la capa oculta

    # Actualización de pesos y biases
    w_o += learning_rate * np.outer(h, d_o)  # Actualización de pesos de salida
    b_o += learning_rate * d_o  # Actualización de bias de salida
    w_h += learning_rate * np.outer(x, d_h)  # Actualización de pesos ocultos
    b_h += learning_rate * d_h  # Actualización de bias ocultos

    # Imprimir el error y el accuracy cada 1000 épocas
    if epoch % 1000 == 0:
        print(f"Época {epoch}, Error: {mse:.6f}, Accuracy: {accuracy * 100:.2f}%")

    # Detener el entrenamiento si se alcanza el accuracy deseado
    if accuracy >= accuracy_threshold:
        print(f"\nEntrenamiento detenido en la época {epoch}. Accuracy alcanzado: {accuracy * 100:.2f}%")
        break

# Imprimir valores finales de weights y biases
print("\nValores finales de los weights y biases:")
print(f"Pesos capa oculta (w_h):\n{w_h}")
print(f"Bias capa oculta (b_h): {b_h}")
print(f"Pesos capa salida (w_o):\n{w_o}")
print(f"Bias capa salida (b_o): {b_o}\n")

# Explicación del cálculo del error y accuracy
print("Cálculo del error y accuracy:")
print(f"Targets (t): {t}")
print(f"Salida de la red (o): {o}")
print(f"Error (t - o): {error}")
print(f"Error cuadrático medio (MSE): {mse:.6f}")
print(f"Accuracy: {accuracy * 100:.2f}% (Deseado: {accuracy_threshold * 100}%)\n")

# Función para probar que los patrones han sido aprendidos
def probar_patron(patron):
    z_h = np.dot(patron, w_h) + b_h  # Suma ponderada en la capa oculta
    h = sigmoid(z_h)  # Salida de la capa oculta
    z_o = np.dot(h, w_o) + b_o  # Suma ponderada en la capa de salida
    o = sigmoid(z_o)  # Salida de la capa de salida
    return o

# Probar el patrón de entrada
print("Probando el patrón de entrada:")
patron_prueba = np.array([1, 4, 5])  # Mismo patrón de entrenamiento
salida_prueba = probar_patron(patron_prueba)
print(f"Entrada: {patron_prueba}, Salida: {salida_prueba}")
