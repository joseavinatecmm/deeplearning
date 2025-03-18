import numpy as np
import matplotlib.pyplot as plt

# Datos de la compuerta lógica AND
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([-1, -1, -1, 1])  # Etiquetas: -1 para False, 1 para True

# Inicialización de pesos y sesgo
w = np.array([0.0, 0.0])  # Vector de pesos
b = 0.0                   # Sesgo
eta = 0.1                 # Tasa de aprendizaje
epochs = 10               # Número de épocas

# Función para graficar el hiperplano y los datos
def plot_decision_boundary(X, y, w, b, epoch):
    plt.figure(figsize=(8, 6))
    
    # Graficar los puntos de datos
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='False (0)', marker='o')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='True (1)', marker='x')
    
    # Graficar el hiperplano (línea de decisión)
    x1 = np.linspace(-0.5, 1.5, 100)
    x2 = -(w[0] * x1 + b) / w[1]  # Despejamos x2 de w1*x1 + w2*x2 + b = 0
    plt.plot(x1, x2, color='green', label='Hiperplano')
    
    # Graficar el vector de pesos (ortogonal al hiperplano)
    origin = np.array([0.5, 0.5])  # Punto de origen para el vector
    plt.quiver(*origin, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='purple', label='Vector de pesos')
    
    # Configuración del gráfico
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.title(f'Perceptrón - Época {epoch}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

# Entrenamiento del perceptrón
for epoch in range(epochs):
    print(f"Época {epoch + 1}")
    for i in range(len(X)):
        # Calcular la salida del perceptrón
        z = np.dot(w, X[i]) + b
        y_pred = 1 if z >= 0 else -1
        
        # Actualizar pesos y sesgo si hay error
        if y_pred != y[i]:
            w += eta * y[i] * X[i]
            b += eta * y[i]
        
        # Graficar en cada paso
        plot_decision_boundary(X, y, w, b, epoch + 1)
    
    print(f"Pesos: {w}, Sesgo: {b}\n")

# Resultado final
print("Entrenamiento completado.")
print(f"Pesos finales: {w}, Sesgo final: {b}")
