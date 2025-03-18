# Entrenamiento de un Perceptrón para la Compuerta AND

Un **perceptrón** es un modelo de neurona artificial que se usa para clasificar datos linealmente separables. En este caso, entrenaremos un perceptrón para aprender la compuerta lógica **AND**.

## 1. Definición de la Compuerta AND
La compuerta lógica AND tiene la siguiente tabla de verdad:

| Entrada x1 | Entrada x2 | Salida AND |
|------------|------------|-------------|
| 0          | 0          | 0           |
| 0          | 1          | 0           |
| 1          | 0          | 0           |
| 1          | 1          | 1           |

## 2. Modelo del Perceptrón

$$\hat{y}=g\left(\sum_{i=1}  w_i x_i\right)$$

$$\hat{y}=g\left(w_0+\sum_{i=1}^m x_i w_i\right)$$

$$
 y = f(w_1 x_1 + w_2 x_2 + b)
$$

donde:
- $( x_1, x_2 )$ son las entradas.
- $( w_1, w_2 )$ son los pesos.
- \( b \) es el sesgo.
- \( f(z) ó g(z) \) es la función de activación (usamos la función escalón de Heaviside):

$$
g(z) = f(z) = 
\begin{cases}
1, & \text{si } z \geq 0 \\
0, & \text{si } z < 0
\end{cases}
$$

## 3. Algoritmo de Entrenamiento
Se usa la **regla de actualización de pesos**:

$$w_i^{(t+1)} = w_i^{(t)} + \eta (y - \hat{y}) x_i$$

$$
b^{(t+1)} = b^{(t)} + \eta (y - \hat{y})
$$

donde:
- $( \eta )$ es la tasa de aprendizaje (usaremos 0.1).
- $( y )$ es la salida esperada.
- $( \hat{y} )$ es la salida predicha por el perceptrón.

### Inicialización de Parámetros

- La inicialización es arbitraria

Supongamos:
- $w_1 = 0$
- $w_2 = 0$, 
- $b = 0$
- $\eta = 0.1$

### Iteración Paso a Paso
#### **Época 1**
Para cada muestra, calculamos la salida $( \hat{y} )$ y actualizamos los pesos si es necesario.

1. **Entrada (0,0) → Salida esperada: 0**
   - $( z = (0 \times 0) + (0 \times 0) + 0 = 0 )$
   - $( \hat{y} = f(0) = 1 )$ → Incorrecto

   - Actualización:
     - $( w_1 = 0 + 0.1 (0 - 1) (0) = 0 )$
     - $( w_2 = 0 + 0.1 (0 - 1) (0) = 0 )$
     - $( b = 0 + 0.1 (0 - 1) = -0.1 )$

2. **Entrada (0,1) → Salida esperada: 0**
   - $( z = (0 \times 0) + (0 \times 1) - 0.1 = -0.1 )$
   - $( \hat{y} = f(-0.1) = 0 )$ → Correcto
   - No se actualizan pesos.

3. **Entrada (1,0) → Salida esperada: 0**
   - $( z = (0 \times 1) + (0 \times 0) - 0.1 = -0.1 )$
   - $( \hat{y} = f(-0.1) = 0 )$ → Correcto
   - No se actualizan pesos.

4. **Entrada (1,1) → Salida esperada: 1**
   - $( z = (0 \times 1) + (0 \times 1) - 0.1 = -0.1 )$
   - $( \hat{y} = f(-0.1) = 0 )$ → Incorrecto

   - Actualización:
     - $( w_1 = 0 + 0.1 (1 - 0) (1) = 0.1 )$
     - $( w_2 = 0 + 0.1 (1 - 0) (1) = 0.1 )$
     - $( b = -0.1 + 0.1 (1 - 0) = 0 )$

#### **Época 2**

Revisamos si el modelo ya clasifica correctamente.
- (0,0) → \( z = 0 \), \( $\hat{y}$ = 0 \) → Correcto.
- (0,1) → \( z = 0.1 \), \( $\hat{y}$ = 0 \) → Correcto.
- (1,0) → \( z = 0.1 \), \( $\hat{y}$ = 0 \) → Correcto.
- (1,1) → \( z = 0.2 \), \( $\hat{y}$ = 1 \) → Correcto.

Como todas las predicciones son correctas, el modelo ha aprendido la compuerta AND con:

$$w_1 = 0.1, \quad w_2 = 0.1, \quad b = 0$$

### 4. Expresión Final del Perceptrón AND
El perceptrón aprendido implementa la siguiente ecuación:

$$y = f(0.1x_1 + 0.1x_2)$$


Este modelo ahora clasifica correctamente todas las entradas de la compuerta AND. 🎯


