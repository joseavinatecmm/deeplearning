import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer

# Cargar el dataset
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

# Mostrar información del dataset
df.head()

# Seleccionar un subconjunto de características
selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
df_selected = df[selected_features + ['target']].copy()
df_selected['target'] = df_selected['target'].map({0: 'Benigno', 1: 'Maligno'})

# Pair plot
sns.pairplot(df_selected, hue='target', diag_kind='kde', markers=['o', 's'])
plt.show()

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df[selected_features].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Mapa de Calor de Correlación de Características')
plt.show()

# División del conjunto de datos
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalización de datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir la arquitectura del MLP
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Entrenar el modelo
mlp.fit(X_train, y_train)

# Realizar predicciones
y_pred = mlp.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.4f}')
print('\nReporte de clasificación:')
print(classification_report(y_test, y_pred))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.show()

# Función para predecir nuevos casos
def clasificar_nuevo_patron(caracteristicas):
    caracteristicas = np.array(caracteristicas).reshape(1, -1)
    caracteristicas = scaler.transform(caracteristicas)
    prediccion = mlp.predict(caracteristicas)
    return 'Maligno' if prediccion[0] == 1 else 'Benigno'

# Ejemplo de predicción con un nuevo patrón
nuevo_patron = X_test[0]  # Tomamos un ejemplo del conjunto de prueba
resultado = clasificar_nuevo_patron(nuevo_patron)
print(f'Clasificación del nuevo patrón: {resultado}')

