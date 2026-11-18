# Neural Networks from Scratch

Implementación progresiva y didáctica de redes neuronales artificiales construidas **desde cero** en Python, conectando los fundamentos matemáticos (cálculo multivariable, regla de la cadena y optimización por gradiente descendiente) con la programación pura, sin depender de frameworks de alto nivel como PyTorch o TensorFlow.

---

## 🚀 Evolución del Proyecto

### Versión 1.0 - Perceptrón Simple (`perceptron.py`)
Primera versión preliminar del proyecto, desarrollada de forma totalmente autónoma (sin asistencia externa) para comprender los fundamentos matemáticos y algorítmicos de un perceptrón de una sola capa.

* **Puro Python:** Implementado sin librerías externas de cálculo numérico, utilizando únicamente estructuras de datos nativas y el módulo `math`.
* **Funciones de Activación:** Definiciones manuales de funciones comunes y sus derivadas (`Identity`, `Sigmoid`, `Tanh`, `ReLU`).
* **Lógica Recursiva:** Las iteraciones de entrenamiento (`training`) están modeladas mediante llamadas recursivas.
* **Validación:** Resolución de compuertas lógicas linealmente separables (**OR** y **AND**).

---

### Versión 2.0 - Perceptrón Multicapa (`smallNN.py`)
Evolución hacia una arquitectura de **red neuronal de dos capas** (una capa oculta con 2 neuronas y una capa de salida), diseñada para superar la limitación lineal del perceptrón simple y resolver problemas no linealmente separables como la compuerta **XOR**.

* **Arquitectura Multicapa (MLP):** Propagación hacia adelante (*Forward Pass*) con 9 parámetros entrenables (pesos y sesgos distribuidos entre la capa oculta y la de salida).
* **Backpropagation Analítico:** Implementación manual de la retropropagación del error aplicando la regla de la cadena para el cálculo exacto del vector gradiente.
* **Vectorización con NumPy:** Incorporación de `numpy` para el manejo eficiente de los vectores de gradientes y la actualización simultánea de pesos.
* **Entrenamiento Iterativo e Inicialización Aleatoria:** Sustitución de la recursividad por bucles iterativos (`for`) para permitir entrenamientos profundos (decenas de miles de épocas) e introducción de inicialización aleatoria de pesos (`random`) para romper la simetría entre neuronas ocultas.
* **Validación:** Pruebas con compuertas lógicas **OR** y **XOR**, permitiendo experimentar con distintas funciones de activación (`Sigmoid`, `Tanh`, `ReLU`).

---

## 🛠️ Requisitos e Instalación

Para ejecutar `perceptron.py` no se requieren dependencias externas. Para ejecutar `smallNN.py`, es necesario contar con **NumPy**:

```bash
pip install numpy