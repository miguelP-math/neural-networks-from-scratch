# Neural Networks from Scratch (Perceptron)

## Versión 1.0 - Implementación Base (Sin Asistencia y Puro Python)

Este archivo (`perceptron.py`) es la primera versión preliminar del proyecto, desarrollada de forma totalmente autónoma (sin asistencia de IA) para entender los fundamentos matemáticos y de programación de un Perceptrón simple.

### Características principales:
- **Puro Python:** Implementado sin librerías externas de cálculo numérico como NumPy, utilizando únicamente estructuras de datos nativas y el módulo `math`.
- **Funciones de Activación:** Incluye definiciones manuales de funciones comunes y sus derivadas (`Identity`, `Sigmoid`, `Tanh`, `ReLU`).
- **Lógica Recursiva:** Las iteraciones de entrenamiento (`training`) están modeladas mediante llamadas recursivas en lugar de bucles tradicionales.
- **Validación con Compuertas Lógicas:** El script incluye pruebas automáticas al ejecutarse directamente con las compuertas lógicas **OR** y **AND**.

---

### Deducción Matemática (Derivadas Parciales del Error)

Para el entrenamiento del perceptrón, se ha calculado de forma analítica la función de error cuadrático medio y sus respectivas derivadas parciales respecto a los pesos ($w_0, w_1, w_2$)[cite: 1]:

* **Función de Error ($E$):**
  $$E(w, x, t) = \frac{1}{2} [f(w_0 + w_1 x_1 + w_2 x_2) - t]^2$$
  Definiendo $g = f(w_0 + w_1 x_1 + w_2 x_2) - t$[cite: 1].

* **Derivadas Parciales:**
  * Respecto al sesgo ($w_0$)[cite: 1]:
    $$\frac{\partial E}{\partial w_0} = g \cdot f'(w_0 + w_1 x_1 + w_2 x_2)$$
  * Respecto al peso $w_1$[cite: 1]:
    $$\frac{\partial E}{\partial w_1} = g \cdot f'(w_0 + w_1 x_1 + w_2 x_2) \cdot x_1$$
  * Respecto al peso $w_2$[cite: 1]:
    $$\frac{\partial E}{\partial w_2} = g \cdot f'(w_0 + w_1 x_1 + w_2 x_2) \cdot x_2$$