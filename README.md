# Práctica CNN comparativa: MNIST vs Fashion-MNIST

**Curso:** Inteligencia Artificial  
**Docente:** Lazo Mamani, Juan Carlos
**Unidad:** Unidad II - Redes Neuronales Convolucionales  
**Tema:** Arquitecturas CNN y Análisis de Resultados (MNIST vs. Fashion-MNIST)   
**Integrantes:** Delgado Ccorihuaman, Hamlet Nayeli / Leva Ayte, Kelma Ivonne

---

## Estructura del repositorio

![i1](/img1.png)

### .gitignore


---

## Parte 1: Práctica con Dígitos Escritos a Mano (MNIST)

### 1. Inspección Visual del Dataset

Al ejecutar el bloque con `matplotlib` se abre una ventana con una cuadrícula de 3×3 que muestra las primeras 9 imágenes del conjunto de entrenamiento (`X_train`). Cada imagen es un dígito escrito a mano, representado en escala de grises (`cmap='gray'`), con fondo negro y el trazo del dígito en tonos claros. Sobre cada imagen aparece su etiqueta real (`Etiqueta: 5`, `Etiqueta: 0`, etc.), lo que permite verificar que los datos y sus etiquetas están correctamente alineados.

Se observa que:
- Los trazos son simples, de una sola línea, y los dígitos están centrados en el lienzo de 28×28 píxeles.
- Hay variabilidad natural en la caligrafía (grosor del trazo, inclinación, tamaño).
- Algunas imágenes pueden verse más “ruidosas” o con bordes difusos por el proceso de digitalización.

**¿Por qué es útil ver los datos antes de entrenar?**  
Porque permite confirmar que las imágenes y sus etiquetas coinciden (evita errores de carga), entender el rango de valores (0–255) que luego se normalizará a 0–1, detectar problemas como imágenes mal centradas o corruptas, y tener una idea intuitiva de la dificultad del problema antes de introducirlo a la CNN.

![Visualización MNIST](capturas/mnist_visualizacion.png)

### 2. Inspección de `model.summary()` y Shapes

En el `model.summary()` obtenido se observa la evolución de las dimensiones:

```
Conv_1            (None, 26, 26, 32)     320
MaxPool_1         (None, 13, 13, 32)     0
Conv_2            (None, 11, 11, 64)     18,496
MaxPool_2         (None, 5, 5, 64)       0
Aplanado_Flatten  (None, 1600)           0
Capa_Densa_Oculta (None, 64)             102,464
Salida_Softmax    (None, 10)             650
Total params: 121,930
Trainable params: 121,930
Non-trainable params: 0
```

**¿Por qué `Conv_1` pasa de 28×28 a 26×26?**  
Porque se aplica un kernel de 3×3 **sin padding** (`padding='valid'` por defecto). La fórmula es:

\[
\text{salida} = \frac{W - K + 2P}{S} + 1
\]

Donde \(W = 28\), \(K = 3\), \(P = 0\), \(S = 1\). Entonces:

\[
\text{salida} = \frac{28 - 3 + 0}{1} + 1 = 26
\]

El kernel solo puede deslizarse sobre posiciones donde quepan sus 3×3 píxeles completos, perdiendo 1 píxel en cada borde.

**¿Cómo interviene `MaxPool_1`?**  
`MaxPooling2D(pool_size=(2,2))` toma bloques de 2×2 y se queda con el valor máximo, avanzando con stride 2. Por eso reduce cada dimensión a la mitad:

\[
\frac{26}{2} = 13
\]

Quedando un tensor de **13×13×32**. No aprende parámetros (por eso su conteo es 0), solo reduce la resolución espacial y conserva las activaciones más relevantes.

![model.summary MNIST](capturas/mnist_summary.png)

### 3. Análisis de Parámetros

**Parámetros entrenables de `Conv_1`:**  
Recibe un tensor de **28×28×1** y aplica **32 filtros** de 3×3. Cada filtro:

\[
3 \times 3 \times 1 = 9 \text{ pesos} + 1 \text{ bias} = 10
\]

Para 32 filtros:

\[
10 \times 32 = 320 \text{ parámetros}
\]

Coincide con la fila `Conv_1` del `summary`: **320**.

**Parámetros entrenables de `Conv_2`:**  
`Conv_2` recibe un tensor de **13×13×32** y aplica **64 filtros** de tamaño **3×3**. Cada filtro tiene:

\[
\text{pesos} = 3 \times 3 \times 32 = 288
\]

Más un **sesgo (bias)** por filtro:

\[
288 + 1 = 289 \text{ parámetros por filtro}
\]

Para 64 filtros:

\[
289 \times 64 = 18496 \text{ parámetros}
\]

Eso coincide exactamente con lo que muestra `model.summary()` en la fila de `Conv_2`: **18,496**.

**Función de la capa `Flatten`:**  
`Flatten` no tiene parámetros entrenables (por eso aparece con 0). Su función es **convertir el tensor 3D de salida de `MaxPool_2` (5×5×64) en un vector 1D**:

\[
5 \times 5 \times 64 = 1600
\]

Ese vector de 1600 elementos es el “puente estructural” que conecta las capas convolucionales (que trabajan con tensores espaciales) con las capas densas (que trabajan con vectores). Sin `Flatten`, la capa `Dense` no podría recibir la salida de las convoluciones.

### 4. Interpretación de Resultados en Consola

En la terminal se imprimió:

```
Vector de probabilidades crudas (Softmax): 
[0.000e+00 0.000e+00 0.000e+00 4.000e-04 0.000e+00 0.000e+00 0.000e+00
 9.993e-01 0.000e+00 3.000e-04]
Clase predicha por el modelo: 7
Clase real de la imagen: 7
```

**Análisis:**  
- El vector tiene **10 valores**, uno por cada clase (dígitos 0 al 9).  
- Cada valor está entre 0 y 1.  
- **La suma de los 10 valores es aproximadamente 1.0** (0.0000 + 0.0000 + 0.0000 + 0.0004 + 0.0000 + 0.0000 + 0.0000 + 0.9993 + 0.0000 + 0.0003 ≈ 1.0000). Eso representa que la salida Softmax es una **distribución de probabilidad**: la probabilidad de que la imagen pertenezca a cada una de las 10 clases.  
- En este caso, el valor más alto es **0.9993** en la posición 7, lo que indica que el modelo está prácticamente seguro de que la imagen es un “7”.

**¿Cómo actúa `np.argmax()`?**  
`np.argmax(predicciones)` recorre el vector y devuelve el **índice del valor más grande**. En este caso, el máximo está en la posición 7, entonces devuelve `7`. Ese índice se interpreta como la clase predicha.

**Contraste con la etiqueta real:**  
- `Clase predicha por el modelo: 7`  
- `Clase real de la imagen: 7`

Ambos coinciden, por lo que el modelo **acertó** en esta muestra. Esta comparación es la base para calcular métricas como `accuracy`.

Además, la precisión global tras 1 época con 5000 muestras fue **0.7570 (75.70%)** y la pérdida **0.8231**. Es un valor razonable considerando que solo se entrenó con el 8.3% del dataset y por una única época.

![Predicción MNIST](capturas/mnist_prediccion.png)

---

## Parte 2: Práctica con Clasificación de Prendas (Fashion-MNIST)

### 1. Inspección Visual del Dataset

Al visualizar las primeras 9 imágenes de Fashion-MNIST se observan prendas de vestir en escala de grises, con sus nombres correspondientes (`Camiseta/Top`, `Pantalón`, `Pulóver`, etc.). A diferencia de MNIST, las siluetas son más complejas: hay texturas, pliegues, formas irregulares y algunas clases son visualmente muy similares (por ejemplo, Camiseta/Top, Pulóver y Camisa). Además, las prendas no siempre están perfectamente centradas ni ocupan todo el lienzo de 28×28.

**Diferencias estructurales con MNIST:**  
- Los dígitos son trazos simples y centrados.  
- Las prendas tienen siluetas complejas, con variaciones de forma, textura y orientación.  
- La similitud entre clases es mayor en Fashion-MNIST, lo que dificulta la clasificación.

![Visualización Fashion-MNIST](capturas/fashion_visualizacion.png)

### 2. Análisis de la Salida Nominal

El arreglo `class_names` transforma un número entero abstracto (índice devuelto por `np.argmax`) en una categoría de vestimenta comprensible. Por ejemplo:

- Si `np.argmax(predicciones)` devuelve `9`, el modelo predice “Botín”.  
- Si devuelve `0`, predice “Camiseta/Top”.

En esta ejecución, el índice predicho fue **9**, y gracias a `class_names[9]` se mostró como **Botín**, en lugar del número “9”. Esto facilita la interpretación del vector Softmax y permite contrastar directamente con la etiqueta real (`y_test[0]`) usando el mismo arreglo.

### 3. Impacto de la Capa Densa Modificada

En MNIST, la capa densa oculta tiene 64 neuronas:

\[
1600 \times 64 + 64 = 102464 \text{ parámetros}
\]

En Fashion-MNIST, con 128 neuronas:

\[
1600 \times 128 + 128 = 204928 \text{ parámetros}
\]

Además, la capa de salida también cambia porque recibe 128 entradas en lugar de 64:

\[
128 \times 10 + 10 = 1290 \text{ parámetros}
\]

**Comparativa del total de parámetros:**

| Modelo | Capa Densa Oculta | Params totales |
|--------|-------------------|----------------|
| MNIST | 64 neuronas | 102,464 |
| Fashion-MNIST | 128 neuronas | 204,928 |

**Implicancias:**  
- Aumenta la capacidad del modelo para aprender patrones más complejos, lo cual es útil para las siluetas de ropa.  
- También incrementa el número total de parámetros (casi el doble) y el riesgo de sobreajuste, especialmente con pocas muestras de entrenamiento (5000 en esta práctica).  
- En un problema más difícil como Fashion-MNIST, una capa densa más grande puede ayudar a capturar mejor las diferencias entre clases, pero requiere más datos o regularización para generalizar bien. En esta ejecución, la accuracy fue **0.6624 (66.24%)**, menor que en MNIST, lo cual confirma que el problema es más difícil incluso con más capacidad.

![model.summary Fashion-MNIST](capturas/fashion_summary.png)

---

## Parte 3: Análisis Comparativo entre Ambos Algoritmos y Datasets

### 1. Comparativa de Métricas de Entrenamiento

Tras una época con 5000 muestras, los resultados obtenidos fueron:

| Dataset | Loss | Accuracy |
|---------|------|----------|
| MNIST | 0.8231 | 0.7570 (75.70%) |
| Fashion-MNIST | 0.9759 | 0.6624 (66.24%) |

**MNIST logró un aprendizaje más rápido y eficiente**: obtuvo mayor precisión (75.70% vs 66.24%) y menor pérdida (0.8231 vs 0.9759) en el mismo número de épocas y muestras. Esto ocurre porque los dígitos son más simples y homogéneos: los trazos son claros, centrados y con poca variabilidad entre clases. Fashion-MNIST, en cambio, presenta siluetas complejas, texturas y clases visualmente similares, lo que dificulta el aprendizaje en una sola época y con solo 5000 muestras.

### 2. Complejidad Visual y de Patrones

Los dígitos numéricos tienen trazos simples y centrados, lo que facilita su reconocimiento. Las prendas de vestir tienen siluetas complejas, con variaciones de forma, textura y orientación. Estas diferencias afectan la confianza del vector Softmax:

- **MNIST:** la probabilidad dominante fue **0.9993** hacia la clase 7, es decir, el modelo está prácticamente seguro de su predicción. El resto de probabilidades son cercanas a 0.
- **Fashion-MNIST:** la probabilidad más alta fue **0.6954** hacia la clase 9 (Botín), pero con valores secundarios importantes como **0.1413** (clase 7, Zapatilla) y **0.1174** (clase 5, Sandalía). El modelo tiene mucha menos certeza.

Esto demuestra que la complejidad del dominio repercute directamente en la confianza de la red.

### 3. Distribución de Confianza del Modelo

**Vector MNIST (suma ≈ 1.0):**

```
[0.0000 0.0000 0.0000 0.0004 0.0000 0.0000 0.0000 0.9993 0.0000 0.0003]
```

Distribución **fuertemente concentrada**: el 99.93% de la probabilidad se acumula en una sola clase. El modelo está seguro.

**Vector Fashion-MNIST (suma ≈ 1.0):**

```
[0.0004 0.0001 0.0026 0.0007 0.0003 0.1174 0.0008 0.1413 0.0411 0.6954]
```

Distribución **más dispersa**: aunque la clase 9 (Botín) domina con 69.54%, hay clases con probabilidades significativas como Zapatilla (14.13%) y Sandalía (11.74%). El modelo no está completamente seguro y “duda” entre calzados.

**Conclusión:** el dominio de los datos repercute directamente en la certeza de la red. A mayor complejidad visual (Fashion-MNIST), mayor dispersión en el vector Softmax y menor confianza en la predicción. En dominios simples y bien diferenciados (MNIST), el Softmax se concentra en una única clase y refleja alta certeza.

![Predicción Fashion-MNIST](capturas/fashion_prediccion.png)

---

## Parte 4: Subida Oficial del Proyecto a GitHub

Una vez validados ambos scripts en la consola de VS Code, comprobada la visualización gráfica de los datasets y redactado el informe completo en el archivo `README.md`, se sube el repositorio remoto ejecutando los siguientes comandos en la terminal integrada:

```bash
git init
git add .
git commit -m "Entrega completa: Practica comparativa CNN MNIST y Fashion-MNIST con visualizacion y analisis"
git branch -M main
git remote add origin https://github.com/tu-usuario/nombre-de-tu-repo.git
git push -u origin main
```

**Nota:** Crear previamente un repositorio vacío en GitHub sin README ni licencia para evitar conflictos.

![Repositorio GitHub](capturas/github_repo.png)

---
