Predicción de precios de vehículos BMW
Aplicación web desarrollada con Streamlit que permite predecir el precio de vehículos BMW a partir de sus características, utilizando un modelo de Machine Learning entrenado con datos históricos reales.

El flujo de que ha seguido el modelo ML para ser implementado en la web:

Preparación de datos
Entrenamiento del modelo
Evaluación con métricas
Selección del modelo más eficiente
Desarrollo de la App Web

Desarrollo del Front-End y Back-End con Streamlit
Implementación del modelo ML dentro de la App
Despliegue en una aplicación web interactiva en la nube, Stream Cloud
El uso de modelos de ensemble permitió mejorar significativamente la precisión de las predicciones, siendo Random Forest el modelo que ofreció los mejores resultados globales, por lo que fue seleccionado para su despliegue en la aplicación web.

Ver la app en la nube: https://prediccion-precios-web-gdnhtgdgkx5fx3qmrr9zzn.streamlit.app/

Tabla comparativa de métricas
Tras entrenar y evaluar distintos modelos de Machine Learning para la predicción del precio de vehículos BMW, se obtuvieron las siguientes métricas sobre el conjunto de prueba:

Modelo	MAE (€)	MSE (€²)	R²
Regresión Lineal	~3.200	~18.5 M	0.82
Árbol de Decisión	~2.900	~16.2 M	0.85
Random Forest	~2.100	~9.8 M	0.91
Descripción de las métricas:

MAE (Mean Absolute Error): error medio absoluto en euros.
MSE (Mean Squared Error): penaliza más los errores grandes.
R²: proporción de la varianza explicada por el modelo (más cercano a 1 es mejor).
Análisis crítico de los resultados
La Regresión Lineal, aunque sencilla e interpretable, presenta un error elevado, lo que indica que la relación entre las variables predictoras y el precio no es estrictamente lineal. Esto limita su capacidad para capturar interacciones complejas entre características como el modelo, el tipo de combustible o la transmisión.

El Árbol de Decisión mejora ligeramente los resultados al poder modelar relaciones no lineales. Sin embargo, sufre de una alta variabilidad y tiende a sobreajustarse a los datos de entrenamiento, lo que afecta a su capacidad de generalización.

El Random Forest ofrece el mejor rendimiento global. Al combinar múltiples árboles de decisión entrenados sobre subconjuntos aleatorios de los datos, reduce significativamente el sobreajuste y mejora la robustez del modelo. Esto se refleja en el menor MAE y MSE, así como en el mayor valor de R².

Por su parte, Gradient Boosting muestra un rendimiento muy competitivo y cercano al Random Forest. No obstante, su ajuste requiere un mayor cuidado en la selección de hiperparámetros y es más sensible al ruido en los datos.

Justificación del modelo seleccionado
Se seleccionó Random Forest Regressor como modelo final debido a:

Menor error medio absoluto (MAE), lo que implica predicciones más precisas en términos de precio.
Mayor valor de R², explicando aproximadamente el 91% de la variabilidad del precio.
Mayor estabilidad y robustez frente al sobreajuste en comparación con modelos más simples.
Capacidad para capturar relaciones no lineales y combinaciones complejas de variables.
Facilidad de integración en producción, ya que se comporta de forma consistente con datos no vistos.
En consecuencia, el modelo Random Forest representa el mejor equilibrio entre precisión, robustez y facilidad de despliegue para el problema planteado.

Ejecución del proyecto en local
Clonacnioón del repositorio git clone http://github.com/AlexDev101/prediccion_precios_web.git

Accedemos a la carpeta del proyecto cd prediccion_precios_web

Instalamos las herramientas necesarias con requeriments.txt pip install requeriments.txt

Instalamos Streamlit para poder levantar la app pip install streamlit

Por último levantamos el proyecto streamlit run app.py