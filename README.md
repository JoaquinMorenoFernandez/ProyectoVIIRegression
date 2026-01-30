Proyecto de Predicción de Precios de Vehículos

Este proyecto consiste en una herramienta digital capaz de estimar el precio de mercado de un coche de segunda mano. Para lograrlo, hemos procesado miles de datos históricos y creado un modelo matemático que ha aprendido a valorar vehículos basándose en sus características.

1. Obtención y Limpieza de la Información (Data Cleaning)

Lo primero que hace el código en el archivo de análisis es tomar los datos brutos (used_cars.csv). Estos datos venían "sucios" y el ordenador no podía entenderlos directamente.

   El código elimina símbolos de moneda (como el signo $) y unidades de medida (como "millas") para convertir el texto en números puros. También rellena huecos donde faltaba información y corrige errores tipográficos.

  Su función: Transformar una hoja de cálculo desordenada en una tabla matemática limpia que el algoritmo pueda procesar sin errores.

2. Traducción de Datos (Encoding)

El ordenador no entiende palabras como "Toyota", "Gasolina" o "Automático", solo entiende números.

  Lo que hicimos: Aplicamos un proceso llamado "codificación". Asignamos un número único a cada marca, modelo y tipo de combustible. Además, guardamos este "diccionario de traducción" para usarlo más tarde.

  Su función: Permitir que las características cualitativas del coche sean parte de la ecuación matemática del precio.

3. Entrenamiento del Modelo (Machine Learning)

Esta es la parte central del proyecto. Usamos los datos limpios para enseñar al programa.

  Lo que hicimos: Dividimos los datos en dos grupos: uno para estudiar y otro para hacer un examen. Probamos varios algoritmos (fórmulas matemáticas complejas) y seleccionamos el que cometía menos errores al adivinar los precios del grupo de examen.

  Su función: Crear un "cerebro" digital (best_model.pkl) que ha aprendido los patrones del mercado (por ejemplo, cuánto valor pierde un coche por cada año de antigüedad o por cada kilómetro recorrido).

4. Creación de la Interfaz de Usuario (App)

El archivo app.py es la herramienta que utiliza una persona normal para interactuar con nuestro modelo matemático.

  Lo que hicimos: Creamos un formulario donde el usuario selecciona las características de su coche (marca, año, motor, etc.).

  Su función: Recoger los datos del usuario, aplicarles la misma "traducción" que usamos en el paso 2 y enviárselos al "cerebro" entrenado en el paso 3 para que devuelva un precio estimado.

5. Visualización Comparativa

No basta con dar un número; el usuario necesita contexto.

  Lo que hicimos: El código busca en la base de datos original coches que sean muy parecidos al que el usuario está consultando y genera gráficos.

  Su función: Mostrar al usuario dónde se sitúa su coche en comparación con el mercado real (por ejemplo, un gráfico que muestra si su coche es más caro o barato que otros con el mismo kilometraje).

6. Registro de Consultas (Logging)

Finalmente, el sistema no solo da respuestas, sino que guarda información nueva.

  Lo que hicimos: Programamos el sistema para que, cada vez que alguien hace una consulta, se guarde en un archivo nuevo (production_logs.csv) junto con la predicción realizada.

  Su función: Crear un historial de uso que nos servirá en el futuro para ver qué coches busca la gente y reentrenar el modelo para que sea cada vez más preciso.


  --------------------------------------------------------------

  Cosas como el K-Fold explicadas:

  1. Validación Cruzada (K-Fold)
¿Qué es? Imagina que estás estudiando para un examen. En lugar de estudiar siempre con las mismas preguntas (lo que te haría memorizar las respuestas), divides tu libro en 5 partes. Estudias con 4 y te examinas con 1. Luego rotas las partes y repites el proceso 5 veces. Así te aseguras de que realmente sabes, no solo que memorizaste. Eso es K-Fold
