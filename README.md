<h1 align='center' style="font-weight:light; text-align:justify; margin-left: 80px; margin-right: 100px;">
  Desarrollo de un Modelo de Clasificación para un Sistema de Recomendación de Películas
</h1>

<p align="center">
  <img src="Image/logo.png" alt="Logo del Proyecto" style="float:right; width:80px;" />
</p>

<h2 align='center'>
  Proyecto Individual I 🎬
</h2>



## Introducción:

En este proyecto de Machine Learning, asumiré el rol de un Data Engineer y ML Operations y llevaré a cabo todos los procesos necesarios, desde el tratamiento y recolección de datos hasta el entrenamiento y despliegue del modelo de ML. El objetivo principal es desarrollar un sistema de recomendación de películas basado en técnicas de similitud y algoritmos de Machine Learning.

## Objetivos del proyecto:
---
1. **Generación de API´s que procesan funciones que responden a consultas acerca de características de películas**

2. **Deployment de un modelo de clasificación para un sistema de recomendación de películas**

---
## Resumen de los procesos:
---
### 1. Proceso de Extracción, Transformación, Carga ( _enlace:_ [ETL ](https://github.com/abelfranco/PI_ML/blob/master/ETL.ipynb))

En el archivo **ETL.py**, se llevó a cabo el proceso de extracción de datos de dos fuentes, la transformación de los datos para su limpieza y preprocesamiento, y finalmente la carga de los datos en un formato adecuado (archivo **ds_clean.csv**) para su posterior análisis y entrenamiento del modelo. (tambien ver **data_dictionary.md**)


### 2. Implementación de API´s ( _enlace:_ [main.py ](https://github.com/abelfranco/PI_ML/blob/master/main.py))

En el archivo **main.py**, se creará una interfaz utilizando la biblioteca **FastAPI y Uvicorn**. Esta interfaz permitirá a los usuarios interactuar con el modelo de Machine Learning, proporcionando los datos de entrada necesarios y obteniendo las predicciones correspondientes.


### 3. Análisis Exploratorio de Datos ( _enlace:_ [EDA ](https://github.com/abelfranco/PI_ML/blob/master/EDA.ipynb))

En el notebook **EDA.ipynb**, se realizará un **`INFORME`** de Análisis exhaustivo de los datos y la factiblidad de modelos de clasificación para el caso en estudio. Esto incluirá la visualización de los datos, reducción de dimensionalidad, tratamiento de valores atípicos y la generación de conclusiones relevantes entorno a las variables y la elección del modelo.


### 4. Desarrollo del Modelo de Machine Learning ( _enlace:_ [model.py ](https://github.com/abelfranco/PI_ML/blob/master/model.py))

En el archivo **model.py**, se implementará un modelo de Machine Learning utilizando **Similitud de cosenos**. Este modelo se entrenó utilizando los datos preprocesados y preparados durante el EDA (archivo **ds_model.csv**).Finalmente se realizó el deployemnt de la aplicación usando [RENDER ](https://dashboard.render.com/web/srv-cijd6sd9aq01qqirngrg).

---
- _"Cada etapa fué realizada de manera sistemática y documentada adecuadamente en los archivos correspondientes, siguiendo las mejores prácticas de ciencia de datos y ML Operations, para garantizar la reproducibilidad, calidad y mantenibilidad del proyecto"_
---

<div style="display:flex; align-items:center;">
  <div style="width:50%; padding-right:20px;">
    <h2>Herramientas Utilizadas</h2>
    <ul style="text-align: justify;">
      <li><b>📊Scikit Learn</b>: Utilizado para vectorizar, tokenizar y calcular la similitud coseno.</li>
      <li><b>🐍Python</b>: Lenguaje de programación principal utilizado en el desarrollo del proyecto.</li>
      <li><b>💻Numpy</b>: Utilizado para realizar operaciones numéricas y manipulación de datos.</li>
      <li><b>🐼Pandas</b>: Utilizado para la manipulación y análisis de datos estructurados.</li>
      <li><b>📈Matplotlib</b>: Utilizado para la visualización de datos y generación de gráficos.</li>
      <li><b>📳FastAPI</b>: Utilizado para crear la interfaz de la aplicación y procesar los parámetros de funciones.</li>
      <li><b>🦄Uvicorn</b>: Servidor ASGI utilizado para ejecutar la aplicación FastAPI.</li>
      <li><b>🌐Render</b>: Plataforma utilizada para el despliegue del modelo y la aplicación.</li>
    </ul>
  </div>
  <div style="width:50%; text-align:center;">
    <figure>
      <img src="Image/infograph.jpg" alt="Ejemplo del deployment usando Heroku(Render)" style="margin-left:auto; margin-right:auto;" />
      <figcaption style="font-size: smaller; font-style: italic; text-align: center;">Descripción del despliegue de una aplicación desde un repositorio en GitHub usando Heroku (similar a Render)</figcaption>
    </figure>
  </div>
</div>

## Links

- [API´s de consultas de películas y Modelo de sistema de recomendación](https://movies-recomendation-system-bgw9.onrender.com/docs#/)
- [Videotutorial del trabajo realizado en YouTube](#)

## Recomendaciones

- Al hacer las consultas Usar la primera letra de cada palabra en mayusculas.
- No hacer uso de caracteres especiales.
- Para la función **peliculas_idioma**, ingresar solamente las abreviaturas del idioma (por ejemplo, "inglés" sería "en").

## Autor

Abel Franco Ccapa

- Correo electrónico: abel.ccapa@tecsup.edu.pe

- LinkedIn: [Perfil de LinkedIn](https://www.linkedin.com/in/abelfrancoccapa)
