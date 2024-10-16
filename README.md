<h1 align='center' style="font-weight:light; text-align:justify; margin-left: 80px; margin-right: 100px;">
  Desarrollo de una API para consulta sobre juegos Steam
</h1>


<h2 align='center'>
  Proyecto Individual I
</h2>



## Introducción:

En este proyecto, asumiré el rol de un MLOps Engineer y llevaré a cabo todos los procesos necesarios, desde el tratamiento y recolección de datos hasta el entrenamiento y despliegue del modelo. El objetivo principal es poder desarrollar y responder las consultas requeridas

## Objetivos del proyecto:
---
1. **Generación de API que procese funciones que responden a consultas acerca videojuegos**

2. **Deployment de un modelo de clasificación para un sistema de recomendación de videojuegos**

---
## Resumen de los procesos:
---
### 1. Proceso de Extracción, Transformación, Carga ( _enlace:_ [[ETL ](https://github.com/AballayNicolas/Proyecto1/blob/main/ETL.ipynb)]

En el archivo **ETL.py**, se llevó a cabo el proceso de extracción de datos de dos fuentes, la transformación de los datos para su limpieza y preprocesamiento, y finalmente la carga de los datos en un formato adecuado


### 2. Implementación de API´s ( _enlace:_ [main.py ](https://github.com/AballayNicolas/Proyecto1/blob/main/main.py))

En el archivo **main.py**, se creará una interfaz utilizando la biblioteca **FastAPI y Uvicorn**. Esta interfaz permitirá a los usuarios interactuar con el modelo, proporcionando los datos de entrada necesarios y obteniendo las predicciones correspondientes.


### 3. Análisis Exploratorio de Datos ( _enlace:_ [EDA ](https://github.com/AballayNicolas/Proyecto1/blob/main/EDA/EDA.ipynb))

En el notebook **EDA.ipynb**, se realizará un **`INFORME`** de Análisis exhaustivo de los datos y la factiblidad de modelos de clasificación para el caso en estudio. Esto incluirá la visualización de los datos,  y la generación de conclusiones relevantes entorno a las variables y la elección del modelo.


### 4. Desarrollo del Modelo de Machine Learning ( _enlace:_[modelo_Machine_Learning](https://github.com/AballayNicolas/Proyecto1/blob/main/Machine_learning.ipynb))

En el archivo **model**, se implementará un modelo de Machine Learning utilizando **Similitud de cosenos**. Este modelo se entrenó utilizando los datos preprocesados y preparados durante el ETL
