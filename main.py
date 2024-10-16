from fastapi import FastAPI
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

#uvicorn main:app --reload
#http://127.0.0.1:8000

app = FastAPI()



@app.get("/")
def mensaje():
    return "Esta es una API para realizar consultas de juegos en Steam."

#--------------------------------------------------------------------------------

# Funcion para devolver la cantidad de contenido free por developer
@app.get('/developer/')
def get_developer_stats(desarrollador: str):

    df = pd.read_csv('Datasets/steam_games_normalizado.csv')


    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')


    developer_df = df[df['developer'] == desarrollador]


    items_por_año = developer_df.groupby(df['release_date'].dt.year).size().reset_index(name='Cantidad de Items')


    free_por_año = developer_df[developer_df['price'] == 'Free'].groupby(df['release_date'].dt.year).size().reset_index(name='Contenido Free')


    result_df = items_por_año.merge(free_por_año, on='release_date', how='left')
    result_df['Contenido Free'] = (result_df['Contenido Free'] / result_df['Cantidad de Items'] * 100).fillna(0).astype(int).astype(str) + '%'


    result_df.rename(columns={'release_date': 'Año'}, inplace=True)

    return result_df.to_dict(orient='records')



# ---------------------------------------------------------------------------------------------------------------------------------------------------------


# Funcion para devolver la cantidad de dinero gastado por el usuario y el porcentaje de recomendacion y cantidad de items
@app.get("/userdata/")
def get_user_data(User_id: str):
    df_games = pd.read_csv('Datasets/steam_games_normalizado.csv')
    df_items = pd.read_parquet('user_items_normalizado.parquet')
    df_reviews = pd.read_csv("Datasets/user_reviews_Analisis_sentimientos.csv")

    games_copy = df_games.copy()
    items_copy = df_items.copy()
    reviews_copy = df_reviews.copy()

    # Convertir la columna 'user_id' a str
    items_copy['user_id'] = items_copy['user_id'].astype(str)
    reviews_copy['user_id'] = reviews_copy['user_id'].astype(str)

    # Filtrar df_items por el user_id dado
    user_items = items_copy[items_copy['user_id'] == str(User_id)]

    # Calcular la cantidad de dinero gastado por el usuario
    # Convertir la columna 'price' a numérica
    games_copy['price'] = pd.to_numeric(games_copy['price'], errors='coerce')
    money_spent = user_items.merge(games_copy[['id', 'price']], left_on='item_id', right_on='id')['price'].sum()
#Calcular la cantidad total de items del usuario
    total_items = user_items['items_count'].sum()

    # Filtrar df_reviews por el user_id dado
    user_reviews = reviews_copy[reviews_copy['user_id'] == str(User_id)]
#Calcular el porcentaje de recomendación promedio del usuario
    if user_reviews.shape[0] > 0:
        recommendation_percentage = (user_reviews['recommend'].sum() / user_reviews.shape[0]) * 100
    else:
        recommendation_percentage = 0

    # Convertir valores de numpy.int64 a tipos de datos estándar
    money_spent = float(money_spent) if not pd.isnull(money_spent) else 0.0  # Convertir a float, manejar NaN si es necesario
    recommendation_percentage = float(recommendation_percentage) if not pd.isnull(recommendation_percentage) else 0.0  # Convertir a float, manejar NaN si es necesario

#Crear el diccionario de resultados
    result = {
        "Usuario": str(User_id),
        "Dinero gastado": f"{money_spent:.2f} USD",  # Ajustamos el formato para mostrar dos decimales
        "% de recomendación": f"{recommendation_percentage:.2f}%",
        "Cantidad de items": int(total_items)
    }

    return JSONResponse(content=result)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Funcion para devolver el usuario con mas horas jugadas del genero dado
@app.get('/user-for-genre/')
def user_for_genre(genero: str):
    """
    Analiza el género dado y devuelve el usuario con más horas jugadas en ese género
    y el total de horas jugadas por año.

    Parámetros:
    - genero: Género del juego (str).

    Retorna:
    - el usuario que más ha jugado y las horas jugadas por año.
    """
    try:
        # Cargar los datasets
        df_games = pd.read_csv('Datasets/steam_games_normalizado.csv')
        df_items = pd.read_parquet('user_items_normalizado.parquet')

        # Verificar la existencia de la columna 'genres'
        if 'genres' not in df_games.columns:
            raise HTTPException(status_code=400, detail="El DataFrame no tiene una columna llamada 'genres'.")

        # Convertir la columna de fecha a formato datetime
        df_games['release_date'] = pd.to_datetime(df_games['release_date'], errors='coerce')

        # Filtrar juegos por el género especificado
        juegos_genero = df_games[df_games['genres'] == genero]

        # Unir datos de juegos con datos de usuarios
        juegos_usuario = pd.merge(juegos_genero, df_items, left_on='id', right_on='item_id')

        # Agrupar las horas jugadas por usuario y calcular la suma de horas
        horas_por_usuario = juegos_usuario.groupby('user_id')['playtime_forever'].sum()

        # Obtener el usuario con más horas jugadas
        usuario_max_horas = horas_por_usuario.idxmax()
        max_horas = horas_por_usuario.loc[usuario_max_horas]

        # Agrupar las horas jugadas por año y calcular la suma de horas
        horas_por_año = juegos_usuario.groupby(juegos_usuario['release_date'].dt.year)['playtime_forever'].sum()
        horas_por_año = horas_por_año.reset_index(name='Horas')

        # Convertir el DataFrame a una lista de diccionarios para una salida más clara
        horas_por_año = horas_por_año.to_dict('records')

        # Preparar el resultado
        result = {
            "Usuario con más horas jugadas en el género {}: ".format(genero): usuario_max_horas,
            "Horas jugadas por el usuario": max_horas,
            "Horas jugadas por año": horas_por_año
        }

        # Devolver la respuesta como JSON
        return JSONResponse(content=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"No se pudo encontrar el archivo: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Se produjo un error: {str(e)}")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Funcion para devolver una desarrolladora con su cantidad de reseñas negativas y positivas
@app.get("/developer_reviews_analysis/")
def developer_reviews_analysis(desarrollador: str):
    """
    Analiza las reseñas de un desarrollador específico, mostrando la cantidad de reseñas positivas y negativas.

    Parámetros:
    - desarrollador: Nombre del desarrollador (str).
    
    Retorna:
    - la cantidad de reseñas positivas y negativas, o un mensaje de error si el desarrollador no existe.
    """
    try:
        # Cargar los datasets
        games = pd.read_csv('Datasets/steam_games_normalizado.csv')
        sentiment = pd.read_csv('Datasets/user_reviews_Analisis_sentimientos.csv')

        # Combinar los conjuntos de datos en las columnas apropiadas ('item_id' en reviews y 'id' en games)
        merged_data = pd.merge(sentiment, games, left_on='item_id', right_on='id')

        # Filtrar filas donde el puntaje de sentimiento es positivo (2) o negativo (0)
        filtered_data = merged_data[merged_data['analisis_sentimiento'] != 1]  # Excluir sentimiento neutral

        # Agrupar por desarrollador y puntaje de sentimiento, contar la cantidad de reseñas
        grouped_data = (
            filtered_data.groupby(['developer', 'analisis_sentimiento'])
            .size()
            .unstack(fill_value=0)
        )

        # Verificar si el desarrollador está en el DataFrame
        if desarrollador not in grouped_data.index:
            return JSONResponse(
                content={"error": f"No se encontró información sobre el desarrollador {desarrollador}"},
                status_code=404
            )

        # Extraer cantidad de reseñas positivas y negativas para el desarrollador especificado
        developer_reviews = grouped_data.loc[desarrollador]

        # Convertir los valores a tipos de datos estándar de Python (int)
        developer_reviews_summary = {
            "Negativas": int(developer_reviews.get(0, 0)),
            "Positivas": int(developer_reviews.get(2, 0))
        }

        # Devolver la respuesta como JSON
        return JSONResponse(content={desarrollador: developer_reviews_summary})

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"No se pudo encontrar el archivo: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Se produjo un error: {str(e)}")

    
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Cargar el archivo CSV y crear la matriz de géneros

archivo = r'Datasets/steam_games_normalizado.csv'
df_games = pd.read_csv(archivo)
df_subset = df_games.head(20000)  # Hacemos un recorte más pequeño del dataframe

# Llenar valores NaN en la columna 'genres' con una cadena vacía para evitar problemas
df_subset['genres'] = df_subset['genres'].fillna('')

# Usamos CountVectorizer para convertir los géneros a una matriz de conteo
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
genres_matrix = vectorizer.fit_transform(df_subset['genres'])

# Calcular la matriz de similitud del coseno
cosine_sim = cosine_similarity(genres_matrix)

@app.get("/recomendaciones/")
def obtener_similitud_juego(nombre_juego: str, n_recomendaciones: int = 5):
    """
    Retorna una lista de juegos similares al nombre de juego dado, utilizando la similitud de coseno en los géneros.

    Parámetros:
    - nombre_juego: Nombre del juego para el cual buscar recomendaciones (str).
    - n_recomendaciones: Número de recomendaciones a devolver (int, por defecto 5).

    Retorna:
    - JSONResponse con una lista de juegos recomendados.
    """
    try:
        # Verificar si el juego existe en el dataframe
        if nombre_juego not in df_subset['app_name'].values:
            return JSONResponse(
                content={"error": f"El juego '{nombre_juego}' no se encuentra en la base de datos."},
                status_code=404
            )

        # Obtener el índice del juego especificado
        idx = df_subset[df_subset['app_name'] == nombre_juego].index[0]

        # Obtener el vector del juego especificado y calcular las similitudes
        juego_vector = genres_matrix[idx]
        similitudes = cosine_similarity(juego_vector, genres_matrix).flatten()

        # Ordenar los índices de juegos similares y obtener los n_recomendaciones más altos
        indices_similares = similitudes.argsort()[-(n_recomendaciones + 1):-1][::-1]

        # Obtener los nombres de los juegos similares
        juegos_similares = df_subset.iloc[indices_similares]['app_name'].tolist()

        # Preparar la respuesta como JSON
        result = {
            "Juego Base": nombre_juego,
            "Recomendaciones": juegos_similares
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            content={"error": f"Ocurrió un error: {str(e)}"},
            status_code=500
        )


      