import nltk
import requests
from nltk.tokenize import word_tokenize
from django.shortcuts import render
from django.db import connection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def buscar_unidades_pensamientoDHH(parrafo, similitudUsuario):
    # Obtener versículos desde la API usando la palabra clave y sus sinónimos
    versiculos = buscar_versiculos_con_sinonimos(parrafo)
    
    if not versiculos:
        print("No se encontraron versículos.")
        return []

    # Procesar los datos obtenidos
    contenidos = [versiculo['verse'] for versiculo in versiculos]
    referencias = [(versiculo['book'], versiculo['chapter'], versiculo['number']) for versiculo in versiculos]

    # Calcular similitud usando TF-IDF
    vectorizador = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        smooth_idf=False
    )

    tfidf_matrix = vectorizador.fit_transform(contenidos)
    parrafo_tfidf = vectorizador.transform([parrafo])
    similitudes = cosine_similarity(parrafo_tfidf, tfidf_matrix)

    # Filtrar y ordenar por similitud
    unidades_similares = []
    for similitud in sorted(similitudes[0], reverse=True):
        if similitud >= similitudUsuario:
            indice_similitud = list(similitudes[0]).index(similitud)
            unidad_pensamiento_similar = contenidos[indice_similitud]
            referencia_similar = referencias[indice_similitud]
            unidades_similares.append({
                'similitud': similitud,
                'contenido': unidad_pensamiento_similar,
                'libro': referencia_similar[0],
                'capitulo': referencia_similar[1],
                'versiculo': referencia_similar[2]
            })
        
    return unidades_similares

def buscar_versiculos_con_sinonimos(parrafo):
    palabras = word_tokenize(parrafo)
    versiculos = []

    # Para cada palabra en el párrafo, obtiene versículos similares
    for palabra in palabras:
        versiculos += buscar_versiculos(palabra)

        # Obtener palabras similares a través de Word2Vec y buscar versículos para cada una
        sinonimos = obtener_sinonimos_word2vec(palabra)
        for sinonimo in sinonimos:
            versiculos += buscar_versiculos(sinonimo)

    # Eliminar duplicados de versículos
    versiculos_unicos = {v['id']: v for v in versiculos}.values()
    return list(versiculos_unicos)

def buscar_versiculos(palabra):
    # Llamada a la API de la Biblia para obtener versículos por palabra
    url = f"https://bible-api.deno.dev/api/read/nvi/search?q={palabra}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json().get('data', [])
        return [
            {
                'verse': item['verse'],
                'book': item['book'],
                'chapter': item['chapter'],
                'number': item['number'],
                'id': item['id']
            } for item in data
        ]
    else:
        print(f"Error al buscar versículos para '{palabra}':", response.status_code)
        return []


def obtener_sinonimos_word2vec(palabra):
    # Llamada a la API de Word2Vec para obtener sinónimos
    url = f"http://127.0.0.1:5000/?word={palabra}"
    response = requests.get(url)
    sinonimos = []
    if response.status_code == 200:
        datos = response.json()
        for resultado in datos:
            palabra_clave = resultado[0]
            similitud = resultado[1]
            if similitud >= 0.05:
                sinonimos.append(palabra_clave)
    else:
        print(f"No se encuentra en el vocabulario sinónimos para '{palabra}':", response.status_code)
    
    return sinonimos

# Funciones de vistas para Django
def index(request):
    return render(request, 'unidades_pensamientoDHH/index.html')

def resultados(request):
    if request.method == 'GET' and 'q' in request.GET:
        query = request.GET.get('q', '').strip()
        similitud_usuario_str = request.GET.get('similitud', '').strip()
        try:
            similitud_usuario = float(similitud_usuario_str)
        except ValueError:
            similitud_usuario = 0.05
        resultados = buscar_unidades_pensamientoDHH(query, similitud_usuario)
        if not resultados:
            return render(request, 'unidades_pensamientoDHH/notfound.html', {'query': query})
        else:
            return render(request, 'unidades_pensamientoDHH/resultados.html', {'resultados': resultados, 'query': query})
    return render(request, 'unidades_pensamientoDHH/index.html')