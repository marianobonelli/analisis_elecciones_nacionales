import streamlit as st
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Cargar la imagen
page_icon = Image.open("assets/elecciones.png")

st.set_page_config(
    page_title="Analisis Elecciones 2023",
    page_icon=page_icon,
    layout="wide",
)

# Leer el archivo JSON
with open('assets/indice.json', 'r', encoding='utf-8') as file:
    indice = json.load(file)

def generar_df(env, scope):
    with open(f'assets/{env}/{env}_{scope}.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    max_level = max([item.get('level', 0) for item in data['mapa']])

    scopes_names = []
    partidos_names = []
    votos_values = []
    codigo_ids = []

    for item in data['mapa']:
        if item.get('level', 0) == max_level:
            for scope in item['scopes']:
                scope_id = scope['codigo']
                for partido in scope['partidos']:
                    scopes_names.append(scope['name'][:13])
                    partidos_names.append(partido['name'][:13])
                    votos_values.append(partido['votos'])
                    codigo_ids.append(scope_id)
                scopes_names.append(scope['name'][:13])
                partidos_names.append('BLANCOS')
                votos_values.append(scope['blancos'])
                codigo_ids.append(scope_id)

    df = pd.DataFrame({
        'ID': codigo_ids,
        'Scope': scopes_names,
        'Partido': partidos_names,
        'Votos': votos_values
    })

    df = df.pivot(index=['Scope', 'ID'], columns='Partido', values='Votos')
    df = df.fillna(0).astype(int)

    # Usar solo 'Scope' como índice
    df.index = df.index.get_level_values('Scope')

    return df

def simplificar_df(df, partidos_generales):
    # Truncar los nombres de los partidos generales a 13 caracteres
    partidos_generales = [partido[:13] for partido in partidos_generales]
    
    # Identificar los partidos que no están en las elecciones generales
    partidos_a_eliminar = [partido for partido in df.columns if partido not in partidos_generales and partido != 'BLANCOS']
    
    # Sumar los votos de esos partidos y crear una nueva columna 'Otros'
    if partidos_a_eliminar:
        df['Otros'] = df[partidos_a_eliminar].sum(axis=1)
    
    # Eliminar las columnas de los partidos que no están en las elecciones generales
    df = df.drop(columns=partidos_a_eliminar)
    
    # Ordenar las columnas para que 'Otros' esté primera, si existe
    if 'Otros' in df.columns:
        columnas = ['Otros'] + [col for col in df.columns if col != 'Otros']
        df = df[columnas]
    
    return df

def agregar_total(df):
    df['TOTAL'] = df.sum(axis=1)
    df.loc['TOTAL'] = df.sum(axis=0)
    return df

def mostrar_heatmap_votos(df, env):
    if env == "DIFERENCIA":
        # Orden específico de columnas para el gráfico de diferencia de votos
        orden_columnas = ['Otros', 'BLANCOS', 'FRENTE DE IZQ', 'HACEMOS POR N', 'JUNTOS POR EL', 'LA LIBERTAD A', 'UNION POR LA ', 'TOTAL']
        
        # Asegurarse de que todas las columnas en el orden específico estén presentes
        for columna in orden_columnas:
            if columna not in df.columns:
                df[columna] = 0
                
        df = df[orden_columnas]

    df = agregar_total(df.copy())
    fmt = ','
    vmax = np.nanmax(df.drop('TOTAL', errors='ignore').values)
    figsize = (max(10, len(df.columns)*1), max(8, len(df.index)*0.3))

    plt.figure(figsize=figsize)
    sns.heatmap(df, cmap="YlGnBu", annot=True, fmt=fmt, linewidths=.5, vmax=vmax, cbar=False)
    plt.title("Votos")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return df

def mostrar_heatmap_porcentaje_total(df):
    total_votos = df.values.sum()
    df_percentage = (df / total_votos) * 100

    # Calcular columna TOTAL
    df_percentage['TOTAL'] = (df.sum(axis=1) / total_votos) * 100
    df_percentage.loc['TOTAL'] = df_percentage.sum(axis=0)

    plt.figure(figsize=(max(10, len(df.columns)*1), max(8, len(df.index)*0.3)))
    sns.heatmap(df_percentage, cmap="YlGnBu", annot=True, fmt='.2f', linewidths=.5, vmax=50, cbar=False)
    plt.title("Porcentaje sobre el total")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return df_percentage

def mostrar_heatmap_porcentaje_partido(df, total_col):
    total_by_party = df.sum(axis=0)
    df_percentage = (df.divide(total_by_party, axis=1)) * 100

    df_percentage['TOTAL'] = total_col
    df_percentage.loc['TOTAL'] = df_percentage.sum(axis=0)

    plt.figure(figsize=(max(10, len(df.columns)*1), max(8, len(df.index)*0.3)))
    sns.heatmap(df_percentage, cmap="YlGnBu", annot=True, fmt='.2f', linewidths=.5, vmax=50, cbar=False)
    plt.title("Porcentaje por partido")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return df_percentage

def calcular_porcentaje_total(df):
    total_votos = df.values.sum()
    df_percentage = (df / total_votos) * 100
    df_percentage['TOTAL'] = (df.sum(axis=1) / total_votos) * 100
    df_percentage.loc['TOTAL'] = df_percentage.sum(axis=0)
    return df_percentage

def reordenar_columnas(df):
    if 'Otros' in df.columns:
        columnas_ordenadas = ['Otros'] + [col for col in df.columns if col != 'Otros']
        df = df[columnas_ordenadas]
    return df

def calcular_diferencia_porcentaje(df_generales, df_paso):
    # Asegurarse de que ambos DataFrames tienen las mismas columnas
    for col in df_generales.columns:
        if col not in df_paso.columns:
            df_paso[col] = 0
    for col in df_paso.columns:
        if col not in df_generales.columns:
            df_generales[col] = 0
    
    # Calcular el porcentaje sobre el total para cada DataFrame
    df_generales_porcentaje = calcular_porcentaje_total(df_generales)
    df_paso_porcentaje = calcular_porcentaje_total(df_paso)
    
    # Calcular la diferencia en porcentaje
    df_diferencia = df_generales_porcentaje - df_paso_porcentaje
    
    return df_diferencia

def heatmap_porcentaje_total(df, df_percentage_total):
    plt.figure(figsize=(max(10, len(df.columns)*1), max(8, len(df.index)*0.3)))
    sns.heatmap(df_percentage_total, cmap="YlGnBu", annot=True, fmt='.2f', linewidths=.5, vmax=50, cbar=False)
    plt.title("Porcentaje sobre el total")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return df_percentage_total

def heatmap_diferencia_total_otros(df_diferencia_porcentaje):
    if 'Otros' not in df_diferencia_porcentaje.columns:
        df_diferencia_porcentaje['Otros'] = 0

    # Orden específico de columnas
    orden_columnas = ['Otros', 'BLANCOS', 'FRENTE DE IZQ', 'HACEMOS POR N', 'JUNTOS POR EL', 'LA LIBERTAD A', 'UNION POR LA ', 'TOTAL']

    # Asegurarse de que todas las columnas en el orden específico estén presentes
    for columna in orden_columnas:
        if columna not in df_diferencia_porcentaje.columns:
            df_diferencia_porcentaje[columna] = 0
            
    df_diferencia_porcentaje = df_diferencia_porcentaje[orden_columnas]

    plt.figure(figsize=(max(10, len(df_diferencia_porcentaje.columns)*1), max(8, len(df_diferencia_porcentaje.index)*0.3)))
    sns.heatmap(df_diferencia_porcentaje, cmap="RdYlGn", annot=True, fmt='.2f', linewidths=.5,vmin=-10, vmax=10, cbar=False)
    plt.title("Diferencia de Porcentaje sobre el Total")
    plt.xticks(rotation=45)

    return df_diferencia_porcentaje


############################################################################################
# main
############################################################################################

st.title("Análisis de Elecciones Nacionales")
st.markdown("""
            [![LinkedIn Badge](https://img.shields.io/badge/marianobonelli-LinkedIn-blue)](https://www.linkedin.com/in/mariano-francisco-bonelli/) [![Twitter Badge](https://img.shields.io/badge/marianobonelli-Twitter-blue?logo=twitter)](https://twitter.com/marianobonelli)
            """)

# Seleccionar Env, Scope y Tipo de Gráfico
env = st.selectbox("Elecciones:", ['GENERALES', 'PASO', 'DIFERENCIA'])
default_index = list(indice.keys()).index('Nacionales') # Asegúrate de que 'indice' esté definido
scope = st.selectbox("Nivel de análisis:", list(indice.keys()), index=default_index)

# Lista de partidos presentes en las elecciones generales
partidos_generales = ['BLANCOS', 'FRENTE DE IZQ', 'HACEMOS POR N', 'JUNTOS POR EL', 'LA LIBERTAD A', 'UNION POR LA ']

if env != 'DIFERENCIA':
    df = generar_df(env, scope)
    if env == 'PASO':
        df = simplificar_df(df, partidos_generales)
        if 'Otros' not in df.columns:
            df['Otros'] = 0
    graph_type = st.selectbox("Tipo de gráfico:", ['Votos', 'Porcentaje Total', 'Porcentaje por Partido'])
else:
    df_generales = generar_df('GENERALES', scope)
    df_paso = generar_df('PASO', scope)
    df_paso = simplificar_df(df_paso, partidos_generales)
    if 'Otros' not in df_generales.columns:
        df_generales['Otros'] = 0
    df = df_generales - df_paso
    # Nota: No estamos reordenando las columnas aquí para "Porcentaje Total Diferencia"
    graph_type = st.selectbox("Tipo de gráfico:", ['Votos', 'Porcentaje Total Diferencia'])

############################################################################################
# Heatmaps
############################################################################################

if graph_type == "Votos":
    mostrar_heatmap_votos(df, env)
    st.pyplot(plt.gcf(), use_container_width=True)
    st.dataframe(data=df, column_order=['Otros', 'BLANCOS', 'FRENTE DE IZQ', 'HACEMOS POR N', 'JUNTOS POR EL', 'LA LIBERTAD A', 'UNION POR LA ', 'TOTAL'])

elif graph_type == "Porcentaje Total":
    df_percentage_total = calcular_porcentaje_total(df)
    heatmap_porcentaje_total(df, df_percentage_total)
    st.pyplot(plt.gcf(), use_container_width=True)
    st.dataframe(data=df_percentage_total)

elif graph_type == "Porcentaje por Partido":
    df_percentage_total = calcular_porcentaje_total(df)
    mostrar_heatmap_porcentaje_partido(df, df_percentage_total['TOTAL'])
    st.pyplot(plt.gcf(), use_container_width=True)
    st.dataframe(data=df_percentage)    

elif graph_type == "Porcentaje Total Diferencia":
    df_diferencia_porcentaje = calcular_diferencia_porcentaje(df_generales, df_paso)
    heatmap_diferencia_total_otros(df_diferencia_porcentaje)
    st.pyplot(plt.gcf(), use_container_width=True)
    st.dataframe(data=df_diferencia_porcentaje.round(2))
