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

def mostrar_heatmap_votos(df):
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

def main():
    st.title("Análisis de Elecciones Nacionales")
    st.markdown("""
                [![LinkedIn Badge](https://img.shields.io/badge/marianobonelli-LinkedIn-blue)](https://www.linkedin.com/in/mariano-francisco-bonelli/) [![Twitter Badge](https://img.shields.io/badge/marianobonelli-Twitter-blue?logo=twitter)](https://twitter.com/marianobonelli)
                """)

    # Seleccionar Env, Scope y Tipo de Gráfico
    env = st.selectbox("Elecciones:", ['GENERALES', 'PASO', 'DIFERENCIA'])
    default_index = list(indice.keys()).index('Nacionales') # Asegúrate de que 'indice' esté definido
    scope = st.selectbox("Nivel de análisis:", list(indice.keys()), index=default_index)

    if env == 'DIFERENCIA':
        graph_type = st.selectbox("Tipo de gráfico:", ['Votos'])  # Solo permitir "Votos" si la selección es "DIFERENCIA"
    else:
        graph_type = st.selectbox("Tipo de gráfico:", ['Votos', 'Porcentaje Total', 'Porcentaje por Partido'])

    # Lista de partidos presentes en las elecciones generales
    partidos_generales = ['BLANCOS', 'FRENTE DE IZQ', 'HACEMOS POR N', 'JUNTOS POR EL', 'LA LIBERTAD A', 'UNION POR LA ']

    if env != 'DIFERENCIA':
        df = generar_df(env, scope)
        if env == 'PASO':
            df = simplificar_df(df, partidos_generales)
            if 'Otros' not in df.columns:
                df['Otros'] = 0
    else:
        df_generales = generar_df('GENERALES', scope)
        df_paso = generar_df('PASO', scope)
        df_paso = simplificar_df(df_paso, partidos_generales)
        if 'Otros' not in df_generales.columns:
            df_generales['Otros'] = 0
        df = df_generales - df_paso
        df = reordenar_columnas(df)  # Reordenar las columnas
    
    if graph_type == "Votos":
        mostrar_heatmap_votos(df)
        st.pyplot(plt.gcf(), use_container_width=True)
    elif graph_type == "Porcentaje Total":
        df_percentage_total = calcular_porcentaje_total(df)
        plt.figure(figsize=(max(10, len(df.columns)*1), max(8, len(df.index)*0.3)))
        sns.heatmap(df_percentage_total, cmap="YlGnBu", annot=True, fmt='.2f', linewidths=.5, vmax=50, cbar=False)
        plt.title("Porcentaje sobre el total")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt.gcf(), use_container_width=True)
    elif graph_type == "Porcentaje por Partido":
        df_percentage_total = calcular_porcentaje_total(df)
        mostrar_heatmap_porcentaje_partido(df, df_percentage_total['TOTAL'])
        st.pyplot(plt.gcf(), use_container_width=True)

if __name__ == "__main__":
    main()