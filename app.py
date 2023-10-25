import streamlit as st
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def agregar_total(df):
    df['TOTAL'] = df.sum(axis=1)
    df.loc['TOTAL'] = df.sum(axis=0)
    return df

def mostrar_heatmap_votos(df):
    df = agregar_total(df.copy())
    fmt = ','
    vmax = np.nanmax(df.drop('TOTAL', errors='ignore').values)  # Valor máximo sin considerar totales
    figsize = (max(10, len(df.columns)*1), max(8, len(df.index)*0.3))

    plt.figure(figsize=figsize)
    sns.heatmap(df, cmap="YlGnBu", annot=True, fmt=fmt, linewidths=.5, vmax=vmax, cbar=False, alpha=0.7)
    plt.title("Votos")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def mostrar_heatmap_porcentaje_total(df):
    total_votos = df.values.sum()
    df_percentage = (df / total_votos) * 100

    # Calcular columna TOTAL
    df_percentage['TOTAL'] = (df.sum(axis=1) / total_votos) * 100
    df_percentage.loc['TOTAL'] = df_percentage.sum(axis=0)

    plt.figure(figsize=(max(10, len(df.columns)*1), max(8, len(df.index)*0.3)))
    sns.heatmap(df_percentage, cmap="YlGnBu", annot=True, fmt='.2f', linewidths=.5, vmax=50, cbar=False, alpha=0.7)
    plt.title("Porcentaje sobre el total")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return df_percentage

def mostrar_heatmap_porcentaje_partido(df, total_col):
    total_by_party = df.sum(axis=0)
    df_percentage = (df.divide(total_by_party, axis=1)) * 100

    # Copiar la columna TOTAL desde el DataFrame de Porcentaje sobre el total
    df_percentage['TOTAL'] = total_col
    df_percentage.loc['TOTAL'] = df_percentage.sum(axis=0)

    plt.figure(figsize=(max(10, len(df.columns)*1), max(8, len(df.index)*0.3)))
    sns.heatmap(df_percentage, cmap="YlGnBu", annot=True, fmt='.2f', linewidths=.5, vmax=50, cbar=False, alpha=0.7)
    plt.title("Porcentaje por partido")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def calcular_porcentaje_total(df):
    total_votos = df.values.sum()
    df_percentage = (df / total_votos) * 100
    df_percentage['TOTAL'] = (df.sum(axis=1) / total_votos) * 100
    df_percentage.loc['TOTAL'] = df_percentage.sum(axis=0)
    return df_percentage

def main():
    st.title("Análisis de Elecciones Nacionales")

    # Seleccionar Env, Scope y Tipo de Gráfico
    env = st.selectbox("Elecciones:", ['GENERALES', 'PASO'])
    default_index = list(indice.keys()).index('Nacionales')
    scope = st.selectbox("Nivel de análisis:", list(indice.keys()), index=default_index)
    graph_type = st.selectbox("Tipo de gráfico:", ['Votos', 'Porcentaje Total', 'Porcentaje por Partido'])

    # Generar el DataFrame
    df = generar_df(env, scope)
    if graph_type == "Votos":
        mostrar_heatmap_votos(df)
        st.pyplot(plt.gcf(), use_container_width=True)  # Mostrar el gráfico en Streamlit
    elif graph_type == "Porcentaje Total":
        df_percentage_total = calcular_porcentaje_total(df)
        plt.figure(figsize=(max(10, len(df.columns)*1), max(8, len(df.index)*0.3)))
        sns.heatmap(df_percentage_total, cmap="YlGnBu", annot=True, fmt='.2f', linewidths=.5, vmax=50, cbar=False, alpha=0.7)
        plt.title("Porcentaje sobre el total")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt.gcf(), use_container_width=True)  # Mostrar el gráfico en Streamlit
    elif graph_type == "Porcentaje por Partido":
        df_percentage_total = calcular_porcentaje_total(df)
        mostrar_heatmap_porcentaje_partido(df, df_percentage_total['TOTAL'])
        st.pyplot(plt.gcf(), use_container_width=True)  # Mostrar el gráfico en Streamlit

if __name__ == "__main__":
    main()