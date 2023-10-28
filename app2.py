import streamlit as st
import json
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
import geopandas as gpd
import plotly.express as px

############################################################################################
# Page config
############################################################################################

# Cargar la imagen
page_icon = Image.open("assets/elecciones.png")

st.set_page_config(
    page_title="Analisis Elecciones 2023",
    page_icon=page_icon,
    layout="wide",
)

st.title("Análisis de Elecciones Nacionales")

st.markdown("""
            [![LinkedIn Badge](https://img.shields.io/badge/marianobonelli-LinkedIn-blue)](https://www.linkedin.com/in/mariano-francisco-bonelli/) [![Twitter Badge](https://img.shields.io/badge/marianobonelli-Twitter-blue?logo=twitter)](https://twitter.com/marianobonelli)
            """)

############################################################################################
# DataFrame
############################################################################################

def cargar_datos(env, scope):
    ruta_archivo = f'assets/{env}/{env}_{scope}.json'
    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        return json.load(file)

def transformar_datos(data):
    max_level = max(item.get('level', 0) for item in data['mapa'])

    registros = []
    for item in data['mapa']:
        if item.get('level', 0) == max_level:
            for scope in item['scopes']:
                scope_id = scope['codigo']
                scope_name = scope['name'][:13]
                for partido in scope['partidos']:
                    registros.append([scope_id, scope_name, partido['name'][:13], partido['votos']])
                registros.append([scope_id, scope_name, 'BLANCOS', scope['blancos']])
    
    df = pd.DataFrame(registros, columns=['ID', 'Scope', 'Partido', 'Votos'])
    return df.pivot_table(index=['Scope', 'ID'], columns='Partido', values='Votos', aggfunc='sum', fill_value=0)

def ajustar_pasos(df):
    partidos_generales = ['BLANCOS', 'FRENTE DE IZQ', 'HACEMOS POR N', 'JUNTOS POR EL', 'LA LIBERTAD A', 'UNION POR LA ']
    partidos_a_eliminar = [partido for partido in df.columns if partido not in partidos_generales and partido != 'BLANCOS']
    
    if partidos_a_eliminar:
        df['Otros'] = df[partidos_a_eliminar].sum(axis=1)
        df = df.drop(columns=partidos_a_eliminar)
        df = df[['Otros'] + [col for col in df.columns if col != 'Otros']]
    
    return df

def generar_df(env, scope):
    data = cargar_datos(env, scope)
    df = transformar_datos(data)
    
    if env == 'PASO':
        df = ajustar_pasos(df)
        
    df = df.astype(int)
    df['TOTAL'] = df.sum(axis=1)
    df.loc[('TOTAL', 'TOTAL'), :] = df.sum()

    for columna in df.columns:
        df[columna] = pd.to_numeric(df[columna], errors='coerce').fillna(0).astype(int)

    return df

def porcentaje_total(df):
    total_general = df.at[('TOTAL', 'TOTAL'), 'TOTAL']
    df_porcentaje = (df / total_general) * 100
    df_porcentaje.at[('TOTAL', 'TOTAL'), 'TOTAL'] = 100  # El total general siempre es 100%
    return df_porcentaje

def porcentaje_por_columna(df):
    totals = df.loc[('TOTAL', 'TOTAL')]
    df_porcentaje = (df.div(totals, axis=1) * 100).fillna(0)
    df_porcentaje.loc[('TOTAL', 'TOTAL')] = 100  # El total de cada columna es 100%
    return df_porcentaje

def restar_dataframes(df1, df2):
    if 'Otros' not in df1.columns:
        df1['Otros'] = 0
    if 'Otros' not in df2.columns:
        df2['Otros'] = 0
    
    # Asegurando que ambos DataFrames tengan el mismo orden de columnas
    df2 = df2[df1.columns]
    
    df = df1 - df2
    return df

############################################################################################
# HeatMap
############################################################################################

def mostrar_heatmap(df, graph_type):
    df = df.reset_index().drop(columns='ID').set_index('Scope')  # Reset MultiIndex, remove 'ID' and then set 'Scope' as the index

    # Order the columns
    if 'Otros' in df.columns:
        column_order = ['Otros', 'BLANCOS', 'FRENTE DE IZQ', 'HACEMOS POR N', 'JUNTOS POR EL', 'LA LIBERTAD A', 'UNION POR LA ', 'TOTAL']
    else:
        column_order = ['BLANCOS', 'FRENTE DE IZQ', 'HACEMOS POR N', 'JUNTOS POR EL', 'LA LIBERTAD A', 'UNION POR LA ', 'TOTAL']

    df = df[column_order]

    # Create the heatmap
    plt.figure(figsize=(10, len(df)*0.5))  # Adjust the size of the plot as needed

    if graph_type == 'Votos':
        ax = sns.heatmap(df, annot=True, fmt=',.0f', cmap='YlGnBu', cbar=False)
    else:
        ax = sns.heatmap(df, annot=True, fmt=',.2f', cmap='YlGnBu', cbar=False)
    
    # Add title and rotate x-axis labels
    plt.xticks(rotation=45)

    # Show the heatmap
    st.pyplot(plt)

############################################################################################
# Main
############################################################################################

# Indice
with open('assets/indice.json', 'r', encoding='utf-8') as file:
    indice = json.load(file)

# Seleccionar Env, Scope y Tipo de Gráfico
env = st.selectbox("Elecciones:", ['GENERALES', 'PASO', 'DIFERENCIA'])
default_index = list(indice.keys()).index('Nacionales') # Asegúrate de que 'indice' esté definido
scope = st.selectbox("Nivel de análisis:", list(indice.keys()), index=default_index)
graph_type = st.selectbox("Tipo de gráfico:", ['Votos', 'Porcentaje Total', 'Porcentaje por Partido'])

column_order=['Otros', 'BLANCOS', 'FRENTE DE IZQ', 'HACEMOS POR N', 'JUNTOS POR EL', 'LA LIBERTAD A', 'UNION POR LA ', 'TOTAL']

tab1, tab2, = st.tabs(["HeatMap", "Tabla"])

if env != 'DIFERENCIA':
    df = generar_df(env, scope)
    altura = (35 * (len(df)) + 40)

    if graph_type == 'Votos':
        with tab1:
            mostrar_heatmap(df, graph_type)
        with tab2:
            st.dataframe(data = df, height=altura, use_container_width=True, column_order=column_order)

    if graph_type == 'Porcentaje Total':
        df = porcentaje_total(df)
        with tab1:
            mostrar_heatmap(df, graph_type)
        with tab2:
            st.dataframe(data = df.round(2), height=altura, use_container_width=True, column_order=column_order)

    if graph_type == 'Porcentaje por Partido':
        df = porcentaje_por_columna(df)
        with tab1:
            mostrar_heatmap(df, graph_type)
        with tab2:
            st.dataframe(data = df.round(2), height=altura, use_container_width=True, column_order=column_order)

if env == 'DIFERENCIA':
    df1 = generar_df('GENERALES', scope)
    df2 = generar_df('PASO', scope)

    if graph_type == 'Votos':
        df = restar_dataframes(df1, df2)
        with tab1:
            mostrar_heatmap(df, graph_type)
        with tab2:
            altura = (35 * (len(df)) + 40)
            st.dataframe(data=df.round(2), height=altura, use_container_width=True, column_order=column_order)

    if graph_type == 'Porcentaje Total':
        df1_porcentaje = porcentaje_total(df1)
        df2_porcentaje = porcentaje_total(df2)
        df = restar_dataframes(df1_porcentaje, df2_porcentaje)
        with tab1:
            mostrar_heatmap(df, graph_type)
        with tab2:
            altura = (35 * (len(df)) + 40)
            st.dataframe(data=df.round(2), height=altura, use_container_width=True, column_order=column_order)

    if graph_type == 'Porcentaje por Partido':
        df1_porcentaje = porcentaje_por_columna(df1)
        df2_porcentaje = porcentaje_por_columna(df2)
        df = restar_dataframes(df1_porcentaje, df2_porcentaje)
        with tab1:
            mostrar_heatmap(df, graph_type)
        with tab2:
            altura = (35 * (len(df)) + 40)
            st.dataframe(data=df.round(2), height=altura, use_container_width=True, column_order=column_order)


def generar_mapa(graph_type, indice, scope, df, selected_column):
    # Ruta al archivo ZIP
    zip_path = f'assets/MAPAS/{indice[scope]}.zip'

    # Verificar si el archivo ZIP existe
    if os.path.exists(zip_path):
        # Abrir el archivo ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extraer el nombre del archivo GeoJSON dentro del ZIP
            geojson_name = [name for name in zip_ref.namelist() if name.endswith('.geojson')]
            if geojson_name:
                # Leer el archivo GeoJSON
                with zip_ref.open(geojson_name[0]) as geojson_file:

                    gdf = gpd.read_file(geojson_file, encoding='utf-8')
                    gdf = gdf.to_crs("EPSG:4326")

                    # Resetear el índice de df para convertir 'Scope' e 'ID' en columnas
                    df_reset = df.reset_index()
                    # Convertir la columna 'ID' a tipo string
                    df_reset['ID'] = df_reset['ID'].astype(str)
                    # Realizar la unión
                    gdf = gdf.merge(df_reset, left_on='name', right_on='ID')

                    if graph_type != 'Votos':
                        gdf[selected_column] = gdf[selected_column].round(2)

                    fig = px.choropleth_mapbox(gdf, geojson=gdf.geometry, locations=gdf.index,
                               color=selected_column,
                               center={"lat": gdf.geometry.centroid.y.mean(), "lon": gdf.geometry.centroid.x.mean()},
                               mapbox_style="carto-positron", zoom=5,
                               color_continuous_scale='YlGnBu', opacity=0.5,
                               )
                    
                    fig.update_traces(hovertemplate=None)
                    fig.update_layout(coloraxis_showscale=False)
                    fig.update_layout(height=600)
                    
                    # Mostrar el gráfico en Streamlit
                    st.plotly_chart(fig, use_container_width=True)

# Supongamos que 'indice', 'scope', y 'df' ya están definidos

if 'Otros' in df.columns:
    total, uxp, lla, jxc, hxc, fdi, blancos, otros = st.tabs(['TOTAL', 'UNION POR LA ', 'LA LIBERTAD A', 'JUNTOS POR EL', 'HACEMOS POR N', 'FRENTE DE IZQ', 'BLANCOS', 'Otros'])
    with total:
        generar_mapa(graph_type, indice, scope, df, 'TOTAL')
    with uxp:
        generar_mapa(graph_type, indice, scope, df, 'UNION POR LA ')
    with lla:
        generar_mapa(graph_type, indice, scope, df, 'LA LIBERTAD A')
    with jxc:
        generar_mapa(graph_type, indice, scope, df, 'JUNTOS POR EL')
    with hxc:
        generar_mapa(graph_type, indice, scope, df, 'HACEMOS POR N')
    with fdi:
        generar_mapa(graph_type, indice, scope, df, 'FRENTE DE IZQ')
    with blancos:
        generar_mapa(graph_type, indice, scope, df, 'BLANCOS')
    with otros:
        generar_mapa(graph_type, indice, scope, df, 'Otros')
else:
    total, uxp, lla, jxc, hxc, fdi, blancos = st.tabs(['TOTAL', 'UNION POR LA ', 'LA LIBERTAD A', 'JUNTOS POR EL', 'HACEMOS POR N', 'FRENTE DE IZQ', 'BLANCOS'])
    with total:
        generar_mapa(graph_type, indice, scope, df, 'TOTAL')
    with uxp:
        generar_mapa(graph_type, indice, scope, df, 'UNION POR LA ')
    with lla:
        generar_mapa(graph_type, indice, scope, df, 'LA LIBERTAD A')
    with jxc:
        generar_mapa(graph_type, indice, scope, df, 'JUNTOS POR EL')
    with hxc:
        generar_mapa(graph_type, indice, scope, df, 'HACEMOS POR N')
    with fdi:
        generar_mapa(graph_type, indice, scope, df, 'FRENTE DE IZQ')
    with blancos:
        generar_mapa(graph_type, indice, scope, df, 'BLANCOS')
