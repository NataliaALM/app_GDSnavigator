import streamlit as st
import GEOparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns 
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="GDSnavigator",
    layout="wide",  # ancho completo
    initial_sidebar_state="expanded"
)

st.title("GDSnavigator")

# Disposición en dos columnas
col1, col2 = st.columns([1, 3])

# -------------------------------
# Primera columna (Colocada a la izquierda)
# -------------------------------
with col1:
    gds_id = st.text_input("Introduzca el ID del GDS que desea analizar", value="")
    if gds_id:
        st.info(f"Descargando y procesando {gds_id}... Esto puede tardar unos segundos.")
        data = GEOparse.get_GEO(geo=gds_id, destdir="./home/natalia/scripts/TFM_UEMC/")
        st.success(f"GDS {gds_id} descargado correctamente")
            # 3️⃣ Selector de variable de interés
        metadata = data.columns
        # El usuario selecciona entre las opciones propuestas (las variables clínicas del usuario)
        variable_interes = st.selectbox(
            "Selecciona la variable de interés:",
            [col for col in metadata.columns if col != "description"]
        )

# -------------------------------
# Segunda columna (Colocada a la derecha)
# -------------------------------
with col2:
    #if gds_id:
    tabs = st.tabs(["Exploración de los datos", "Expresión de los genes con mayor variabilidad", "Clustering Jerárquico", "Reducción de la dimensionalidad"])

    # Sección 1: Exploración de los datos
    with tabs[0]:

        st.subheader("Introducción")
        st.write("GDSnavigator es una aplicación para analizar bases de datos GDS que se encuentran almacenadas en el repositorio Gene Expression Omnibus (GEO). Este análisis incluye una breve exploración de los datos, el análisis de los genes con más variabilidad en los grupos de pacientes que el usuario desee, así como la búsqueda de patrones génicos aplicando el método estadístico clustering jerárquico y la técnica de Análisis de Componentes Principales (PCA).")
        st.write("En la presente sección, se muestra una breve descripción de la base de datos. Primero, se puede ver la cabecera de los datos de expresión, donde los genes siempre se representan en las filas y las muestras o individuos en las columnas. A continuación, se muestra la cabecera de los metadatos o información clínica de los pacientes. Por último, se puede observar un gráfico de sectores con las frecuencias de la variable grupo y un boxplot por muestra para comrpobar que los datos están correctamente normalizados y escalados.")
        st.write("El usuario debe introducir el GDS que desea, así como la variable que define los grupos de pacientes. LINK a la web:")
        st.markdown(
        '<a href="https://www.ncbi.nlm.nih.gov/sites/GDSbrowser?ac" target="_blank">Visita Dataset Bowser of NCBI</a>',
        unsafe_allow_html=True
        )
        # Los identificadores de los genes deben ser los gene symbols
        if gds_id:
            # Definir la matriz de expresión
            expr_matrix = data.table
            expr = expr_matrix.set_index("IDENTIFIER")
            expr = expr.drop(columns=[c for c in expr.columns if not expr[c].dtype.kind in 'fi'])
            expr = expr.groupby(expr.index).mean()
            expr_log = np.log2(expr + 1)

            def quantile_normalize(df):

                sorted_df = np.sort(df.values, axis=0)
                mean_ranks = np.mean(sorted_df, axis=1)
    
                ranks = np.apply_along_axis(lambda x: pd.Series(x).rank(method='min').values.astype(int)-1, 0, df.values)
                df_qn = pd.DataFrame(np.zeros_like(df.values), index=df.index, columns=df.columns)
    
                for j, col in enumerate(df.columns):
                    df_qn[col] = [mean_ranks[r] for r in ranks[:, j]]
        
                return df_qn

            expr_log = quantile_normalize(expr_log)

            # Mostrar la matriz de expresión
            st.subheader("Cabecera de la matriz de expresión escalada")
            st.dataframe(expr_log.head())

            # Mostrar los metadatos
            # metadata = data.columns
            st.subheader("Información clínica de los pacientes")
            st.dataframe(metadata.head())

            if variable_interes in metadata.columns:
                # Definir la variable grupos, que debe ser una columna de la tabla de los metadatos
                grupos = metadata[variable_interes]

                # Gráfico de sectores que muestre las frecuencias de la variable grupo
                st.subheader(f"Frecuencias de la variable grupo {variable_interes}")
                freq = grupos.value_counts()
                plt.figure(figsize=(6,6))
                plt.pie(freq, labels=freq.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
                plt.title(f"Frecuencia de {variable_interes}")
                st.pyplot(plt.gcf())
                plt.close()

                # Convertir a formato largo para el gráfico
                expr_long = expr_log.T.reset_index().melt(id_vars='index', var_name='Gene', value_name='Expression')
                expr_long = expr_long.rename(columns={'index': 'Sample'})
                expr_long['Group'] = expr_long['Sample'].map(metadata[variable_interes])

                # Boxplot por muestra coloreado por la variable grupo
                plt.figure(figsize=(12,6))
                sns.boxplot(x='Sample', y='Expression', hue='Group', data=expr_long, dodge=False, palette="Set2")
                plt.xticks(rotation=90)
                plt.ylabel("Expresión (log2)")
                plt.title(f"Distribución de la expresión génica por muestra ({variable_interes})")
                plt.legend(title=variable_interes, bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(plt.gcf())
                plt.close()

        with tabs[1]:
            st.header("Expresión de los genes con mayor variabilidad en los grupos de pacientes")
            st.write("Uno claro factor observable a partir de datos transcriptómicos para encontrar diferencias entre pacientes e individuos sanos es estudiar la expresión génica de estos. Por ello, \nen esta sección exploraremos la expresión de los 30 genes con más varianza a través de las categoŕias definidas en la variable grupo.")

            # Heatmap solo con top 30 genes más variables (para poder visualizarlo en el heatmap)
            top_genes = expr_log.var(axis=1).sort_values(ascending=False).head(50).index
            expr_top = expr_log.loc[top_genes]
            group_colors = {g: c for g, c in zip(metadata[variable_interes].unique(), sns.color_palette("Set2", len(metadata[variable_interes].unique())))}
            row_colors = metadata[variable_interes].map(group_colors)

            st.subheader("Heatmap de la expresión génica con los top 30 genes más variables")
            heat50 = sns.clustermap(
                expr_top.T,  # muestras x genes
                row_cluster=True,
                col_cluster=True,
                row_colors=row_colors,
                figsize=(12,8),
                cmap="vlag",
                standard_scale=0
            )
            # Añadir leyenda manualmente
            handles = [Patch(facecolor=color, label=label) for label, color in group_colors.items()]
            heat50.ax_row_dendrogram.legend(
            handles=handles,
            title=variable_interes,
            bbox_to_anchor=(1,1),
            bbox_transform=plt.gcf().transFigure,
)
            st.pyplot(plt.gcf())
            plt.close()

        with tabs[2]:
            st.header("Clustering jerárquico")
            st.write("Primero, buscaremos patrones génicos en los individuos utilizando un clustering jerárquico. Este método pertenece a la familia de los no supervisados, y trata de agrupar grupos a distintos niveles con una estructura de árbol (dendrograma)")
            st.write("A continuación, se muestra el diagrama en forma de árbol o dendrograma, donde se pueden ver las agrupaciones de los individuos. Aquellos que estén conectados por una ramificación más próxima, tendrán mayor similitud en base a la expresión génica.")
            st.write("NOTA: Las etiquetas de las muestras o individuos aparecen coloreadas por la variable grupos.")

            data_matrix = expr_log.T # muestras x genes

        # Diagrama en forma de árbol o dendrograma para un clustering jerárquico 
            Z = linkage(data_matrix, method='ward')  # método Ward (muy utilizado)
            plt.figure(figsize=(12,6))
            dendro = dendrogram(
                Z, 
                labels=data_matrix.index,  # mostrar las etiquetas de las muestras
                leaf_rotation=90,
                leaf_font_size=10,
                color_threshold=0
            )  

            # Colorear las hojas según la variable grupo
            ax = plt.gca()
            xlbls = ax.get_xmajorticklabels()
            for lbl in xlbls:
                sample = lbl.get_text()
                lbl.set_color(group_colors[metadata.loc[sample, variable_interes]])

            # Añadir leyenda manualmente
            handles_clus = [Patch(facecolor=color, label=label) for label, color in group_colors.items()]
            plt.legend(
            handles=handles_clus,
            title=variable_interes,
            bbox_to_anchor=(1.02, 1),  # fuera del gráfico a la derecha
            loc='upper left',
            borderaxespad=0
            )
            plt.title("Clustering jerárquico de muestras")
            st.pyplot(plt.gcf())
            plt.close()

            # Selección óptima de número de clústers utilizando el coeficiente de Silhouette
            max_clusters = min(10, data_matrix.shape[0] - 1)  # evitar más clusters que muestras
            silhouette_avgs = []
            cluster_range = range(2, max_clusters + 1)

            for k in cluster_range:
                cluster_labels = fcluster(Z, k, criterion="maxclust")
                score = silhouette_score(data_matrix, cluster_labels)
                silhouette_avgs.append(score)

            best_k = cluster_range[np.argmax(silhouette_avgs)]
        
            st.write("Aunque en el dedrograma anterior podemos encontrar distinto número de clusters en función de la altura del árbol, en la parte inferior se muestra el número óptimo de grupos en base a nuestros datos. Dicho valor se ha calculado utilizando el coeficiente de Silhouette, que mide la distancia intragrupal e intergrupal.El número óptimo de clusters será aquel que tenga el mayor valor promedio de esta métrica. ")
            st.write(f"🌟 El número óptimo de clústers según Silhouette en nuestros datos es **{best_k}**")

            # Mostrar el gráfico del coeficiente de Silhouette
            plt.figure(figsize=(6,4))
            plt.plot(cluster_range, silhouette_avgs, marker="o")
            plt.xlabel("Número de clústers")
            plt.ylabel("Coeficiente promedio de Silhouette")
            plt.title("Selección de número de clústers (Silhouette)")
            st.pyplot(plt.gcf())
            plt.close()

            optimal_clusters = fcluster(Z, best_k, criterion='maxclust')
 
            # Crear una tabla que contenta la asignación para cada muestra 
            cluster_assignments = pd.DataFrame({
            "Sample": data_matrix.index,
            "Cluster": optimal_clusters
            })

            st.write("En la siguiente tabla se pueden ver las asignaciones de los individuos a los grupos, siendo el número de grupos aquel calculado por Silhouette:")
            st.dataframe(cluster_assignments) 

        with tabs[3]:
            st.header("Análisis de Componentes Principales (PCA)")
            st.write("Como una segunda forma de encontrar patrones, realizaremos una reducción de dimensiones aplicando un Análisis de Componentes Principales (PCA). Esta técnica estadística trata de reducir la dimensionalidad construyendo variables latentes no correlacionadas entre sí que permitan maximizar la varianza a partir de variables existentes correlacionadas.")

            # Asegurarse de que el usuario indicó la variable grupo
            if variable_interes not in metadata.columns:
                st.warning(f"La variable '{variable_interes}' no existe en los metadatos.")
            else:
                grupos = metadata[variable_interes]
                categorias = grupos.unique()
                paleta = sns.color_palette("Set2", len(categorias))
                color_dict = dict(zip(categorias, paleta))
                colores = [color_dict[gr] for gr in grupos]

                # PCA
                pca = PCA(n_components=2)
                pcs = pca.fit_transform(expr_log.T)  # muestras en filas

                # Crear un DataFrame para el gráfico
                pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
                pca_df['Group'] = grupos.values
                pca_df['Sample'] = expr_log.columns

                st.write("A continuación, se muestra el PC scatter plot. Este gráfico representa los individuos  en el espacio de las componentes principales. Si dos muestras están próximas en el espacio de las componentes, entonces dichos individuos tienen perfiles génicos similares. En nuestro caso, hemos utilizado 2 componentes debido a que es el número de componentes que nos permite visualizar en un gráfico.")

                # Plot
                plt.figure(figsize=(10,7))
                sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Group', palette=color_dict, s=100)
                for i in range(pca_df.shape[0]):
                    plt.text(pca_df.PC1[i]+0.1, pca_df.PC2[i]+0.1, pca_df.Sample[i], fontsize=8)
                plt.title("PCA de las muestras")
                plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
                plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
                plt.legend(title=variable_interes, bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(plt.gcf())
                plt.close()

                st.write("Como con las nuevas variables o componentes se desea maximizar la varianza, veamos si con 2 componentes es suficiente para sus datos. En teoría, las componentes deberían explicar al menos el 80% de la varianza para ser considerado como un número suficiente de variables latentes. El porcentaje de varianza explicada para cada una de las dos componentes se muestra a continuación:")

                # Mostrar el valor de la varianza explicada de cada componente en forma de gráfico de barras
                expl_var = pca.explained_variance_ratio_ * 100
                plt.figure(figsize=(8,4))
                plt.bar(range(1, len(expl_var)+1), expl_var)
                plt.xlabel('Componentes principales')
                plt.ylabel('% Varianza explicada')
                plt.title('Scree plot')
                st.pyplot(plt.gcf())
                plt.close()

