## Import das bibliotecas

import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import joblib
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import export_graphviz
import graphviz


# Filter and Clean

def filter_columns(df, filters: list): 
    selected_columns = [True] * len(df.columns)  
    for index, column in enumerate(df.columns):
        if any(filter in column for filter in filters): selected_columns[index] = False
    return df[df.columns[selected_columns]]

# função de filtro inversa - selecionar colunas na saida
def filter_columns_in(df, filters: list):  
    selected_columns = [False] * len(df.columns) 
    for index, column in enumerate(df.columns):
        if any(filter in column for filter in filters): selected_columns[index] = True
    return df[df.columns[selected_columns]]

def cleaning_dataset(df):
  _df = df.dropna(subset=df.columns.difference(['NOME']), how='all')
  _df = _df[~_df.isna().all(axis=1)]
  return _df

# Estilo dos graficos do Seaborn

ax = sns.set_style('whitegrid')
ax = sns.set_palette('pastel')

# import da base

st.set_page_config(layout='wide')

st.title('Datathon - Tech 5 - Passos Magicos')

file_path = 'PEDE_PASSOS_DATASET_FIAP.csv'
pd.set_option('display.max_columns', None)
df = pd.read_csv(file_path, delimiter=';')

aba1, aba2, aba3, aba4 = st.tabs(['EDA', 'Clusterização ' , 'Features Importances','Simulador INDE'])

#st.write(df.head(1))

with aba1:

    coluna1, = st.columns(1)
    

    with coluna1:
        st.subheader('Evolução temporal dos principais indicadores')
    
        ''' 
        O estudo a seguir foi motivado pelos seguintes fatores:

    * Objetivos de entrega da atividade (relatório de análise exploratória e/ou modelo preditivo)
    * Insigths obtidos por meio da entrevista com o Sr Dimitri CEO da Passos Mágicos para idealização do projeto e também com os analistas de dados na live realizada em 10/07
    * Análise dos dados disponibilizados para essa atividade
    * Insights e sugestões obtidas com o nosso orientador do projeto (Profº Willian) para direcionamento no desenvolvimento dessa atividade

    Dado a inconsistência de algumas linhas e colunas não disponiveis como os dados dos alunos ao longo dos anos de 2020,  2021 e 2022, além do acréscimo de novas colunas ao longo do tempo e alta dimensionalidade dos dados, 
    aplicamos técnicas distintas para melhor entendimentos.
    Nessa análise buscamos trabalhar com os principais indicadores que de acordo com a coleta das informações junto à equipe da Passos Mágicos, são os indicadores de maior diferencial em seu programa de 
    aceleração do conhecimento'''

    ## Graf 1 - Evolução INDE

    st.subheader('INDE - Indice de Desenvolvimento Educacional')

    '''De acordo com as informações obtidas em entrevista com o time de dados e ex-alunos da Passos Mágicos, esse é um dos indicadores mais importantes na composição do INDE, pois é um indicador que mostra o quanto o aluno está enxergando sua evolução dentro da trilha de aprendizado em cada fase. 
    Esse indicador pode retratar sua auto estima e ser determinante para o ponto de virada do aluno.'''

    df_inde = df[['INDE_2020', 'INDE_2021', 'INDE_2022']]
    df_inde = df_inde.apply(pd.to_numeric, errors='coerce')
    df_inde_mean = df_inde.mean(axis=0, skipna=True) 

    fig_inde = px.line(df_inde_mean, x=df_inde_mean.index, y=df_inde_mean.values, template='plotly_white')

    fig_inde.update_layout(
        title='Evolução do INDE',
        title_font_color='gray',
        yaxis_title='Média INDE',
        yaxis_title_font=dict(color='gray', size=14),
        xaxis_title='',
        xaxis_title_font=dict(color='gray', size=14),
        yaxis2=dict(
            title='',
            title_font=dict(color='gray', size=14),
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis=dict(showgrid=False, tickfont=dict(size=16)),
        yaxis=dict(showgrid=False, tickfont=dict(size=16)),
        plot_bgcolor='white',
        barmode='overlay',  # Define o modo de sobreposição para as barras
        bargap=0,  # Elimina o espaço entre as barras do mesmo grupo
        bargroupgap=0 ,
        xaxis_showline=False  # Elimina o espaço entre os grupos de barras
    )

    fig_inde.update_traces(
        marker=dict(
            color='white',
        ),
        selector=dict(type='bar')
    )

    fig_inde.update_traces(
        marker=dict(
            color=['royalblue' if i < 3 else 'royalblue' for i in range(len(df_inde_mean))],
        ),
        selector=dict(type='bar')
    )

    st.plotly_chart(fig_inde, use_container_width= True)

    ## Bolsitas 

    st.subheader('Bolsitas e Indicados à Bolsa')

    '''
    De acordo com a equipe da Passos Mágicos, um dos grandes objetivos do programa de ensino é a bolsa de estudos. A base apresenta esse dado com a relação de bolsitas ou indicados à bolsa no ano de 2022. Por meio do ano de ingresso, 
    fizemos uma análise do tempo de ingresso desses alunos, para termos uma idéia de temporalidade, uma vez que não temos uma série temporal tão evidente
    '''
    df_bolsa = filter_columns_in(df, ['NOME', 'BOLSA', 'BOLSISTA_','ANO_INGRESSO_2022'])
    df_bolsa = cleaning_dataset(df_bolsa)

    df_bolsistas = df_bolsa.groupby('ANO_INGRESSO_2022')['BOLSISTA_2022'].value_counts().reset_index(name='qtde') 
    df_bolsistas = df_bolsistas[df_bolsistas['BOLSISTA_2022'] == 'Sim']
    df_bolsistas['ANO_INGRESSO_2022'] = pd.to_datetime(df_bolsistas['ANO_INGRESSO_2022'], format='%Y', errors='coerce').dt.date

    fig_bolsa = px.bar(df_bolsistas, x='ANO_INGRESSO_2022', y='qtde',template='plotly_white')

    fig_bolsa.update_layout(
        title='Alunos bolsistas por ano ingresso',
        title_font_color='gray',
        yaxis_title='qtde',
        xaxis_title='',
        xaxis_title_font=dict(color='gray', size=14),
        yaxis_title_font=dict(color='gray',size=14),
        yaxis2=dict(
            title='',
            title_font=dict(color='gray'),
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white'
    )

    fig_bolsa.update_traces(
        marker=dict(
            color='gray',
        ),
        selector=dict(type='bar')
    )

    fig_bolsa.update_traces(
        marker=dict(
            color=['gray' if i < 5 else 'royalblue' for i in range(len(df_bolsistas))],
        ),
        selector=dict(type='bar')
    )

    st.plotly_chart(fig_bolsa, use_container_width= True)

    ## Indicados a bolsa

    df_indicados_bolsa = df_bolsa.groupby('ANO_INGRESSO_2022')['INDICADO_BOLSA_2022'].value_counts().reset_index(name='qtde') 
    df_indicados_bolsa = df_indicados_bolsa[df_indicados_bolsa['INDICADO_BOLSA_2022'] == 'Sim']
    df_indicados_bolsa['ANO_INGRESSO_2022'] = pd.to_datetime(df_indicados_bolsa['ANO_INGRESSO_2022'], format='%Y', errors='coerce').dt.date

    fig_indicado_bolsa = px.bar(df_indicados_bolsa, x='ANO_INGRESSO_2022', y='qtde',template='plotly_white')

    fig_indicado_bolsa.update_layout(
        title='Alunos indicados a bolsa por ano ingresso',
        title_font_color='gray',
        yaxis_title='qtde',
        xaxis_title='',
        xaxis_title_font=dict(color='gray', size=14),
        yaxis_title_font=dict(color='gray', size=14),
        yaxis2=dict(
            title='',
            title_font=dict(color='gray'),
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white'
    )

    fig_indicado_bolsa.update_traces(
        marker=dict(
            color='gray',
        ),
        selector=dict(type='bar')
    )

    fig_indicado_bolsa.update_traces(
        marker=dict(
            color=['gray' if i < 6 else 'royalblue' for i in range(len(df_indicados_bolsa))],
        ),
        selector=dict(type='bar')
    )



    ## Pedras
    st.subheader('Pedras')

    '''
    De acordo com os entendimento realizado com a equipe da Passos Mágicos, a classificação anual do indicador Pedra 
    (indicador com range de valores categóricos na base) é o trabalho de avaliação final do aluno no período do ano, 
    no qual é medido sua avaliação por meio do INDE (I ndice do Desenvolvimento Educacional). 
    A documentação nos mostra as faixas de classificação do INDE conforme range abaixo:

    * Quartzo : 2,405 a 5,506
    * Agata : 5,506 a 6,868
    * Ametista : 6,868 a 8,230
    * Topazio : 8,230 a 9,294
    '''
    # Transformar os dados para contagem de pedras por ano
    df_pedras = df.melt(id_vars=['NOME'], value_vars=['PEDRA_2020', 'PEDRA_2021', 'PEDRA_2022'],
                            var_name='Ano', value_name='Pedra')

    # Remover entradas nulas
    df_pedras.replace({'#NULO!': np.nan, 'D9891/2A': np.nan}, inplace=True)
    df_pedras = df_pedras.dropna()

    # Contar a quantidade de cada tipo de pedra por ano
    df_contagem_pedras = df_pedras.groupby(['Ano', 'Pedra']).size().reset_index(name='Quantidade')

    fig_pedras = px.histogram(df_contagem_pedras, x='Ano', y='Quantidade',color='Pedra', template='plotly_white')

    st.plotly_chart(
    fig_pedras.update_layout(
        title='Status Desempenho - classificação Pedras',
        title_font_color='gray',
        yaxis_title='qtde',
        xaxis_title_font=dict(color='white'),
        yaxis_title_font=dict(color='gray'),
        yaxis2=dict(
            title='',
            title_font=dict(color='gray'),
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white'
    )
    )

    '''
    O gráfico nos mostra uma oscilação na distribuição de alunos classificados / premiados na contagem geral em 2021, contudo pode-se destacar pontos positivos 
    na recuperação do quantitativo, dado uma maior distribuição de alunos nas faixas superiores aos niveis observados em 2020.
    Abaixo podemos visualizar mais nitidamente esse efeito:'''

    fig_pedras = px.line(df_contagem_pedras, x='Ano', y='Quantidade',color='Pedra', template='plotly_white')

    st.plotly_chart(

    fig_pedras.update_layout(
        title='Status Desempenho - classificação Pedras',
        title_font_color='gray',
        yaxis_title='qtde',
        xaxis_title_font=dict(color='white'),
        yaxis_title_font=dict(color='gray'),
        yaxis2=dict(
            title='',
            title_font=dict(color='gray'),
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white'
    )
    )
   
    ''' * Agata - faixa intermediária de classificação  que parte de 5,5 até 6,8 e se sobressai sobre o Quartzo (linha de antecessor e de menor faixa do indicador)'''
    #st.image(image='img/agata.png', caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
    #st.text('imagem obtida em https://pt.wikipedia.org')
    '''* Amentista - faixa de classificação de pontuação acima de 6,8 até 8,2 que precede à pedra Topazio (maiores classificação) e apresentou uma tendencia de crescimento importante na curva'''
    #st.image(image='img/ametista.png', caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
    #st.text('imagem obtida em https://super.abril.com.br/coluna/deriva-continental/rio-grande-do-sul-e-o-maior-produtor-de-ametistas-do-mundo')
    '''* Topázio - faixa top de pontuação do INDE que apresenta uma tendência de crescimento e visualmente 'cola' na faixa mais baixa (Quartzo)'''
    #st.image(image='img/topazio_azul.png', caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
    #st.text('imagem obtida em https://pt.geologyscience.com/minerals/')
    #caixa de multiseleção
    #
    '''
    '''
    cx_mult = st.multiselect(
        'Monte um filtro personalizado com esse dataset',
        df.columns
    )
    st.dataframe(df[cx_mult])
   
###### Aba 2
    with aba2:

        coluna1, = st.columns(1)
    

        with coluna1:
            st.subheader('Clusterização')
        
            ''' 
            A análise exploratória dos dados continua com a clusterização. A clusterização é a tarefa de identificar instâncias semelhantes e atribuí-las a clusters ou seja, grupos de instâncias semelhantes (Géron, 2021).
            Utilizamos o K-means, um algoritmo não supervisonado, para criarmos um cluster de alunos com base nos indicadores principais que compõe o INDE para tentarmos observar melhor o comportamento desses índices ao longo do tempo, 
            dado que nossa base não possui uma série temporal bem definida, bem como temos dados ausentes no histórico de cada aluno, 
            o que nos desafia para uma análise mais assertiva.
            
            A escolha do K-means deve-se pelo fato de ser um algoritmo rápido e eficaz, além de possuir uma caracteristica de convergência em etapas finitas (sem oscilar indefinidamente) já que a
            a distância quadrada média entre as instâncias e o centróide mais próximo só diminui a cada etapa de aprendizado. Seu uso está presente em diversas áreas tais
            como segmentação de clientes, mecanismos de buscas, segmentação de imagens, detecção de anomalias entre outros.

            Os eixos dos gráficos abaixo estão representados da seguinte forma:
            
            * x = dados do ano de 2020
            * y = dados do ano de 2021
            * z = dados do ano de 2022
            
            '''
            # dataframe do EDA realizado com o K-Means 
            file_path_inde = 'bases/df_inde_aluno.csv'
            pd.set_option('display.max_columns', None)
            df_inde_aluno = pd.read_csv(file_path_inde, delimiter=';')

            st.subheader('INDE')

            '''Iniciamos com o INDE'''

            # Criar figura 3D com Plotly
            fig_kmeans_inde = go.Figure(data=[go.Scatter3d(
                x=df_inde_aluno['INDE_2020'],
                y=df_inde_aluno['INDE_2021'],
                z=df_inde_aluno['INDE_2022'],
                mode='markers',
                marker=dict(color=df_inde_aluno['cluster'], colorscale='Viridis', size=14, opacity=0.7, colorbar=dict(title='Cluster', outlinecolor='rgba(0,0,0,0)'))
            )])

            # Configurações de layout
            fig_kmeans_inde.update_layout(
                title='Clusters dos Alunos (K-Means) - INDE',
                scene=dict(
                    xaxis_title='INDE 2020',
                    yaxis_title='INDE 2021',
                    zaxis_title='INDE 2022',
                ),
                width=800,  # largura da figura em pixels
                height=800,  # altura da figura em pixels
            )
            st.plotly_chart(fig_kmeans_inde,use_container_width= True)

            '''
           Observando a clusterização por meio do indicador INDE, é possível destacarmos algumas informações importantes, especialmente quanto à evolução dos grupos a partir de 2020:
            - Cluster 0 - possui 69 alunos, média decrescente ao longo dos 3 anos entre 7,1 em 2020, 6,4 em 2021 e 5,9 em 2022  (nota minima 3 e máxima 7,5 nesse ultimo ano)
            - Cluster 1 - grupo de 78 alunos com a média na casa de 8 pontos, uma estabilidade nessa faixa observada pela mediana acima de 8, com uma pequena oscilação para baixo na média ao longo do período ( 8,5em 2020 para 8,1 em 2022). Não foram observados nesse grupos minimos e máximos no limite (notas zero ou 10), porém o mínimo global ficou em 6,9 em 2022 e máximo em 9,4 (todos anos acima de 9)
            - Cluster 2 - grupo de 40 alunos com média entre 6,2 e 5,7, com médiana de 5,7 em 2022, onde temos o cluster com menores médias, especialmente em 2021 (4,3 com máximo de 5,7)
            - Cluster 3 - esse grupo possui 73 alunos com média performance descrescente ao longo do tempo (média 2020 = 7,2, 2021 = 6,5 e 2022 = 5,9), sendo o 2º grupo de menor performance, atrás do cluster 0. O que os diferencia é que nesse grupo há uma tendência de queda nos 3 anos observados. Já no cluster 0, temos uma baixa média, porém com uma queda e ligeira recuperação no indicador mais recente (2022)

            Nota-se também de forma geral um desvio padrão abaixo de 1 em todos os clusters
           
            '''

        # indicador - Auto Avaliação
            st.subheader('IAA - Indicador de Auto Avaliação')
            '''
            De acordo com as informações obtidas em entrevista com o time de dados e ex-alunos da Passos Mágicos, esse é um dos indicadores mais importantes na composição do INDE, 
            pois é um indicador que mostra o quanto o aluno está enxergando sua evolução dentro da trilha de aprendizado em cada fase. 
            Esse indicador pode retratar sua auto estima e ser determinante para o ponto de virada do aluno
            '''
            file_path_iaa = 'bases/df_iaa.csv'
            pd.set_option('display.max_columns', None)
            df_iaa = pd.read_csv(file_path_iaa, delimiter=';')

            # Criar figura 3D com Plotly
            fig_kmeans_iaa = go.Figure(data=[go.Scatter3d(
                x=df_iaa['IAA_2020'],
                y=df_iaa['IAA_2021'],
                z=df_iaa['IAA_2022'],
                mode='markers',
                marker=dict(color=df_iaa['cluster'], colorscale='Viridis', size=14, opacity=0.7, colorbar=dict(title='Cluster', outlinecolor='rgba(0,0,0,0)'))
            )])

            # Configurações de layout
            fig_kmeans_iaa.update_layout(
                title='Clusters dos Alunos (K-Means) - IAA',
                scene=dict(
                    xaxis_title='IAA 2020',
                    yaxis_title='IAA 2021',
                    zaxis_title='IAA 2022',
                ),
                width=800,  # largura da figura em pixels
                height=800,  # altura da figura em pixels
            )
        

            st.plotly_chart(fig_kmeans_iaa, use_container_width= True)

        '''
    - Cluster 0 - possui 190 alunos com uma média entre 9,1 e 9,0 (mediana também nessa faixa) entre os anos de 2020 e 2022 (pequena oscilação de queda). Esse cluster obteve uma nota minima de 7,0
    - Cluster 1 - possui 17 alunos com uma média maior em 2020 (7,4) e 2022 (9,1) e com alunos com queda ou indicador zerado em 2021 (0,2)
    - Cluster 2 - possui 18 alunos com média alta em 2020 e 2021 (8,7 e 8,0) e em seguida aparecem zerados ou baixa avaliação em 2022 (média de 0,3 nesse ano com máximo de 3,4)
    - Cluster 3 - possui 89 alunos com médias entre 8 e 7,5 (decaindo gradativamente) nos 3 anos observados, sendo o segundo grupo de melhor performance nesse indicador
        '''
        st.subheader('IPV - Indicador Ponto de Virada')

        file_path_ipv = 'bases/df_ipv.csv'
        pd.set_option('display.max_columns', None)
        df_ipv = pd.read_csv(file_path_ipv, delimiter=';')

         # Criar figura 3D com Plotly
        fig_kmeans_ipv = go.Figure(data=[go.Scatter3d(
            x=df_ipv['IPV_2020'],
            y=df_ipv['IPV_2021'],
            z=df_ipv['IPV_2022'],
            mode='markers',
            marker=dict(color=df_ipv['cluster'], colorscale='Viridis', size=14, opacity=0.7, colorbar=dict(title='Cluster', outlinecolor='rgba(0,0,0,0)'))
        )])

        # Configurações de layout
        fig_kmeans_ipv.update_layout(
            title='Clusters dos Alunos (K-Means) - IPV',
            scene=dict(
                xaxis_title='IPV 2020',
                yaxis_title='IPV 2021',
                zaxis_title='IPV 2022',
            ),
            width=800,  # largura da figura em pixels
            height=800,  # altura da figura em pixels
        )
        '''
        Seguimos na análise da clusterização pelo indicador de ponto de virada também utilizando a premissa de tentarmos separar em 4 grupos, nos quais observamos:
        '''
        st.plotly_chart(fig_kmeans_ipv, use_container_width= True)

        '''
        - Cluster 0 - compõe um cluster de 48 alunos com média entre 7,2 em 2020, 4,7 em 2021 e 6,2 em 2022, com uma queda e ligeira evolução nesse indicador, sendo um cluster de baixa mediana (2022 = 6,3)
        - Cluster 1 - esse cluster possui 116 alunos e apresenta médias entre 8,3 em 2020, 8,8 em 2021 e uma ligeira queda em 2022 (8,3 pontos). A mediana oscila da mesma forma(maior pico em 2021)
        - Cluster 2 - esse cluster contém 20 alunos e apresenta uma média menor em relação aos 2 grupos anteriores (4,9 em 2020, 6,4 em 2021 e 6,9 em 2022) vindo de uma baixa média e recuperação em média, porém em oscilação de piora para min e max em 2022 
        - Cluster 3 - esse cluster também com 20 alunos é caracterizado por uma média de 7,6 em 2020 e 2021, queda de quase um ponto em 2022 para 6,9 em 2022. Seu min teve o pior indice nesse ultimo ano (3,7) e pior máximo (8,8) 
        oscilando para cima no desvio padrão (grupo com performance também em destaque pela queda observada)
        '''

        st.subheader('IEG - Indicador de Engajamento')

        file_path_ieg = 'bases/df_ieg.csv'
        pd.set_option('display.max_columns', None)
        df_ieg = pd.read_csv(file_path_ieg, delimiter=';')

        kmeans_ieg = KMeans(n_clusters=4, random_state=42)
        kmeans_ieg.fit(df_ieg[['IEG_2020', 'IEG_2021', 'IEG_2022']])
        df_ieg['cluster'] = kmeans_ieg.labels_

         # Criar figura 3D com Plotly
        fig_kmeans_ieg = go.Figure(data=[go.Scatter3d(
            x=df_ieg['IEG_2020'],
            y=df_ieg['IEG_2021'],
            z=df_ieg['IEG_2022'],
            mode='markers',
            marker=dict(color=df_ieg['cluster'], colorscale='Viridis', size=14, opacity=0.7, colorbar=dict(title='Cluster', outlinecolor='rgba(0,0,0,0)'))
        )])

        # Configurações de layout
        fig_kmeans_ieg.update_layout(
            title='Clusters dos Alunos (K-Means) - IEG',
            scene=dict(
                xaxis_title='IEG 2020',
                yaxis_title='IEG 2021',
                zaxis_title='IEG 2022',
            ),
            width=800,  # largura da figura em pixels
            height=800,  # altura da figura em pixels
        )
        st.plotly_chart(fig_kmeans_ieg, use_container_width= True)

        '''
        - Cluster 0 - possui 32 alunos cujo a média em 2020 é de 7,5 com queda expressiva em 2021 e recuperação (abaixo ainda do inicial) em 6,8 em 2022
        - cluster 1 - possui 114 alunos com média inicial de 8,6, com uma queda em 2021 para 6,7 e ligeira recuperação com média de 7,5 
        - Cluster 2 - possui 38 alunos com baixa média entre e queda nos anos observados 6,0 em 2020, 5,3 em 2021 e 4,5 em 2022 (grupo de menor valor no indicador de Engajamento)
        - Cluster 3 - possui um grupo de 130 alunos com as maiores médias observadas nesse indicador entre 9,3 e 8,9 (com máximas observadas nos 3 anos de notas 10 e minimo de 9,1) 
        '''

        st.subheader('IPS - Indicador Psicosocial')

        file_path_ips = 'bases/df_ips.csv'
        pd.set_option('display.max_columns', None)
        df_ips = pd.read_csv(file_path_ips, delimiter=';')

        fig_kmeans_ips = go.Figure(data=[go.Scatter3d(
            x=df_ips['IPS_2020'],
            y=df_ips['IPS_2021'],
            z=df_ips['IPS_2022'],
            mode='markers',
            marker=dict(color=df_ips['cluster'], colorscale='Viridis', size=14, opacity=0.7, colorbar=dict(title='Cluster', outlinecolor='rgba(0,0,0,0)'))
        )])

        fig_kmeans_ips.update_layout(
            title='Clusters dos Alunos (K-Means) - IPS',
            scene=dict(
                xaxis_title='IPS 2020',
                yaxis_title='IPS 2021',
                zaxis_title='IPS 2022',
            ),
            width=800,  # largura da figura em pixels
            height=800,  # altura da figura em pixels
        )

        st.plotly_chart(fig_kmeans_ips, use_container_width= True)

        '''
        - Cluster 0 - possui 62 alunos com uma média baixa inicial de 4,6 em 2020, 6,3 em 2021 e queda para 5,6, sendo o cluster com as menores notas minimas e máximas (apresenta o menor desempenho no indicador)
        - Cluster 1 - possui 162 alunos com média estável nos 3 anos entre 7,4 (2020) e 7,6 (2022). Observa-se também uma estabulidade na sua mediana e desempenho ligeiramente crescente dessas médias
        - Cluster 2 - possui 22 alunos com as piores médias e medianas observadas entre 5,4 2020, queda em 2021 (3,4) e ligeiro aumento em 2022 (5,6)
        - Cluster 3 - possui 68 alunos com a segunda pior média observada com comportamento descrescente com o tempo e com destaque à oscilação para baixo também da mediana (7,5 em 2020 e 2021 e 5,6 em 2022)
                '''
        st.subheader('IDA - Indicador de Aprendizagem')

        file_path_ida = 'bases/df_ida.csv'
        pd.set_option('display.max_columns', None)
        df_ida = pd.read_csv(file_path_ida, delimiter=';')

        fig_kmeans_ida = go.Figure(data=[go.Scatter3d(
            x=df_ida['IDA_2020'],
            y=df_ida['IDA_2021'],
            z=df_ida['IDA_2022'],
            mode='markers',
            marker=dict(color=df_ips['cluster'], colorscale='Viridis', size=14, opacity=0.7, colorbar=dict(title='Cluster', outlinecolor='rgba(0,0,0,0)'))
        )])

        fig_kmeans_ida.update_layout(
            title='Clusters dos Alunos (K-Means) - IDA',
            scene=dict(
                xaxis_title='IDA 2020',
                yaxis_title='IDA 2021',
                zaxis_title='IDA 2022',
            ),
            width=800,  # largura da figura em pixels
            height=800,  # altura da figura em pixels
        )
        st.plotly_chart(fig_kmeans_ida, use_container_width= True)

        '''
        - Cluster 0 - possui 78 alunos com média nesse indicador descrecente de forma expressiva a partir de 8,4 em 2020, 4,4 e 4,2 respectivamente em 2021 e 2022
        - Cluster 1 - possui 85 alunos e destaca-se pela média estável entre 6,5 (2020) e 6,7 (2022) observada pela mediana e desvio padrão
        - Cluster 2 - possui 95 alunos com alta média em 2020 (9,1), queda para faixa de 7,3 (2021) e 7,5 (2022). Seus máximos e mínimos oscilaram para cima no último periodo observado e é o cluster com maior média nesse indicador
        - Cluster 3 - possui 56 alunos e apresenta a pior média nos 3 períodos observados (3,3 em 2020, 3,2 em 2021 e 4,2 em 2022). Os máximos observados também foram baixo com uma recuperação em 2022 (8 pontos)
        '''

        st.subheader('IAN - Indicador de Adequação ao Nivel')

        file_path_ian = 'bases/df_ian.csv'
        pd.set_option('display.max_columns', None)
        df_ian = pd.read_csv(file_path_ian, delimiter=';')

            # Criar figura 3D com Plotly
        fig_kmeans_ian = go.Figure(data=[go.Scatter3d(
            x=df_ian['IAN_2020'],
            y=df_ian['IAN_2021'],
            z=df_ian['IAN_2022'],
            mode='markers',
            marker=dict(color=df_ian['cluster'], colorscale='Viridis', size=14, opacity=0.7, colorbar=dict(title='Cluster', outlinecolor='rgba(0,0,0,0)'))
        )])

        # Configurações de layout
        fig_kmeans_ian.update_layout(
            title='Clusters dos Alunos (K-Means) - IAN',
            scene=dict(
                xaxis_title='IAN 2020',
                yaxis_title='IAN 2021',
                zaxis_title='IAN 2022',
            ),
            width=800,  # largura da figura em pixels
            height=800,  # altura da figura em pixels
        )
        st.plotly_chart(fig_kmeans_ian, use_container_width= True)

        '''
        - Cluster 0 - é composto por 61 alunos com alta média em 2020 (8,4), avaliação 10 em 2021 e média 5 em 2022 
        - Cluster 1 - contém 62 alunos também com média inicio de 10 em 2020 porém com queda mais expressiva em 2021 (4,9) e baixa recuperação (nota 6 no útimo período)
        - Cluster 2 - cluster com 128 alunos e piores médias: 2020 4,9, 2021 4,9 2022 5,2
        - Cluster 3 - contém 63 alunos com as maiores médias oscilando de 9,4 em 2020 para 10 anos seguintes
        Os valores aparentemente atipicos acima são corroborados pelo desvio padrão zerado 
        '''
        st.subheader('IPP - Indicador Psicopedagógico')

        file_path_ipp = 'bases/df_ipp.csv'
        pd.set_option('display.max_columns', None)
        df_ipp = pd.read_csv(file_path_ipp, delimiter=';')

        fig_kmeans_ipp = go.Figure(data=[go.Scatter3d(
            x=df_ipp['IPP_2020'],
            y=df_ipp['IPP_2021'],
            z=df_ipp['IPP_2022'],
            mode='markers',
            marker=dict(color=df_ipp['cluster'], colorscale='Viridis', size=14, opacity=0.7, colorbar=dict(title='Cluster', outlinecolor='rgba(0,0,0,0)'))
        )])

        fig_kmeans_ipp.update_layout(
            title='Clusters dos Alunos (K-Means) - IPP',
            scene=dict(
                xaxis_title='IPP 2020',
                yaxis_title='IPP 2021',
                zaxis_title='IPP 2022',
            ),
            width=800,  # largura da figura em pixels
            height=800,  # altura da figura em pixels
        )
        st.plotly_chart(fig_kmeans_ipp, use_container_width= True)
        '''
        - Cluster 0 - contém 57 alunos com média e mediana entre 6 e 7 com baixo desvio padrão 
        - Cluster 1 - cluster com 30 alunos e piores médias nas observações sendo 3,0 em 2020, 6,4 em 2021 e 5,4 em 2022
        - Cluster 2 - possui 130 alunos com uma média estável em 2020 (7,6) e 2021 (7,7) e queda importante observada no último periodo na média (5,9) 
        bem como nos pontos máximos obtidos (somente 7), valores também com baixo desvio padrão
        '''

    with aba3:

        coluna1, = st.columns(1)

        with coluna1:
            st.subheader('Features Importances')

            '''

            Fizemos uma avaliação dos dados para entendermos as importâncias dos indicadores para um modelo de previsão do INDE. Dado que temos 7 indicadores que compões o INDE, recorremos à um algoritimo de Machine Learning muito robusto e eficiente para essa análise.
            Utilizamos o Random Forest Regressor que é um metaestimador que ajusta um número de regressores de árvores de decisão em várias subamostras do conjunto de dados 
            e usa a média para melhorar a precisão preditiva e controlar o sobreajustes (técnica de bootstrap). Utilizamos a função “squared_error” para o erro quadrático médio, 
            que é igual à redução de variância como critério de seleção de recursos e minimiza a perda L2 usando a média de cada nó terminal.

            Esse outra vertente pode servir como base para estudos futuros para determinados comportamentos dos indicadores do ponto de vista da predição do INDE
            considerando métodos de esemble.
    
            '''

            # Pre processing
            # Função para carregar e processar dados
     
            # Função para pré-processamento dos dados
            def preprocess_data(df, categorical_cols, label_encoders, numeric_imputer, feature_names):
                # Adicionar colunas faltantes com valores nulos
                for feature in feature_names:
                    if feature not in df.columns:
                        df[feature] = pd.NA
                
                # Garantir que as colunas estejam na ordem correta
                df = df[feature_names]

                # Codificar variáveis categóricas
                for col in categorical_cols:
                    le = label_encoders.get(col)
                    if le:
                        df[col] = le.transform(df[col].astype(str))

                # Converter colunas restantes para numérico
                numeric_features = df.apply(pd.to_numeric, errors='coerce')

                # Verificar se há colunas numéricas
                if numeric_features.empty:
                    st.error('Não há colunas numéricas detectadas nos dados de entrada.')
                    return pd.DataFrame()

                # Imputar valores faltantes
                df_imputed = pd.DataFrame(numeric_imputer.transform(numeric_features), columns=numeric_features.columns)
                return df_imputed

            @st.cache_resource
            def load_model_and_encoders():
                model_features = joblib.load('model/rf_features.pkl')
                label_encoders = joblib.load('model/label_encoders.pkl')
                numeric_imputer = joblib.load('model/numeric_imputer.pkl')
                return model_features, label_encoders, numeric_imputer

            @st.cache_data
            def load_original_data():
                df_features = pd.read_csv('bases/df_features_importances.csv')
                return df_features

            st.subheader('Predição Random Forest e Features Importances')

            model_features, label_encoders, numeric_imputer = load_model_and_encoders()
            df_features = load_original_data()

            feature_names = df_features.columns.tolist()
            if 'INDE_2022' in feature_names:
                feature_names.remove('INDE_2022')

            st.write('Insira os valores para todas as features:')

            input_data = {}
            for feature in feature_names:
                value = st.number_input(
                    f'Digite uma nota de 0 a 10 (use ponto para decimais) para {feature}', 
                    min_value=0.0, 
                    max_value=10.0, 
                    format="%.2f"
                )
                input_data[feature] = [value]

            input_df = pd.DataFrame(input_data)

            # Identificar colunas categóricas (se houver)
            categorical_cols = input_df.select_dtypes(include=['object']).columns

            if st.button('Enviar', key='features_button'):
                input_df_processed = preprocess_data(input_df, categorical_cols, label_encoders, numeric_imputer, feature_names)

                if not input_df_processed.empty:
                    prediction = model_features.predict(input_df_processed)

                    # Classificar a previsão
                    if 2.405 <= prediction[0] <= 5.506:
                        label = 'Quartzo'
                    elif 5.506 < prediction[0] <= 6.868:
                        label = 'Agata'
                    elif 6.868 < prediction[0] <= 8.230:
                        label = 'Ametista'
                    elif 8.230 < prediction[0] <= 9.294:
                        label = 'Topazio'
                    else:
                        label = 'Unknown'

                    st.subheader(f'Predição INDE com Random Forest (Features Importances): {prediction[0]:.3f}')
                    st.subheader(f'Pedra: {label}')

                    # Obter as importâncias das características do modelo
                    importances = model_features.feature_importances_

                    # Criar DataFrame para visualização
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    })
                    
                    # Ordenar por importância
                    importance_df = importance_df.sort_values(by='importance', ascending=True)
                    
                    # Plotar a importância das características
                    fig_importance = px.bar(
                        importance_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Features Importances',
                        labels={'importance': 'Importância', 'feature': 'Características'},
                    )
                    st.plotly_chart(fig_importance)

                    '''
                    O algoritimo identificou durante a fase de treinamento as 3 features de maior relevância na predição do INDE (base de dados 2022): IDA (indicador de Aprendizagem - maior destaque),
                    IEG (Indicador de Engajamento) e IPV (indicador de Ponto de Virada). De certa forma corrobora com uma dos insigths trazidos pela equipe da Passos do ponto de vista do olhar na evolução dos indicadores de Aprendizagem (foco do desenvolvimento escolar),
                    Engajamento (importante para a contribuição e participação nas atividades extra-classe e de integração social) e Ponto de Virada (quando o aluno de fato está no "flow" do aprendizado).

                    O MSE (Mean Squared Error) do Random Forest foi de 0,1019.
                                                           
                    '''

            else:
                st.write('')

                

    with aba4:

        coluna1, = st.columns(1)

        with coluna1:
            st.subheader('Previsão do INDE com KNN Regressor')
            '''
            De acordo com a documentação do Sklearn, a regressão baseada no KNN é comendada para ser utilizada em casos em que os rótulos de dados são variáveis ​​contínuas em vez de discretas, por essa razão definimos esse algoritmo como preditor,
            dado que temos um conjunto de notas médias.

            O treinamento do modelo teve como base as informações do último ano da base (2022) considerando o período pós pandemia, com o objetivo também de reduzirmos a dimesionalidade dos dados para melhor comportamento do algoritimo.

            O grafico abaixo mostra o resultado do teste do modelo KNN após o treinamento cujo MSE (Mean Squared Error) foi de 0.038:
            
            
            '''
            st.image(image='img/treinamento_knn.png', caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
            

        # Carregar o KNN
        regressor_knn = joblib.load('model/knn_model_inde.pkl')

        st.subheader('Calcule a previsão')

        input_features = []

        numeric_cols = ['IAA', 'IEG', 'IPS', 'IDA', 'IPP', 'IPV', 'IAN']

        # Criação dos inputs interativos para o Streamlit
        for feature in numeric_cols:
            value = st.number_input(f'Digite uma nota de 0 a 10 (use ponto para decimais) para {feature}', min_value=0.0, max_value=10.0, format="%.2f")
            input_features.append(value)

        if st.button('Enviar', key='knn_button'):   

            # Converter a lista para um array numpy e reformatar
            input_features = np.array(input_features).reshape(1, -1)

            # Fazer a previsão usando o modelo treinado
            prediction = regressor_knn.predict(input_features)

            # Classificar a previsão com base nos intervalos de INDE_2022
            if 2.405 <= prediction[0] <= 5.506:
                label = 'Quartzo'
            elif 5.506 < prediction[0] <= 6.868:
                label = 'Agata'
            elif 6.868 < prediction[0] <= 8.230:
                label = 'Ametista'
            elif 8.230 < prediction[0] <= 9.294:
                label = 'Topazio'
            else:
                label = 'Unknown'

            # Exibir a previsão e sua classificação
            #st.subheader('Resultados')
            st.subheader(f'INDE previsto: {prediction[0]:.3f}')
            st.subheader(f'Pedra: {label}')

            fig_prev = go.Figure()

            fig_prev.add_trace(go.Bar(
                x=numeric_cols,
                y=input_features[0],
                name='Valores Digitados'
            ))

            fig_prev.add_trace(go.Bar(
                x=['INDE Previsto'], 
                y=[prediction[0]],
                name='INDE Previsto',
                marker_color='gray'
            ))

            fig_prev.update_layout(
                title='Media dos indicadores e Previsão do INDE',
                xaxis_title='Indicadores',
                yaxis_title='nota média',
                template='plotly_white',
                barmode='group'  
            )

            st.plotly_chart(fig_prev)

    
