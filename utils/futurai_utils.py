import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import boto3
from datetime import datetime, timedelta
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy
from scipy.stats import invgauss
from scipy.stats.distributions import chi2
from scipy import stats
import warnings

import futurai_ppd as ppd
import requests
import shutil
import json

from time import sleep
import concurrent.futures
import threading
from functools import partial

warnings.filterwarnings("ignore")


def graph_predict(phi, eixo_x, threshold, freq="1T", list_periods=None):
    df = pd.DataFrame({"phi": phi, "timestamp": eixo_x})

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("timestamp")
    df = df.resample(freq).asfreq()
    df = df["phi"].fillna(0)
    df = df.reset_index()

    df["threshold"] = threshold

    layout = go.Layout(
        plot_bgcolor="#FFF"
    )
    
    fig = go.Figure(layout=layout)
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["phi"], mode="lines", name="Índice", fill="tozeroy", line_color="#0F293A")
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["threshold"], mode="lines", name="Limiar", line_color="#FB8102")
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#CECFD1')
    fig.update_layout(hovermode='x unified')
    fig.update_layout(legend=dict(orientation="h"))
    fig.update_layout(yaxis_range=[0, 4 * threshold])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

    if list_periods and list_periods != []:
        ### Add a single dummy shape plot for the legend ###
        fig.add_shape(
                showlegend=True,
                type="rect",
                x0=str(list_periods[0]["date_ini"]),
                y0=0,
                x1=str(list_periods[0]["date_ini"]),
                y1=0,
                fillcolor='#68cbf8',
                opacity=1,
                line_width=0,
                layer="below",
                name="Desligado",
            )
        fig.add_shape(
                showlegend=True,
                type="rect",
                x0=str(list_periods[0]["date_ini"]),
                y0=0,
                x1=str(list_periods[0]["date_ini"]),
                y1=0,
                fillcolor='#b3e5fc',
                opacity=1,
                line_width=0,
                layer="below",
                name="Transitório",
            )
        for periodo in list_periods:
            if periodo["type"] == "desligado":
                color = '#68cbf8'
            elif periodo["type"] == "transitorio":
                color = '#b3e5fc'
            else:
                color = "red"

            fig.add_shape(
                type="rect",
                x0=str(periodo["date_ini"]),
                y0=0,
                x1=str(periodo["date_end"]),
                y1=df["phi"].max()*4,
                fillcolor=color,
                opacity=1,
                line_width=0,
                layer="below"
            )

    return fig


def graph_variables(df, eixo ,variaveis=[], freq="1T", pre_process=None, list_periods_anom=None, df_projection=None):
    ## Converte eixoX para lista
    df["timestamp"] = eixo.to_list()
    
    ## Criando lista de períodos desligados
    list_periods_off = []
    if pre_process:
        for pro in pre_process:
            _,_,list_aux = ppd.drop_transitorio_desligado(df,pro["variable_off"],pro["limit_off"],pro["interval_off"],"timestamp",pre_corte=pro["pre_cut"],pos_corte=pro["after_cut"])
            after_cut = pro["after_cut"]
            list_periods_off = [*list_periods_off,*list_aux]    
    
    ## Ajustando eixoX com amostras faltantes
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M:%S")
    df = df.set_index("timestamp")
    df = df.resample(freq).asfreq()
    df = df.fillna(0)
    df = df.reset_index()
    eixo = df["timestamp"]
    df = df.drop("timestamp", axis=1)
    
    ## Cria objeto figura
    fig = go.Figure()
    
    ## Lista com 1 item = Univariado || Lista com de um item = Normalizado || Lista vazia = Normalizado com todas as variáveis
    if len(variaveis) == 1:
        ## Cria eixo Y com valor das variáveis
        y = [item for sublist in df[variaveis].values for item in sublist]
        fig.add_trace(go.Scatter(x=eixo, y=y, mode="lines", name=variaveis[0],line_color="#0F293A"))
        
        ## Definindo máximo e mínimo do eixo y
        y_max = max(y)
        y_min = min(y)
        ## df_projection = None então gráfico gerado tem projeção
        if df_projection is not None:
            ## Definindo novamente y_max caso necessário
            if max(df_projection[variaveis[0]]) > y_max:
                y_max = max(df_projection[variaveis[0]])
            ## Definindo novamente y_min caso necessário    
            if min(df_projection[variaveis[0]]) < y_min:
                y_min = min(df_projection[variaveis[0]])
            ## Verifica se anomalia se inicia com um desligado e corrige o número de amostras    
            if len(list_periods_off) > 0:
                if list_periods_off[0]["date_ini"] == eixo[0].to_pydatetime():
                    diff = list_periods_off[0]['date_end'] - list_periods_off[0]['date_ini']
                    diff_minutes = diff.total_seconds() / 60
                    df_projection = pd.concat([pd.DataFrame(np.nan, index=range(int(diff_minutes)+int(after_cut)), columns=df_projection.columns), df_projection], ignore_index=True)
                
            fig.add_trace(go.Scatter(x=eixo, y=df_projection[variaveis[0]], mode="lines", name="Projeção",line_color="red",line = dict(width=3, dash='dot')))
        
        
        ##### Adicionando sombreado nos períodos de ultrapassagem do limiar #####
        if list_periods_anom and list_periods_anom != []:
            ### Adiciona ponto no gráfico para aparecer na legenda ###
            fig.add_shape(
                showlegend=True,
                type="rect",
                x0=str(list_periods_anom[0]["date_ini"]),
                y0=min(y),
                x1=str(list_periods_anom[0]["date_ini"]),
                y1=min(y),
                fillcolor='red',
                opacity=0.2,
                line_width=0,
                layer="below",
                name="Em Anomalia"

            )
            for periodo in list_periods_anom:
                fig.add_shape(
                    type="rect",
                    x0=str(periodo["date_ini"]),
                    y0=y_min,
                    x1=str(periodo["date_end"]),
                    y1=y_max,
                    fillcolor='red',
                    opacity=0.2,
                    line_width=0,
                    layer="below"
                )

        ##### Adicionando sombreado nos períodos de desligado #####
        if pre_process and list_periods_off != []:
            ### Adiciona ponto no gráfico para aparecer na legenda ###
            fig.add_shape(
                showlegend=True,
                type="rect",
                x0=str(list_periods_off[0]["date_ini"]),
                y0=min(y),
                x1=str(list_periods_off[0]["date_ini"]),
                y1=min(y),
                fillcolor='#68cbf8',
                opacity=1,
                line_width=0,
                layer="below",
                name="Desligado"
            )
            fig.add_shape(
                    showlegend=True,
                    type="rect",
                    x0=str(list_periods_off[0]["date_ini"]),
                    y0=0,
                    x1=str(list_periods_off[0]["date_ini"]),
                    y1=0,
                    fillcolor='#b3e5fc',
                    opacity=1,
                    line_width=0,
                    layer="below",
                    name="Transitório",
                )
            
            for periodo in list_periods_off:
                if periodo["type"] == "desligado":
                    fig.add_shape(
                        type="rect",
                        x0=str(periodo["date_ini"]),
                        y0=y_min,
                        x1=str(periodo["date_end"]),
                        y1=y_max,
                        fillcolor='#68cbf8',
                        opacity=1,
                        line_width=0,
                        layer="below"
                    )
                if periodo["type"] == "transitorio":
                    fig.add_shape(
                        type="rect",
                        x0=str(periodo["date_ini"]),
                        y0=y_min,
                        x1=str(periodo["date_end"]),
                        y1=y_max,
                        fillcolor='#b3e5fc',
                        opacity=1,
                        line_width=0,
                        layer="below"
                    )     
        
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(yaxis_range=[y_min,y_max])
        fig.update_layout(legend=dict(orientation="h"))
        fig.update_layout(showlegend=True)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#CECFD1')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#CECFD1')
        
    else:
        if len(variaveis) != 0:
            df = df.loc[:, variaveis]

        tamanho = df.shape
        start = 0
        end = 1
        x1 = eixo
        colunas = df.columns
        
        ## Adicionando sombreado nos períodos de desligado
        if pre_process and list_periods_off != []:
            ### Add a single dummy shape plot for the legend ###
            fig.add_shape(
                showlegend=True,
                type="rect",
                x0=str(list_periods_off[0]["date_ini"]),
                y0=0,
                x1=str(list_periods_off[0]["date_ini"]),
                y1=0,
                fillcolor='#68cbf8',
                opacity=1,
                line_width=0,
                layer="below",
                name="Desligado"

            )
            for periodo in list_periods_off:
                if periodo["type"] == "desligado":
                    fig.add_shape(
                        type="rect",
                        x0=str(periodo["date_ini"]),
                        y0=0,
                        x1=str(periodo["date_end"]),
                        y1=tamanho[1],
                        fillcolor='#68cbf8',
                        opacity=1,
                        line_width=0,
                        layer="below"
                    )
                    
        ##### Adicionando sombreado nos períodos de ultrapassagem do limiar #####
        if list_periods_anom and list_periods_anom != []:
            ### Add a single dummy shape plot for the legend ###
            fig.add_shape(
                showlegend=True,
                type="rect",
                x0=str(list_periods_anom[0]["date_ini"]),
                y0=0,
                x1=str(list_periods_anom[0]["date_ini"]),
                y1=0,
                fillcolor='red',
                opacity=0.2,
                line_width=0,
                layer="below",
                name="Em Anomalia"

            )
            for periodo in list_periods_anom:
                fig.add_shape(
                    type="rect",
                    x0=str(periodo["date_ini"]),
                    y0=0,
                    x1=str(periodo["date_end"]),
                    y1=tamanho[1],
                    fillcolor='red',
                    opacity=0.2,
                    line_width=0,
                    layer="below"
                )

        for x in range(tamanho[1]):
            arr = df.iloc[:, x]
            width = end - start
            res = (arr - arr.min()) / np.ptp(arr) * width + start

            fig.add_trace(go.Scatter(x=x1, y=res, mode="lines", name=colunas[x]))
            fig.update_traces(line=dict(width=1))
            start = start + 1
            end = end + 1
        
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(legend_traceorder="reversed")
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#CECFD1')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#CECFD1')
        
    return fig

def graph_contribution(dict_contribuicao, dict_score, escala, freq="1T"):
    df_contribuicao = pd.DataFrame().from_dict(dict_contribuicao)

    df_contribuicao["timestamp"] = pd.to_datetime(
        df_contribuicao["timestamp"], format="%Y-%m-%d %H:%M:%S"
    )
    df_contribuicao = df_contribuicao.set_index("timestamp")
    df_contribuicao = df_contribuicao.resample(freq).asfreq()
    df_contribuicao = df_contribuicao.fillna(0)
    df_contribuicao = df_contribuicao.reset_index()

    eixo_x = df_contribuicao["timestamp"]
    df_contribuicao = df_contribuicao.drop("timestamp", axis=1)

    # Filtrar as variáveis que possuem score
    variables_with_score = [dict_score['VARIAVEL'][str(i)] for i in range(len(dict_score['VARIAVEL'])) if dict_score['score'][str(i)]]

    # Filtrar apenas as variáveis com score do DataFrame de contribuição
    df_contribuicao = df_contribuicao[variables_with_score]

    fig1 = px.imshow(
        df_contribuicao.T,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        x=eixo_x,
        zmax=escala,
    )

    dict_score = pd.DataFrame().from_dict(dict_score)
    dict_score.rename(
        columns={"SISTEMA": "LOCAL", "DESC": "Descricao"}, inplace=True
    )  

    # Filtrar apenas as entradas com score
    dict_score = dict_score[dict_score['score'] > 0]

    fig2 = px.sunburst(
        data_frame=dict_score,
        path=["LOCAL", "Descricao"],
        values="score",
        maxdepth=-1,
        color="%",
        color_continuous_midpoint=1,
        color_continuous_scale=px.colors.sequential.Blues,
        range_color=[1, 100],
    )

    fig2.update_traces(textinfo="label+percent entry")
    fig2.update_layout(margin=dict(t=0, l=0, r=0, b=0))

    return fig1, fig2


def select_training_period(df_dataset, days, timestamp):
    df_np = df_dataset.copy()
    time = df_np[timestamp].to_list()
    df_np = df_np.drop(timestamp, axis=1).copy()
    array_np = df_np.to_numpy()

    ##PCA
    scaler = StandardScaler()
    # Fit on training set only.
    array_np_std = scaler.fit_transform(array_np)
    cov = np.cov(array_np_std.T)
    u, s, vh = np.linalg.svd(cov)
    pca = PCA(0.95)
    pca.fit(array_np_std)
    pc = pca.transform(array_np_std)
    nc = pc.shape[1]
    s_diag = np.diag(s)
    s_pcs = s_diag[:nc, :nc]

    ##T2
    t2 = []
    for i in range(pc.shape[0]):
        termo1 = pc[i]
        termo2 = np.linalg.inv(s_pcs)
        termo3 = pc[i].T

        t2.append(termo1.dot(termo2).dot(termo3))
    M = pc.shape[1]
    N = pc.shape[0]
    F = scipy.stats.f.ppf(0.95, M, N - M)
    t2_lim = (M * (N - 1) / (N - M)) * F

    ##SPE
    spe = []
    for i in range(pc.shape[0]):
        rs = array_np_std[i].dot(u[:, nc - 1 :])
        termo1 = rs.T
        termo2 = rs
        spe.append(termo1.dot(termo2))
    teta1 = (s_diag[nc - 1 :]).sum()
    teta2 = (s_diag[nc - 1 :] ** 2).sum()
    teta3 = (s_diag[nc:-1, :] ** 3).sum()
    h0 = 1 - (2 * teta1 * teta3) / (3 * teta2**2)
    mu = 0.145462645553
    vals = invgauss.ppf([0, 0.999], mu)
    ca = invgauss.cdf(vals, mu)[1]
    spe_lim = teta1 * (
        (h0 * ca * np.sqrt(2 * teta2) / teta1)
        + 1
        + (teta2 * h0 * (h0 - 1)) / (teta1**2)
    ) ** (1 / h0)

    ##PHI
    phi = []
    for i in range(pc.shape[0]):
        phi.append((spe[i] / spe_lim) + (t2[i] / t2_lim))
    gphi = ((nc / t2_lim**2) + (teta2 / spe_lim**2)) / (
        (nc / t2_lim) + (teta1 / spe_lim)
    )
    hphi = ((nc / t2_lim) + (teta1 / spe_lim)) ** 2 / (
        (nc / t2_lim**2) + (teta2 / spe_lim**2)
    )
    chi2.ppf(0.975, df=2)
    phi_lim = gphi * chi2.ppf(0.99, hphi)
    df_t2 = pd.DataFrame(
        {
            "time": time,
            "t2": t2,
            "spe": spe,
            "phi": phi,
        }
    )

    df_t2["t2_lim"] = t2_lim
    df_t2["spe_lim"] = spe_lim
    df_t2["phi_lim"] = phi_lim

    df_t2["t2"] = df_t2["t2"].ewm(alpha=0.01).mean()
    df_t2["spe"] = df_t2["spe"].ewm(alpha=0.01).mean()
    df_t2["phi"] = df_t2["phi"].ewm(alpha=0.01).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_t2["time"], y=df_t2["phi"], mode="lines"))
    fig.add_trace(go.Scatter(x=df_t2["time"], y=df_t2["phi_lim"], mode="lines"))

    data_max = df_t2["time"].max()
    data_min = df_t2["time"].min()
    data_aux = data_min

    lista_data_inicio = []
    lista_data_fim = []
    numero_dias = []
    while data_aux <= data_max:
        mask = (df_t2["time"] >= data_aux) & (df_t2["phi"] < phi_lim)
        df_aux = df_t2.loc[mask]
        if not df_aux.empty:
            data_pos_min = df_aux["time"].min()
        else:
            break

        mask = (df_t2["time"] >= data_aux) & (df_t2["phi"] >= phi_lim)
        df_aux2 = df_t2.loc[mask]
        if not df_aux2.empty:
            data_pos_max = df_aux2["time"].min()
            data_aux = data_pos_max
        else:
            break

        interval = data_pos_max - data_pos_min
        if interval.days >= days:
            lista_data_inicio.append(data_pos_min)
            lista_data_fim.append(data_pos_max)
            numero_dias.append(interval.days)

        data_aux = data_pos_max
        data_aux = data_aux + timedelta(minutes=1)
    
    media = []
    desvio = []
    variancia = []
    for i in range(0,len(lista_data_inicio)):
        mask = (df_t2["time"] >= lista_data_inicio[i]) & (df_t2["time"] <= lista_data_fim[i])
        df_aux = df_t2.loc[mask]
        
        media.append(df_aux['phi'].mean())
        desvio.append(df_aux['phi'].std())
        variancia.append(df_aux['phi'].var())
    
    
    df_aux2 = pd.DataFrame(
        {
            "DATA INÍCIO": lista_data_inicio,
            "DATA FIM": lista_data_fim,
            "NÚMERO DE DIAS": numero_dias,
            "MÉDIA PHI": media,
            "DESVIO PADRÃO PHI": desvio,
            "VARIÂNCIA PHI": variancia
        }
    )
    #df_aux2 = df_aux2.sort_values(["VARIÂNCIA PHI"])
    
    # Criar uma lista para armazenar os novos dados
    new_data = []

    # Percorrer as linhas do DataFrame original
    for index, row in df_aux2.iterrows():
        num_dias = row["NÚMERO DE DIAS"]
        if num_dias > days:
            # Calcular quantos períodos de 'days' dias são necessários
            num_periodos = num_dias // days
            dias_resto = num_dias % days
            for i in range(num_periodos):
                # Selecionar os dados dentro do período de 'days' dias
                mask = (df_t2["time"] >= row["DATA INÍCIO"] + pd.Timedelta(days=i*days)) & \
                       (df_t2["time"] <= row["DATA INÍCIO"] + pd.Timedelta(days=(i+1)*days - 1))
                df_periodo = df_t2.loc[mask]

                # Calcular média, desvio padrão e variância para o período de 'days' dias
                media_phi = df_periodo['phi'].mean()
                desvio_padrao_phi = df_periodo['phi'].std()
                variancia_phi = df_periodo['phi'].var()

                # Adicionar um novo período de 'days' dias ao DataFrame
                new_data.append({
                    "DATA INÍCIO": row["DATA INÍCIO"] + pd.Timedelta(days=i*days),
                    "DATA FIM": row["DATA INÍCIO"] + pd.Timedelta(days=(i+1)*days - 1),
                    "NÚMERO DE DIAS": days,
                    "MÉDIA PHI": media_phi,
                    "DESVIO PADRÃO PHI": desvio_padrao_phi,
                    "VARIÂNCIA PHI": variancia_phi,
                    "TERÇO": i % 3 + 1
                })
            # Se houver um resto, adicionar o período restante
            if dias_resto > 0:
                # Selecionar os dados dentro do período restante
                mask = (df_t2["time"] >= row["DATA INÍCIO"] + pd.Timedelta(days=num_periodos*days)) & \
                       (df_t2["time"] <= row["DATA FIM"])
                df_periodo = df_t2.loc[mask]

                # Calcular média, desvio padrão e variância para o período restante
                media_phi = df_periodo['phi'].mean()
                desvio_padrao_phi = df_periodo['phi'].std()
                variancia_phi = df_periodo['phi'].var()

                new_data.append({
                    "DATA INÍCIO": row["DATA INÍCIO"] + pd.Timedelta(days=num_periodos*days),
                    "DATA FIM": row["DATA FIM"],
                    "NÚMERO DE DIAS": dias_resto,
                    "MÉDIA PHI": media_phi,
                    "DESVIO PADRÃO PHI": desvio_padrao_phi,
                    "VARIÂNCIA PHI": variancia_phi,
                    "TERÇO": num_periodos % 3 + 1
                })
        else:
            # Se o número de dias for menor ou igual a 'days', manter o período original
            new_data.append({
                "DATA INÍCIO": row["DATA INÍCIO"],
                "DATA FIM": row["DATA FIM"],
                "NÚMERO DE DIAS": num_dias,
                "MÉDIA PHI": row["MÉDIA PHI"],
                "DESVIO PADRÃO PHI": row["DESVIO PADRÃO PHI"],
                "VARIÂNCIA PHI": row["VARIÂNCIA PHI"],
                "TERÇO": 0
            })

    # Criar um novo DataFrame com os novos dados
    df_ok = pd.DataFrame(new_data)

    # Ordenar o DataFrame pelos valores da VARIÂNCIA PHI de forma crescente
    #df_ok = df_ok.sort_values(by="VARIÂNCIA PHI")

    return df_ok, fig


def get_dados():
    company_id = input('company_id: ')
    process_id = input('process_id: ')
    message = input('message: ')
    notebook_name = input('notebook_name: ')

    while True:
        env = int(input('Env - Digite [1]->prod | [2]->dev: '))
        if env == 1:
            env = 'prod'
            break
        elif env == 2:
            env = 'dev'
            break
        else:
            print("Escolha inválida. Digite 1 ou 2.")

    while True:
        eventbridge = int(input('EventBridge - Digite [1]->True | [2]->False: '))
        if eventbridge == 1:
            eventbridge = True
            break
        elif eventbridge == 2:
            eventbridge = False
            break
        else:
            print("Escolha inválida. Digite 1 ou 2.")

    while True:
        atv_realizada = int(input('Atividade Realizada - Digite [1]->Retreino | [2]->Aumento de transitório | [3]->Sem alterações | [4]->Remodelagem/Novo modelo:'))
        if atv_realizada == 1:
            atv_realizada = "Retreino"
            break
        elif atv_realizada == 2:
            atv_realizada = "Aumento de transitório"
            break
        elif atv_realizada == 3:
            atv_realizada = "Sem alterações"
            break
        elif atv_realizada == 4:
            atv_realizada = "Remodelagem/Novo modelo"
            break
        else:
            print("Escolha inválida!")

    while True:
        datafactory = int(input('DataFactory - Digite [1]->True | [2]->False: '))
        if datafactory == 1:
            datafactory = True
            break
        elif datafactory == 2:
            datafactory = False
            break
        else:
            print("Escolha inválida. Digite 1 ou 2.")

    while True:
        pp_period_good = int(input('pp_period_good: '))
        if pp_period_good <= 0:
            print("Escolha inválida. Digite valor maior que zero!")
        else:
            break

    while True:
        type_task = int(input('Tipo da Task - Digite [1]->NOVO MODELO/PILOTO | [2]->REAVALIAÇÃO: '))
        if type_task == 1:
            type_task = "Novo Modelo"
            break
        elif type_task == 2:
            type_task = "Reavaliação"
            break
        else:
            print("Escolha inválida. Digite 1 ou 2.")
            
    while True:
        author = int(input('Autor - Digite [1]->Gabriel Falcone | [2]->Gustavo Davel: '))
        if author == 1:
            author = 'Gabriel Falcone'
            break
        elif author == 2:
            author = 'Gustavo Davel'
            break
        else:
            print("Escolha inválida. Digite 1 ou 2.")

    return company_id, process_id, message, notebook_name, env, eventbridge, pp_period_good, type_task, atv_realizada, datafactory, author

def get_dados_classifier():
    company_id = input('company_id: ')
    process_id = input('process_id: ')
    notebook_name = input('notebook_name: ')
 
    while True:
        env = int(input('Env - Digite [1]->prod | [2]->dev: '))
        if env == 1:
            env = 'prod'
            break
        elif env == 2:
            env = 'dev'
            break
        else:
            print("Escolha inválida. Digite 1 ou 2.")
 
    while True:
        classification_type = int(input('Classification Type - Digite [1]->proba | [2]->binary: '))
        if classification_type == 1:
            classification_type = "proba"
            break
        elif classification_type == 2:
            classification_type = "binary"
            break
        else:
            print("Escolha inválida. Digite 1 ou 2.")
           
    while True:
        type_trigger = int(input('Trigger para anomalia - Digite [1]->PREDICT(Anomalia classificada após ultrapassagem do limiar) | [2]->CLASSIFICATION(Anomalia gerada pela classificação): '))
        if type_trigger == 1:
            type_trigger = "PREDICT"
            break
        elif type_trigger == 2:
            type_trigger = "CLASSIFICATION"
            break
        else:
            print("Escolha inválida. Digite 1 ou 2")

    pp_period_good = input('pp_period_good: ')
   
    print("")
           
    return company_id, process_id, notebook_name, classification_type, type_trigger, pp_period_good, env


def sens_verify(result,threshold):
    phi_last_7days = result['phi'][-10100:]
    
    terceiro_quartil = np.quantile(phi_last_7days, [0.75])
    quarto_quartil  = np.quantile(phi_last_7days, [1])

    terceiro_quartil_verify = threshold/terceiro_quartil[0]
    quarto_quartil_verify = threshold/quarto_quartil[0]
    

    if (terceiro_quartil_verify >= 12 and quarto_quartil_verify >= 3) or (terceiro_quartil_verify >25): 
        print("A sensibilidade do modelo está BAIXA, verifique o limiar!")
        print("ÍNDICE 3º Quartil: ", terceiro_quartil_verify) 
        print("ÍNDICE 4º Quartil: ", quarto_quartil_verify)
        
    else:
        print("Sensibilidade OK!")
        print("ÍNDICE 3º Quartil: ", terceiro_quartil_verify) 
        print("ÍNDICE 4º Quartil: ", quarto_quartil_verify)
    
    return True

def get_database_on_aws_by_period(company_id,process_id,process_name,timestamp,env,list_dates,raw_data=True):
    lock = threading.Lock()
    try:
        date_start = pd.to_datetime(list_dates['start_date'], infer_datetime_format=True)
        date_end = pd.to_datetime(list_dates['end_date'], infer_datetime_format=True)

        date_start_aux = date_start.date()
        date_end_aux = date_end.date()

        date_end_str = datetime.strftime(date_end_aux,'%d-%m-%Y')
        date_start_str = datetime.strftime(date_start_aux,'%d-%m-%Y')

        list_date = pd.date_range(date_start_aux, date_end_aux, freq='D')    

        if env == "prod":
            region = os.environ["REGION_PROD"]
            ACCESS_KEY = os.environ["ACCESS_KEY_PROD"]
            SECRET_KEY = os.environ["SECRET_KEY_PROD"]
        else:
            region = os.environ["REGION_DEV"]
            ACCESS_KEY = os.environ["ACCESS_KEY_DEV"]
            SECRET_KEY = os.environ["SECRET_KEY_DEV"]

        s3 = boto3.resource('s3',region_name=region,aws_access_key_id=ACCESS_KEY,aws_secret_access_key=SECRET_KEY)

        bucket_name = "client-" + company_id
        bucket = s3.Bucket(bucket_name)

        if raw_data:
            folder = 'raw-data/'
        else:
            folder = 'good-data/'

        df_data = pd.DataFrame()
        list_date = pd.date_range(date_start_aux, date_end_aux, freq='D')

        prefix_process = process_id
        for date in list_date:
            filtro = folder+ process_id + '/' + prefix_process + '_' + datetime.strftime(date,'%d-%m-%Y')
            for file in bucket.objects.filter(Prefix= filtro):
                file_name = process_id + date_end_str + date_start_str +'.csv'
                bucket.download_file(file.key,file_name)
                df_aux = pd.read_csv(file_name,sep=';',decimal='.')
                df_data = pd.concat([df_data, df_aux], ignore_index=True)
                os.remove(file_name)

        if df_data.empty:
            prefix_process = process_name
            for date in list_date:
                filtro = folder+ process_id + '/' + prefix_process + '_' + datetime.strftime(date,'%d-%m-%Y')
                for file in bucket.objects.filter(Prefix= filtro):
                    file_name = process_id+ date_end_str + date_start_str +'.csv'
                    bucket.download_file(file.key,file_name)
                    df_aux = pd.read_csv(file_name,sep=';',decimal='.')
                    df_data = pd.concat([df_data, df_aux], ignore_index=True)
                    os.remove(file_name)
        
        if df_data.empty:
            prefix_process = process_id
            for date in list_date:
                filtro = folder+ process_id + '/' + prefix_process + '_' + datetime.strftime(date,'%Y-%m-%d')
                for file in bucket.objects.filter(Prefix= filtro):
                    file_name = process_id+ date_end_str + date_start_str +'.csv'
                    bucket.download_file(file.key,file_name)
                    df_aux = pd.read_csv(file_name,sep=';',decimal='.')
                    df_data = pd.concat([df_data, df_aux], ignore_index=True)
                    os.remove(file_name)

        df_data.drop_duplicates(timestamp,inplace=True)
        df_data[timestamp]= pd.to_datetime(df_data[timestamp],format="%Y-%m-%d %H:%M:%S")
        mask = (df_data[timestamp] >= date_start) & (df_data[timestamp] <= date_end)
        df_data = df_data.loc[mask]

        sleep(0.5)
        
    except Exception as e:
        with lock:
            print("Período sem dados: " + list_dates['start_date'] + " a " + list_dates['end_date'])
    
    return df_data


def verificar_fuso_horario(data):
    try:
        formato_sem_fuso = "%Y-%m-%d %H:%M:%S"
        formato_com_fuso = "%Y-%m-%d %H:%M:%S%z"

        datetime.strptime(data, formato_com_fuso)
        return True
        
    except ValueError:
        return False


def get_s3_data(start_date, end_date,company_id,process_id,process_name,timestamp,env,raw_data):
    
    # Flags para verificar formato da data    
    flag_fuso_start = verificar_fuso_horario(start_date)
    flag_fuso_end = verificar_fuso_horario(end_date)

    if flag_fuso_start and flag_fuso_end:
        data_inicio = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S-03:00')
        data_fim = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S-03:00')
    else:
        # Converter as strings de data para objetos datetime
        data_inicio = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        data_fim = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

    # Lista para armazenar os dicionários de datas
    list_dates = []

    # Enquanto a data de início for menor ou igual à data final
    while data_inicio <= data_fim:
        # Verifica se a próxima data de início ultrapassaria a data final
        if (data_inicio + timedelta(days=6)) > data_fim:
            data_final_semana = data_fim  # Define a data final da semana como a data fim
        else:
            data_final_semana = data_inicio + timedelta(days=6)

        # Criar um dicionário com a data de início e fim da semana
        if flag_fuso_start and flag_fuso_end:
            novo_dicionario = {
                'start_date': data_inicio.strftime('%Y-%m-%d %H:%M:%S-03:00'),
                'end_date': data_final_semana.strftime('%Y-%m-%d %H:%M:%S-03:00')
            }
        else:
            novo_dicionario = {
                'start_date': data_inicio.strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': data_final_semana.strftime('%Y-%m-%d %H:%M:%S')
            }

        # Adicionar o dicionário à lista
        list_dates.append(novo_dicionario)

        # Avançar 7 dias para a próxima semana
        data_inicio += timedelta(days=5)
    
    max_workers_input = len(list_dates)

    get_database_on_aws_by_period_params = partial(get_database_on_aws_by_period,company_id,process_id,process_name,timestamp,env,raw_data=raw_data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_input) as executor:
        dataframes = executor.map(get_database_on_aws_by_period_params, list_dates)

    result_df = pd.concat(dataframes, ignore_index=True)
    
    return result_df
        

def get_anomalies(process_id, start_date, end_date):
    import pandas as pd

    # Carregar as anomalias
    df_anomalias = pd.read_csv("anomalias.csv", sep=',')

    # Converter as colunas de data
    df_anomalias['timestamp'] = pd.to_datetime(df_anomalias['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df_anomalias['start_date'] = pd.to_datetime(df_anomalias['start_date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df_anomalias['end_date'] = pd.to_datetime(df_anomalias['end_date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Normalizar a coluna real_anomaly
    df_anomalias['real_anomaly'] = df_anomalias['real_anomaly'].astype(str).str.lower()

    # Filtrar as anomalias
    df_anomalias_filtradas = df_anomalias[
        (df_anomalias["process_id"] == process_id) &
        (df_anomalias["real_anomaly"].isin(['true', 'false'])) &
        (df_anomalias["start_date"] >= pd.to_datetime(start_date)) &
        (df_anomalias["end_date"] <= pd.to_datetime(end_date))
    ]

    return df_anomalias_filtradas

def dev_select_training_period(df_dataset, days, timestamp, process_id, start_date, end_date):
    # Copiar o dataset e processar o timestamp
    df_np = df_dataset.copy()
    time = df_np[timestamp].to_list()
    df_np = df_np.drop(timestamp, axis=1).copy()
    array_np = df_np.to_numpy()

    ## PCA
    scaler = StandardScaler()
    array_np_std = scaler.fit_transform(array_np)
    cov = np.cov(array_np_std.T)
    u, s, vh = np.linalg.svd(cov)
    pca = PCA(0.95)
    pca.fit(array_np_std)
    pc = pca.transform(array_np_std)
    nc = pc.shape[1]
    s_diag = np.diag(s)
    s_pcs = s_diag[:nc, :nc]

    ## T2
    t2 = []
    for i in range(pc.shape[0]):
        termo1 = pc[i]
        termo2 = np.linalg.inv(s_pcs)
        termo3 = pc[i].T
        t2.append(termo1.dot(termo2).dot(termo3))
    M = pc.shape[1]
    N = pc.shape[0]
    F = scipy.stats.f.ppf(0.95, M, N - M)
    t2_lim = (M * (N - 1) / (N - M)) * F

    ## SPE
    spe = []
    for i in range(pc.shape[0]):
        rs = array_np_std[i].dot(u[:, nc - 1 :])
        termo1 = rs.T
        termo2 = rs
        spe.append(termo1.dot(termo2))
    teta1 = (s_diag[nc - 1 :]).sum()
    teta2 = (s_diag[nc - 1 :] ** 2).sum()
    teta3 = (s_diag[nc:-1, :] ** 3).sum()
    h0 = 1 - (2 * teta1 * teta3) / (3 * teta2**2)
    mu = 0.145462645553
    vals = invgauss.ppf([0, 0.999], mu)
    ca = invgauss.cdf(vals, mu)[1]
    spe_lim = teta1 * (
        (h0 * ca * np.sqrt(2 * teta2) / teta1)
        + 1
        + (teta2 * h0 * (h0 - 1)) / (teta1**2)
    ) ** (1 / h0)

    ## PHI
    phi = []
    for i in range(pc.shape[0]):
        phi.append((spe[i] / spe_lim) + (t2[i] / t2_lim))
    gphi = ((nc / t2_lim**2) + (teta2 / spe_lim**2)) / (
        (nc / t2_lim) + (teta1 / spe_lim)
    )
    hphi = ((nc / t2_lim) + (teta1 / spe_lim)) ** 2 / (
        (nc / t2_lim**2) + (teta2 / spe_lim**2)
    )
    phi_lim = gphi * chi2.ppf(0.99, hphi)

    df_t2 = pd.DataFrame(
        {
            "time": time,
            "t2": t2,
            "spe": spe,
            "phi": phi,
        }
    )

    df_t2["t2_lim"] = t2_lim
    df_t2["spe_lim"] = spe_lim
    df_t2["phi_lim"] = phi_lim

    df_t2["t2"] = df_t2["t2"].ewm(alpha=0.01).mean()
    df_t2["spe"] = df_t2["spe"].ewm(alpha=0.01).mean()
    df_t2["phi"] = df_t2["phi"].ewm(alpha=0.01).mean()

    # Verificar o último valor do timestamp para ajustar o end_date
    last_timestamp = df_t2["time"].max()

    # Se o end_date fornecido for maior que o último timestamp, ajustar o end_date para o último valor disponível
    if end_date > last_timestamp:
        end_date = last_timestamp

    # Buscar anomalias
    anomalias_filt = get_anomalies(process_id, start_date, end_date)

    # Criar o gráfico com as faixas de anomalias
    fig = go.Figure()

    # Adicionar as linhas de "PHI" e "PHI Limite"
    fig.add_trace(go.Scatter(x=df_t2["time"], y=df_t2["phi"], mode="lines", name="PHI"))
    fig.add_trace(go.Scatter(x=df_t2["time"], y=df_t2["phi_lim"], mode="lines", name="PHI Limite"))

    # Adicionar uma forma fictícia para a legenda "Anomalia Real" (aparece apenas uma vez na legenda)
    fig.add_shape(
        type="rect",
        x0=0, x1=0,
        y0=0, y1=0,
        fillcolor="red",
        opacity=0.2,
        line_width=0,
        layer="below",
        showlegend=True,
        name="Anomalia Real"  # Adiciona uma entrada única na legenda
    )

    # Adicionar faixas de anomalias reais no gráfico
    for _, row in anomalias_filt.iterrows():
        fig.add_shape(
            type="rect",
            x0=row["start_date"],
            x1=row["end_date"],
            y0=0,
            y1=df_t2["phi"].max() * 4,
            fillcolor="red",
            opacity=0.2,
            line_width=0,
            layer="below"
        )

    # Ajustar o eixo X para refletir o intervalo de datas correto
    fig.update_xaxes(range=[start_date, end_date])

    return df_t2, fig

def evaluate_model_accuracy_with_feedback(df_t2, df_anomalias, threshold):
    """
    Avalia o percentual de acerto do modelo de detecção de anomalias e cria um DataFrame com a avaliação.
    """

    total_acertos = 0
    total_erros = 0
    total_anomalias = len(df_anomalias)

    # Lista para armazenar o status das anomalias
    result_list = []

    # Para cada anomalia real, verificar se o modelo acertou
    for _, row in df_anomalias.iterrows():
        start_date = row["start_date"]
        end_date = row["end_date"]
        real_anomaly = row["real_anomaly"]

        # Verificar se o gráfico ultrapassou o limite (threshold) durante o período da anomalia
        mask = (df_t2["time"] >= start_date) & (df_t2["time"] <= end_date)
        period_data = df_t2[mask]

        # Verificar se o modelo detectou a anomalia corretamente
        detected_anomaly = (period_data["phi"] > threshold).any()

        # Caso 1: A anomalia é verdadeira e o modelo detectou corretamente
        if real_anomaly == 'true' and detected_anomaly:
            total_acertos += 1
            result_list.append({
                "start_date": start_date,
                "end_date": end_date,
                "real_anomaly": real_anomaly,
                "status": "Correto"
            })

        # Caso 2: A anomalia é falsa e o modelo NÃO detectou
        elif real_anomaly == 'false' and not detected_anomaly:
            total_acertos += 1
            result_list.append({
                "start_date": start_date,
                "end_date": end_date,
                "real_anomaly": real_anomaly,
                "status": "Correto"
            })

        # Caso 3: A anomalia é verdadeira, mas o modelo NÃO detectou
        elif real_anomaly == 'true' and not detected_anomaly:
            total_erros += 1
            result_list.append({
                "start_date": start_date,
                "end_date": end_date,
                "real_anomaly": real_anomaly,
                "status": "Incorreto"
            })

        # Caso 4: A anomalia é falsa, mas o modelo detectou
        elif real_anomaly == 'false' and detected_anomaly:
            total_erros += 1
            result_list.append({
                "start_date": start_date,
                "end_date": end_date,
                "real_anomaly": real_anomaly,
                "status": "Incorreto"
            })

    # Calcular o percentual de acertos
    acertos_percentual = (total_acertos / total_anomalias) * 100 if total_anomalias > 0 else 0.0

    # Criar o DataFrame com os resultados
    df_result = pd.DataFrame(result_list)

    return acertos_percentual, df_result


def dev_graph_predict(phi, eixo_x, threshold, process_id, start_date, end_date, freq="1T", list_periods=None, plot_anomalies=True):
    """
    Função para gerar o gráfico de detecção de anomalias e retornar o DataFrame `df_t2` com os dados de detecção.
    """

    # Criar o dataframe com os dados de phi e timestamps
    df = pd.DataFrame({"phi": phi, "timestamp": eixo_x})

    # Converter os timestamps para datetime e configurar o índice
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.set_index("timestamp")
    df = df.resample(freq).asfreq()
    df = df["phi"].fillna(0)
    df = df.reset_index()

    df["threshold"] = threshold

    # Criar o DataFrame df_t2 com as colunas 'time' e 'phi'
    df_t2 = pd.DataFrame({
        "time": df["timestamp"],  # Usando a coluna de timestamps
        "phi": df["phi"]  # Usando os valores de phi da predição
    })

    # Criar o gráfico
    layout = go.Layout(plot_bgcolor="#FFF")
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["phi"], mode="lines", name="Índice", fill="tozeroy", line_color="#0F293A"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["threshold"], mode="lines", name="Limiar", line_color="#FB8102"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#CECFD1')
    fig.update_layout(hovermode='x unified', legend=dict(orientation="h"))
    fig.update_layout(yaxis_range=[0, 4 * threshold], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    # Ajustar end_date se for maior que o último timestamp
    last_timestamp = df["timestamp"].max()
    if end_date > last_timestamp:
        end_date = last_timestamp

    # Adicionar faixas de anomalias, se habilitado
    if plot_anomalies:
        anomalias_filt = get_anomalies(process_id, start_date, end_date)

        # Adicionar legendas fictícias
        fig.add_shape(type="rect", x0=(list_periods[0]["date_ini"]), x1=(list_periods[0]["date_ini"]), y0=0, y1=0, fillcolor="green", opacity=0.2,
                      line_width=0, layer="below", showlegend=True, name="Anomalia Real")
        fig.add_shape(type="rect", x0=(list_periods[0]["date_ini"]), x1=(list_periods[0]["date_ini"]), y0=0, y1=0, fillcolor="red", opacity=0.2,
                      line_width=0, layer="below", showlegend=True, name="Anomalia Falsa")

        # Plotar as anomalias
        for _, row in anomalias_filt.iterrows():
            color = 'green' if row["real_anomaly"] == 'true' else 'red'
            fig.add_shape(
                type="rect",
                x0=row["start_date"],
                x1=row["end_date"],
                y0=0,
                y1=df["phi"].max() * 4,
                fillcolor=color,
                opacity=0.2,
                line_width=0,
                layer="below"
            )

    # Adicionar faixas de períodos desligados/transitórios
    if list_periods and list_periods != []:
        fig.add_shape(showlegend=True, type="rect", x0=list_periods[0]["date_ini"], y0=0,
                      x1=list_periods[0]["date_ini"], y1=0, fillcolor='#68cbf8',
                      opacity=1, line_width=0, layer="below", name="Desligado")
        fig.add_shape(showlegend=True, type="rect", x0=list_periods[0]["date_ini"], y0=0,
                      x1=list_periods[0]["date_ini"], y1=0, fillcolor='#b3e5fc',
                      opacity=1, line_width=0, layer="below", name="Transitório")
        for periodo in list_periods:
            if periodo["type"] == "desligado":
                color = '#68cbf8'
            elif periodo["type"] == "transitorio":
                color = '#b3e5fc'
            else:
                color = "red"

            fig.add_shape(
                type="rect",
                x0=periodo["date_ini"],
                y0=0,
                x1=periodo["date_end"],
                y1=df["phi"].max() * 4,
                fillcolor=color,
                opacity=1,
                line_width=0,
                layer="below"
            )

    fig.update_xaxes(range=[start_date, end_date])

    return fig, df_t2

def get_process_id(process_name):
    # Nome do arquivo Excel local
    file_name = "processos.xlsx"

    try:
        # Ler o arquivo Excel
        df_process = pd.read_excel(file_name)

        # Consultar o process_id correspondente
        process_id = df_process.loc[df_process['process_name'] == process_name, 'process_id'].values[0]
    
    except (FileNotFoundError, IndexError):
        # Se o arquivo não for encontrado ou o processo não estiver no arquivo
        process_id = None
    
    return process_id
