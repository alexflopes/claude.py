import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytz
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pygame
from scipy.stats import norm, pearsonr
import time
import sqlite3
from plotly.subplots import make_subplots
import talib
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(layout="wide", page_title="Plataforma de Análise B3")

# Inicialização do banco de dados
def inicializar_bd():
    conn = sqlite3.connect('dados_trading.db')
    c = conn.cursor()
    
    # Criar tabela para armazenar dados históricos
    c.execute('''
    CREATE TABLE IF NOT EXISTS dados_historicos (
        timestamp TEXT,
        ativo TEXT,
        timeframe TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        PRIMARY KEY (timestamp, ativo, timeframe)
    )
    ''')
    
    # Criar tabela para zonas de suporte e resistência
    c.execute('''
    CREATE TABLE IF NOT EXISTS zonas_sr (
        ativo TEXT,
        preco REAL,
        tipo TEXT,
        forca INTEGER,
        data_criacao TEXT,
        volume INTEGER,
        PRIMARY KEY (ativo, preco)
    )
    ''')
    
    conn.commit()
    conn.close()

# Inicializa o pygame para tocar alerta
pygame.mixer.init()

def tocar_alerta_mp3(arquivo='alerta.mp3'):
    try:
        pygame.mixer.music.load(arquivo)
        pygame.mixer.music.play()
    except Exception as e:
        st.error(f"Erro ao tocar alerta: {e}")

# Conexão com o MetaTrader 5
@st.cache_resource
def conectar_mt5():
    if not mt5.initialize():
        st.error(f"Falha ao inicializar MT5: {mt5.last_error()}")
        return False
    return True

# Função para obter dados do MetaTrader 5
def obter_dados_mt5(ativo, timeframe=mt5.TIMEFRAME_M1, num_barras=100):
    """Obter dados históricos do MT5"""
    if not mt5.symbol_select(ativo, True):
        st.error(f"Falha ao selecionar ativo {ativo}: {mt5.last_error()}")
        return pd.DataFrame()
    
    rates = mt5.copy_rates_from_pos(ativo, timeframe, 0, num_barras)
    if rates is None or len(rates) == 0:
        st.error(f"Não foi possível obter dados para {ativo}")
        return pd.DataFrame()
    
    # Converter para DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Salvar no banco de dados
    salvar_dados_historicos(df, ativo, str(timeframe))
    
    return df

def salvar_dados_historicos(df, ativo, timeframe):
    """Salvar dados históricos no banco de dados SQLite"""
    conn = sqlite3.connect('dados_trading.db')
    
    # Preparar os dados para inserção
    df_copy = df.reset_index()
    df_copy['timestamp'] = df_copy['time'].astype(str)
    df_copy['ativo'] = ativo
    df_copy['timeframe'] = timeframe
    
    # Inserir no banco de dados (ignorar duplicatas)
    df_copy[['timestamp', 'ativo', 'timeframe', 'open', 'high', 'low', 'close', 'volume']].to_sql(
        'dados_historicos', conn, if_exists='append', index=False, method='multi')
    
    conn.commit()
    conn.close()

# Detectar zonas de suporte e resistência
def detectar_zonas_suporte_resistencia(df, janela=20, sensibilidade=0.01):
    """
    Detecta zonas de suporte e resistência baseadas em:
    1. Pivôs de preço
    2. Volumes significativos
    3. Retrações de Fibonacci
    """
    zonas = []
    
    # 1. Identificar pivôs
    df['pivot_high'] = df['high'].rolling(window=janela, center=True).apply(
        lambda x: x[len(x)//2] == max(x), raw=True)
    df['pivot_low'] = df['low'].rolling(window=janela, center=True).apply(
        lambda x: x[len(x)//2] == min(x), raw=True)
    
    # Picos de alta (resistências)
    for idx in df[df['pivot_high'] == 1.0].index:
        zonas.append({
            'Preço': df.loc[idx, 'high'],
            'Tipo': 'Resistência',
            'Data': idx,
            'Força': 1,
            'Volume': df.loc[idx, 'volume']
        })
    
    # Vales (suportes)
    for idx in df[df['pivot_low'] == 1.0].index:
        zonas.append({
            'Preço': df.loc[idx, 'low'],
            'Tipo': 'Suporte',
            'Data': idx,
            'Força': 1,
            'Volume': df.loc[idx, 'volume']
        })
    
    # 2. Volumes significativos
    vol_medio = df['volume'].mean()
    vol_std = df['volume'].std()
    vol_threshold = vol_medio + 2 * vol_std  # Volume significativo (2 desvios padrão acima da média)
    
    for idx in df[df['volume'] > vol_threshold].index:
        tipo = 'Suporte' if df.loc[idx, 'close'] > df.loc[idx, 'open'] else 'Resistência'
        zonas.append({
            'Preço': df.loc[idx, 'close'],
            'Tipo': f"{tipo} (Volume)",
            'Data': idx,
            'Força': 2,
            'Volume': df.loc[idx, 'volume']
        })
    
    # 3. Retrações de Fibonacci (do máximo para o mínimo global)
    max_price = df['high'].max()
    min_price = df['low'].min()
    range_price = max_price - min_price
    
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    for level in fib_levels:
        fib_price = max_price - range_price * level
        zonas.append({
            'Preço': fib_price,
            'Tipo': f"Fibonacci {level}",
            'Data': df.index[-1],
            'Força': 3,
            'Volume': 0
        })
    
    # Consolidar zonas próximas
    zonas_df = pd.DataFrame(zonas)
    if zonas_df.empty:
        return pd.DataFrame(columns=['Preço', 'Tipo', 'Data', 'Força', 'Volume'])
    
    # Agrupar zonas próximas
    zonas_df = zonas_df.sort_values('Preço')
    zonas_consolidadas = []
    
    i = 0
    while i < len(zonas_df):
        preco_atual = zonas_df.iloc[i]['Preço']
        forca_acumulada = zonas_df.iloc[i]['Força']
        volume_acumulado = zonas_df.iloc[i]['Volume']
        tipo = zonas_df.iloc[i]['Tipo']
        data = zonas_df.iloc[i]['Data']
        
        j = i + 1
        while j < len(zonas_df) and abs(zonas_df.iloc[j]['Preço'] - preco_atual) < (preco_atual * sensibilidade):
            forca_acumulada += zonas_df.iloc[j]['Força']
            volume_acumulado += zonas_df.iloc[j]['Volume']
            j += 1
        
        zonas_consolidadas.append({
            'Preço': preco_atual,
            'Tipo': tipo,
            'Data': data,
            'Força': forca_acumulada,
            'Volume': volume_acumulado
        })
        
        i = j
    
    return pd.DataFrame(zonas_consolidadas)

# Funções para calcular indicadores técnicos
def calcular_indicadores(df):
    """Calcular indicadores técnicos usando talib"""
    # Adicionar indicadores
    if len(df) > 14:  # Verifica se há dados suficientes
        df['ma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['ma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ma_200'] = talib.SMA(df['close'], timeperiod=200)
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        
        # Identificar padrões de candles
        for pattern in [
            'CDLENGULFING', 'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLDOJI',
            'CDLMORNINGSTAR', 'CDLSHOOTINGSTAR', 'CDLHARAMI'
        ]:
            df[pattern] = getattr(talib, pattern)(df['open'], df['high'], df['low'], df['close'])
    
    return df

# Função para detectar divergências
def detectar_divergencias(df):
    """Detectar divergências entre preço e indicadores"""
    divergencias = []
    
    if 'rsi' not in df.columns or len(df) < 20:
        return divergencias
    
    # Encontrar pivôs locais nos preços
    for i in range(10, len(df) - 10):
        if (df['high'].iloc[i] > df['high'].iloc[i-1:i+1].max() and
            df['high'].iloc[i] > df['high'].iloc[i+1:i+10].max()):
            # Pivô de alta no preço
            
            # Verificar se o RSI está formando uma divergência (mais baixo)
            if df['rsi'].iloc[i] < df['rsi'].iloc[i-10:i].max():
                divergencias.append({
                    'Tipo': 'Divergência Bearish',
                    'Data': df.index[i],
                    'Preço': df['high'].iloc[i],
                    'RSI': df['rsi'].iloc[i]
                })
        
        if (df['low'].iloc[i] < df['low'].iloc[i-10:i].min() and
            df['low'].iloc[i] < df['low'].iloc[i+1:i+10].min()):
            # Pivô de baixa no preço
            
            # Verificar se o RSI está formando uma divergência (mais alto)
            if df['rsi'].iloc[i] > df['rsi'].iloc[i-10:i].min():
                divergencias.append({
                    'Tipo': 'Divergência Bullish',
                    'Data': df.index[i],
                    'Preço': df['low'].iloc[i],
                    'RSI': df['rsi'].iloc[i]
                })
    
    return divergencias

# Funções quantitativas
def dolar_justo(spot, cupom_cambial, taxa_di):
    """Cálculo do dólar justo"""
    return spot * (1 + cupom_cambial) / (1 + taxa_di)

def correlacao(x, y):
    """Correlação de Pearson entre duas séries"""
    return pearsonr(x, y)[0]

def distribuicao_gauss(media, desvio, x):
    """Densidade da distribuição normal"""
    return norm.pdf(x, media, desvio)

def volatilidade(retornos):
    """Volatilidade anualizada dos retornos"""
    return np.std(retornos) * np.sqrt(252)  # Anualizada assumindo 252 dias úteis

def black_scholes(S, K, T, r, sigma):
    """Modelo Black-Scholes para precificação de opções"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Valores da opção de compra e venda
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return {
        'call': call, 
        'put': put,
        'delta_call': norm.cdf(d1),
        'delta_put': norm.cdf(d1) - 1,
        'gamma': norm.pdf(d1) / (S * sigma * np.sqrt(T)),
        'vega': S * norm.pdf(d1) * np.sqrt(T) / 100  # Vega em % por 1 ponto de volatilidade
    }

def regressao(x, y):
    """Regressão linear simples"""
    coef = np.polyfit(x, y, 1)
    predict = np.poly1d(coef)
    return {
        'coeficientes': coef,
        'previsao': predict,
        'r_squared': np.corrcoef(x, y)[0, 1] ** 2
    }

def var_normal(retornos, alpha=0.05):
    """Value at Risk com distribuição normal"""
    media = np.mean(retornos)
    sigma = np.std(retornos)
    z = norm.ppf(alpha)
    var = -(media + z * sigma)
    return var

def curva_vol(strikes, vols):
    """Curva de volatilidade para diferentes strikes"""
    curve = dict(zip(strikes, vols))
    
    # Interpolar valores entre strikes
    strike_min = min(strikes)
    strike_max = max(strikes)
    strike_range = np.linspace(strike_min, strike_max, 100)
    
    # Ajustar uma curva polinomial
    coef = np.polyfit(strikes, vols, 2)
    vol_interp = np.poly1d(coef)
    
    curve_interp = dict(zip(strike_range, vol_interp(strike_range)))
    
    return {
        'original': curve,
        'interpolada': curve_interp,
        'coeficientes': coef
    }

def detectar_padroes(df):
    """Detectar padrões gráficos"""
    padroes = []
    
    # Verificar se há colunas de padrões de candles
    candle_patterns = [col for col in df.columns if col.startswith('CDL')]
    for pattern in candle_patterns:
        for idx in df[df[pattern] != 0].index:
            sinal = 'Bullish' if df.loc[idx, pattern] > 0 else 'Bearish'
            nome_padrao = pattern.replace('CDL', '')
            padroes.append({
                'Tipo': f"Candle {nome_padrao} ({sinal})",
                'Data': idx,
                'Preço': df.loc[idx, 'close']
            })
    
    # Detectar padrões de alta e baixa
    # Double top/bottom, head and shoulders, etc.
    # (Esta é uma implementação simplificada)
    
    return padroes

# Função para backtest de estratégias simples
def backtest_estrategia(df, estrategia='cruzamento_medias'):
    """Backtest simples de estratégias"""
    df_bt = df.copy()
    df_bt['retorno'] = df_bt['close'].pct_change()
    
    # Estratégia de cruzamento de médias móveis
    if estrategia == 'cruzamento_medias':
        df_bt['sinal'] = 0
        df_bt['ma_rapida'] = talib.SMA(df_bt['close'], timeperiod=20)
        df_bt['ma_lenta'] = talib.SMA(df_bt['close'], timeperiod=50)
        
        # Gerar sinais (1 = compra, -1 = venda)
        df_bt.loc[df_bt['ma_rapida'] > df_bt['ma_lenta'], 'sinal'] = 1
        df_bt.loc[df_bt['ma_rapida'] < df_bt['ma_lenta'], 'sinal'] = -1
        
        # Calcular retornos da estratégia
        df_bt['retorno_estrategia'] = df_bt['sinal'].shift(1) * df_bt['retorno']
        
        # Calcular equity curve
        df_bt['equity_curve'] = (1 + df_bt['retorno_estrategia']).cumprod()
        
        # Calcular métricas
        total_return = df_bt['equity_curve'].iloc[-1] - 1
        anual_return = (1 + total_return) ** (252 / len(df_bt)) - 1
        sharpe = (df_bt['retorno_estrategia'].mean() / df_bt['retorno_estrategia'].std()) * np.sqrt(252)
        
        return {
            'df': df_bt,
            'total_return': total_return,
            'anual_return': anual_return,
            'sharpe': sharpe
        }
    
    # Outras estratégias podem ser implementadas aqui
    
    return None

# Interface principal
def main():
    inicializar_bd()
    
    if not conectar_mt5():
        st.error("Não foi possível conectar ao MetaTrader 5. Verifique a instalação.")
        return
    
    st.title("📈 Plataforma de Análise - B3")
    
    # Navegação principal
    aba = st.sidebar.radio("Navegação", [
        "📊 Painel de Preço", 
        "📐 Análise Quantitativa",
        "🔍 Backtesting",
        "⚙️ Configurações"
    ])
    
    # Painel de preço
    if aba == "📊 Painel de Preço":
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.header("⚙️ Configurações")
            timeframes_mt5 = {
                "1 minuto": mt5.TIMEFRAME_M1,
                "5 minutos": mt5.TIMEFRAME_M5, 
                "15 minutos": mt5.TIMEFRAME_M15,
                "30 minutos": mt5.TIMEFRAME_M30,
                "1 hora": mt5.TIMEFRAME_H1,
                "4 horas": mt5.TIMEFRAME_H4,
                "Diário": mt5.TIMEFRAME_D1
            }
            tf_selecionado = st.selectbox("Timeframe", list(timeframes_mt5.keys()), index=0)
            tf_mt5 = timeframes_mt5[tf_selecionado]
            
            ativos = ["WIN$N", "WDO$N", "WINV25", "WDOV25", "DOL$"]  # Adicionar mais conforme necessário
            ativo_selecionado = st.selectbox("Ativo", ativos, index=0)
            
            num_barras = st.slider("Número de barras", 50, 500, 100)
            
            tempo_atualizacao = st.slider("Tempo de atualização (segundos)", 1, 60, 5)
            
            mostrar_sr = st.checkbox("Mostrar Suporte/Resistência", value=True)
            mostrar_indicadores = st.checkbox("Mostrar Indicadores", value=True)
            mostrar_padroes = st.checkbox("Mostrar Padrões", value=True)
            mostrar_divergencias = st.checkbox("Mostrar Divergências", value=True)
        
        with col1:
            st.header(f"📈 {ativo_selecionado} - {tf_selecionado}")
            chart_placeholder = st.empty()
            info_placeholder = st.empty()
            
            # Loop de atualização
            while True:
                try:
                    # Obter dados
                    df = obter_dados_mt5(ativo_selecionado, tf_mt5, num_barras)
                    
                    if df.empty:
                        st.error("Não foi possível obter dados. Verifique o ativo e o MetaTrader.")
                        break
                    
                    # Calcular indicadores
                    if mostrar_indicadores:
                        df = calcular_indicadores(df)
                    
                    # Detectar zonas de SR
                    zonas_sr = detectar_zonas_suporte_resistencia(df) if mostrar_sr else pd.DataFrame()
                    
                    # Detectar padrões
                    padroes = detectar_padroes(df) if mostrar_padroes else []
                    
                    # Detectar divergências
                    divergencias = detectar_divergencias(df) if mostrar_divergencias else []
                    
                    # Criar gráfico
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                         row_heights=[0.7, 0.3],
                                         vertical_spacing=0.02)
                    
                    # Adicionar velas
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='Preço'
                    ), row=1, col=1)
                    
                    # Adicionar indicadores
                    if mostrar_indicadores and 'ma_20' in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df.index, 
                            y=df['ma_20'], 
                            name='MA 20',
                            line=dict(color='blue', width=1)
                        ), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=df.index, 
                            y=df['ma_50'], 
                            name='MA 50',
                            line=dict(color='orange', width=1)
                        ), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=df.index, 
                            y=df['ma_200'], 
                            name='MA 200',
                            line=dict(color='red', width=1)
                        ), row=1, col=1)
                        
                        # Adicionar Bandas de Bollinger
                        fig.add_trace(go.Scatter(
                            x=df.index, 
                            y=df['upper_band'], 
                            name='Banda Superior',
                            line=dict(color='rgba(0,128,0,0.3)', width=1),
                            showlegend=False
                        ), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=df.index, 
                            y=df['lower_band'], 
                            name='Banda Inferior',
                            line=dict(color='rgba(0,128,0,0.3)', width=1),
                            fill='tonexty',
                            fillcolor='rgba(0,128,0,0.1)',
                            showlegend=False
                        ), row=1, col=1)
                        
                        # Adicionar RSI
                        fig.add_trace(go.Scatter(
                            x=df.index, 
                            y=df['rsi'], 
                            name='RSI',
                            line=dict(color='purple', width=1)
                        ), row=2, col=1)
                        
                        # Adicionar linhas de referência no RSI
                        fig.add_shape(
                            type="line", line=dict(dash='dash', color='gray'),
                            x0=df.index[0], y0=70, x1=df.index[-1], y1=70,
                            row=2, col=1
                        )
                        
                        fig.add_shape(
                            type="line", line=dict(dash='dash', color='gray'),
                            x0=df.index[0], y0=30, x1=df.index[-1], y1=30,
                            row=2, col=1
                        )
                    
                    # Adicionar zonas de SR
                    if not zonas_sr.empty:
                        preco_atual = df['close'].iloc[-1]
                        
                        for _, zona in zonas_sr.iterrows():
                            fig.add_shape(
                                type="line",
                                x0=df.index[0],
                                y0=zona['Preço'],
                                x1=df.index[-1],
                                y1=zona['Preço'],
                                line=dict(
                                    color="green" if zona['Tipo'].startswith('Suporte') else "red",
                                    width=1 + zona['Força'] // 2,
                                    dash="solid"
                                ),
                                row=1, col=1
                            )
                            
                            # Destacar zonas próximas do preço atual
                            if abs(zona['Preço'] - preco_atual) / preco_atual < 0.01:  # Dentro de 1%
                                tocar_alerta_mp3()
                    
                    # Adicionar padrões encontrados
                    for padrao in padroes:
                        fig.add_trace(go.Scatter(
                            x=[padrao['Data']],
                            y=[padrao['Preço']],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up' if 'Bullish' in padrao['Tipo'] else 'triangle-down',
                                size=10,
                                color='green' if 'Bullish' in padrao['Tipo'] else 'red'
                            ),
                            name=padrao['Tipo'],
                            showlegend=True
                        ), row=1, col=1)
                    
                    # Adicionar divergências
                    for div in divergencias:
                        fig.add_trace(go.Scatter(
                            x=[div['Data']],
                            y=[div['Preço']],
                            mode='markers',
                            marker=dict(
                                symbol='star',
                                size=12,
                                color='green' if 'Bullish' in div['Tipo'] else 'red'
                            ),
                            name=div['Tipo'],
                            showlegend=True
                        ), row=1, col=1)
                    
                    # Configurações do layout
                    fig.update_layout(
                        title=f"{ativo_selecionado} - {tf_selecionado} - Atualizado: {datetime.now().strftime('%H:%M:%S')}",
                        xaxis_rangeslider_visible=False,
                        height=600,
                        yaxis_title="Preço",
                        yaxis2_title="RSI",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    fig.update_yaxes(range=[0, 100], row=2, col=1)
                    
                    # Atualizar gráfico
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar informações adicionais
                    with info_placeholder.container():
                        info_col1, info_col2 = st.columns(2)
                        
                        with info_col1:
                            st.subheader("🔍 Zonas S/R Próximas")
                            if not zonas_sr.empty:
                                st.dataframe(zonas_sr.sort_values('Força', ascending=False), height=200)
                            else:
                                st.write("Nenhuma zona de S/R detectada.")
                        
                        with info_col2:
                            st.subheader("📊 Padrões Detectados")
                            if padroes:
                                st.dataframe(pd.DataFrame(padroes), height=200)
                            else:
                                st.write("Nenhum padrão detectado.")
                    
                    # Esperar pelo tempo de atualização
                    time.sleep(tempo_atualizacao)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    st.error(f"Erro: {e}")
                    time.sleep(10)
    
    # Análise Quantitativa
    elif aba == "📐 Análise Quantitativa":
        st.header("📐 Análise Quantitativa")
        
        # Tabs para diferentes análises
        quant_tab = st.tabs([
            "🏦 Dólar", 
            "📊 Correlação",
            "📉 Volatilidade",
            "💰 Opções",
            "📈 Regressão",
            "⚠️ Risco (VaR)",
            "🔄 Curva de Vol"
        ])
        
        # 1. Análise do Dólar Justo
        with quant_tab[0]:
            st.subheader("🏦 Calculadora de Dólar Justo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                spot = st.number_input("Dólar Spot (R$/US$)", value=5.0, min_value=1.0, max_value=10.0, step=0.01)
                cupom = st.number_input("Cupom Cambial (% a.a.)", value=10.0, min_value=0.0, max_value=20.0, step=0.1) / 100
                taxa = st.number_input("Taxa DI (% a.a.)", value=12.5, min_value=0.0, max_value=20.0, step=0.1) / 100
                
                dolar_calc = dolar_justo(spot, cupom, taxa)
                distorcao = (spot - dolar_calc) / dolar_calc * 100
                
                st.metric(
                    label="Dólar Justo", 
                    value=f"R$ {dolar_calc:.4f}",
                    delta=f"{distorcao:.2f}% {'sobrevalorizado' if distorcao > 0 else 'subvalorizado'}"
                )
            
            with col2:
                # Plotar valores de dólar justo para diferentes cenários
                taxa_range = np.linspace(taxa * 0.7, taxa * 1.3, 10)
                cupom_range = np.linspace(cupom * 0.7, cupom * 1.3, 10)
                
                resultados = []
                for t in taxa_range:
                    for c in cupom_range:
                        dj = dolar_justo(spot, c, t)
                        resultados.append({
                            'Taxa DI (%)': t * 100,
                            'Cupom (%)': c * 100,
                            'Dólar Justo': dj
                        })
                
                df_dolar = pd.DataFrame(resultados)
                
                # Gráfico 3D de sensibilidade
                fig = go.Figure(data=[go.Mesh3d(
                    x=df_dolar['Taxa DI (%)'],
                    y=df_dolar['Cupom (%)'],
                    z=df_dolar['Dólar Justo'],
                    colorscale='Viridis',
                    opacity=0.75,
                    colorbar=dict(title='Dólar Justo')
                )])
                
                fig.update_layout(
                    title='Sensibilidade do Dólar Justo',
                    scene=dict(
                        xaxis_title='Taxa DI (%)',
                        yaxis_title='Cupom Cambial (%)',
                        zaxis_title='Dólar Justo (R$)'
                    ),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # 2. Análise de Correlação
        with quant_tab[1]:
            st.subheader("📊 Análise de Correlação entre Ativos")
            
            # Selecionar ativos para correlação
            ativos_disponiveis = ["WIN$N", "WDO$N", "IBOV", "DOL$", "DI1F25", "PETR4", "VALE3", "ITUB4"]
            
            ativos_correlacao = st.multiselect(
                "Selecione os ativos para correlação", 
                ativos_disponiveis,
                default=["WIN$N", "WDO$N", "DOL$"]
            )
            
            if len(ativos_correlacao) >= 2:
                periodo = st.slider("Período (dias)", 5, 60, 30)
                
                # Obter dados históricos
                dados_correlacao = {}
                for ativo in ativos_correlacao:
                    try:
                        df_ativo = obter_dados_mt5(ativo, mt5.TIMEFRAME_D1, periodo)
                        dados_correlacao[ativo] = df_ativo['close']
                    except:
                        st.warning(f"Não foi possível obter dados para {ativo}")
                
                if len(dados_correlacao) >= 2:
                    df_corr = pd.DataFrame(dados_correlacao)
                    
                    # Normalizar para comparação em gráfico
                    df_norm = df_corr.copy()
                    for col in df_norm.columns:
                        df_norm[col] = df_norm[col] / df_norm[col].iloc[0]
                    
                    # Gráfico de preços normalizados
                    fig_prices = go.Figure()
                    
                    for col in df_norm.columns:
                        fig_prices.add_trace(go.Scatter(
                            x=df_norm.index,
                            y=df_norm[col],
                            mode='lines',
                            name=col
                        ))
                    
                    fig_prices.update_layout(
                        title='Evolução Normalizada dos Preços',
                        xaxis_title='Data',
                        yaxis_title='Preço Normalizado',
                        height=400
                    )
                    
                    st.plotly_chart(fig_prices, use_container_width=True)
                    
                    # Calcular a matriz de correlação
                    corr_matrix = df_corr.pct_change().dropna().corr()
                    
                    # Mapa de calor
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='RdBu',
                        zmin=-1, zmax=1,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate="%{text:.2f}"
                    ))
                    
                    fig_corr.update_layout(
                        title='Matriz de Correlação',
                        height=400
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Interpretação das correlações
                    st.subheader("📝 Interpretação das Correlações")
                    
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            ativo1 = corr_matrix.columns[i]
                            ativo2 = corr_matrix.columns[j]
                            corr_val = corr_matrix.iloc[i, j]
                            
                            tipo_corr = "forte positiva" if corr_val > 0.7 else (
                                "moderada positiva" if corr_val > 0.3 else (
                                "fraca" if corr_val > -0.3 else (
                                "moderada negativa" if corr_val > -0.7 else "forte negativa"
                            )))
                            
                            st.write(f"**{ativo1} x {ativo2}**: Correlação {tipo_corr} ({corr_val:.2f})")
                else:
                    st.error("Selecione pelo menos 2 ativos com dados disponíveis")
            else:
                st.info("Selecione pelo menos 2 ativos para análise de correlação")
        
        # 3. Análise de Volatilidade
        with quant_tab[2]:
            st.subheader("📉 Análise de Volatilidade")
            
            ativo_vol = st.selectbox("Selecione o ativo para análise de volatilidade", ativos_disponiveis)
            
            col1, col2 = st.columns(2)
            
            with col1:
                periodo_vol = st.slider("Período (dias)", 30, 252, 60)
                janela_vol = st.slider("Janela para volatilidade (dias)", 5, 30, 21)
            
            with col2:
                alpha_var = st.slider("Nível de confiança para VaR", 0.9, 0.99, 0.95, 0.01)
                tipo_vol = st.radio("Tipo de volatilidade", ["Histórica", "EWMA", "GARCH"])
            
            try:
                # Obter dados históricos
                df_vol = obter_dados_mt5(ativo_vol, mt5.TIMEFRAME_D1, periodo_vol)
                
                # Calcular retornos
                df_vol['retorno'] = df_vol['close'].pct_change().dropna()
                
                # Calcular volatilidade histórica
                df_vol['vol_hist'] = df_vol['retorno'].rolling(window=janela_vol).std() * np.sqrt(252)
                
                # Calcular VaR
                var_value = var_normal(df_vol['retorno'].dropna(), 1 - alpha_var)
                
                # Mostrar métricas
                vol_anual = volatilidade(df_vol['retorno'].dropna())
                
                st.metric("Volatilidade Anualizada", f"{vol_anual*100:.2f}%")
                st.metric(f"Value at Risk ({alpha_var*100:.0f}%)", f"{var_value*100:.2f}%")
                
                # Gráfico de volatilidade
                fig_vol = make_subplots(rows=2, cols=1, 
                                         shared_xaxes=True,
                                         vertical_spacing=0.1,
                                         row_heights=[0.7, 0.3])
                
                fig_vol.add_trace(
                    go.Scatter(
                        x=df_vol.index, 
                        y=df_vol['close'],
                        name='Preço',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                fig_vol.add_trace(
                    go.Scatter(
                        x=df_vol.index, 
                        y=df_vol['vol_hist'],
                        name='Volatilidade',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
                
                fig_vol.update_layout(
                    title=f'Preço e Volatilidade Histórica - {ativo_vol}',
                    xaxis_title='Data',
                    yaxis_title='Preço',
                    yaxis2_title='Volatilidade (%)',
                    height=500
                )
                
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # Distribuição dos retornos
                fig_dist = go.Figure()
                
                fig_dist.add_trace(go.Histogram(
                    x=df_vol['retorno'].dropna(),
                    nbinsx=30,
                    name='Retornos',
                    histnorm='probability',
                    marker_color='blue',
                    opacity=0.5
                ))
                
                # Adicionar curva normal
                x_range = np.linspace(
                    df_vol['retorno'].min(),
                    df_vol['retorno'].max(),
                    100
                )
                
                fig_dist.add_trace(go.Scatter(
                    x=x_range,
                    y=distribuicao_gauss(
                        df_vol['retorno'].mean(),
                        df_vol['retorno'].std(),
                        x_range
                    ),
                    mode='lines',
                    name='Normal',
                    line=dict(color='red')
                ))
                
                # Adicionar linha no VaR
                fig_dist.add_shape(
                    type="line",
                    x0=var_value, y0=0,
                    x1=var_value, y1=distribuicao_gauss(
                        df_vol['retorno'].mean(),
                        df_vol['retorno'].std(),
                        var_value
                    ) * 1.5,
                    line=dict(color="red", width=2, dash="dash")
                )
                
                fig_dist.add_annotation(
                    x=var_value,
                    y=0.05,
                    text=f"VaR {alpha_var*100:.0f}%: {var_value*100:.2f}%",
                    showarrow=True,
                    arrowhead=1
                )
                
                fig_dist.update_layout(
                    title='Distribuição dos Retornos Diários',
                    xaxis_title='Retorno',
                    yaxis_title='Probabilidade',
                    height=400
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erro ao calcular volatilidade: {e}")
        
        # 4. Calculadora de Opções (Black-Scholes)
        with quant_tab[3]:
            st.subheader("💰 Calculadora de Opções (Black-Scholes)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                S = st.number_input("Preço do Ativo (S)", value=115000.0, step=100.0)
                K = st.number_input("Preço de Exercício (K)", value=115000.0, step=100.0)
                T = st.number_input("Tempo até o vencimento (em anos)", value=0.08, min_value=0.01, max_value=2.0, step=0.01)
                r = st.number_input("Taxa livre de risco (% a.a.)", value=12.5, min_value=0.0, max_value=20.0, step=0.1) / 100
                sigma = st.number_input("Volatilidade implícita (% a.a.)", value=20.0, min_value=5.0, max_value=100.0, step=0.5) / 100
                
                # Calcular os resultados
                opcao = black_scholes(S, K, T, r, sigma)
                
                st.write("### Resultados:")
                st.metric("Preço Call", f"{opcao['call']:.2f}")
                st.metric("Preço Put", f"{opcao['put']:.2f}")
                
                st.write("### Gregas:")
                st.metric("Delta Call", f"{opcao['delta_call']:.4f}")
                st.metric("Delta Put", f"{opcao['delta_put']:.4f}")
                st.metric("Gamma", f"{opcao['gamma']:.6f}")
                st.metric("Vega", f"{opcao['vega']:.4f}")
            
            with col2:
                # Análise de sensibilidade
                st.write("### Análise de Sensibilidade")
                
                tipo_sensibilidade = st.selectbox(
                    "Parâmetro para análise de sensibilidade",
                    ["Preço do Ativo", "Volatilidade", "Tempo"]
                )
                
                if tipo_sensibilidade == "Preço do Ativo":
                    precos = np.linspace(S * 0.8, S * 1.2, 50)
                    valores_call = [black_scholes(p, K, T, r, sigma)['call'] for p in precos]
                    valores_put = [black_scholes(p, K, T, r, sigma)['put'] for p in precos]
                    x_axis = precos
                    x_title = "Preço do Ativo"
                
                elif tipo_sensibilidade == "Volatilidade":
                    vols = np.linspace(sigma * 0.5, sigma * 2, 50)
                    valores_call = [black_scholes(S, K, T, r, v)['call'] for v in vols]
                    valores_put = [black_scholes(S, K, T, r, v)['put'] for v in vols]
                    x_axis = vols * 100  # Converter para percentual
                    x_title = "Volatilidade (%)"
                
                else:  # Tempo
                    tempos = np.linspace(max(0.01, T - 0.1), T + 0.2, 50)
                    valores_call = [black_scholes(S, K, t, r, sigma)['call'] for t in tempos]
                    valores_put = [black_scholes(S, K, t, r, sigma)['put'] for t in tempos]
                    x_axis = tempos * 365  # Converter para dias
                    x_title = "Dias até o vencimento"
                
                fig_sens = go.Figure()
                
                fig_sens.add_trace(go.Scatter(
                    x=x_axis,
                    y=valores_call,
                    mode='lines',
                    name='Call',
                    line=dict(color='green')
                ))
                
                fig_sens.add_trace(go.Scatter(
                    x=x_axis,
                    y=valores_put,
                    mode='lines',
                    name='Put',
                    line=dict(color='red')
                ))
                
                fig_sens.update_layout(
                    title=f'Sensibilidade a {tipo_sensibilidade}',
                    xaxis_title=x_title,
                    yaxis_title='Preço da Opção',
                    height=400
                )
                
                st.plotly_chart(fig_sens, use_container_width=True)
        
        # 5. Regressão Linear
        with quant_tab[4]:
            st.subheader("📈 Análise de Regressão")
            
            # Selecionar variáveis para regressão
            var_x = st.selectbox("Variável X (independente)", ativos_disponiveis, index=2)  # IBOV
            var_y = st.selectbox("Variável Y (dependente)", ativos_disponiveis, index=0)    # WIN
            
            if var_x != var_y:
                try:
                    periodo_reg = st.slider("Período para regressão (dias)", 30, 252, 60)
                    
                    # Obter dados
                    df_x = obter_dados_mt5(var_x, mt5.TIMEFRAME_D1, periodo_reg)
                    df_y = obter_dados_mt5(var_y, mt5.TIMEFRAME_D1, periodo_reg)
                    
                    # Alinhar os dados
                    df_reg = pd.DataFrame({
                        'x': df_x['close'],
                        'y': df_y['close']
                    })
                    
                    df_reg = df_reg.dropna()
                    
                    if len(df_reg) > 5:  # Verificar se há dados suficientes
                        # Calcular regressão
                        resultado = regressao(df_reg['x'], df_reg['y'])
                        
                        st.write(f"### Resultados da Regressão Linear entre {var_x} e {var_y}")
                        st.write(f"Equação: y = {resultado['coeficientes'][0]:.4f}x + {resultado['coeficientes'][1]:.4f}")
                        st.write(f"R² = {resultado['r_squared']:.4f}")
                        
                        # Plotar dados e linha de regressão
                        fig_reg = go.Figure()
                        
                        # Pontos originais
                        fig_reg.add_trace(go.Scatter(
                            x=df_reg['x'],
                            y=df_reg['y'],
                            mode='markers',
                            name='Dados',
                            marker=dict(color='blue')
                        ))
                        
                        # Linha de regressão
                        x_range = np.linspace(df_reg['x'].min(), df_reg['x'].max(), 100)
                        y_pred = resultado['previsao'](x_range)
                        
                        fig_reg.add_trace(go.Scatter(
                            x=x_range,
                            y=y_pred,
                            mode='lines',
                            name='Regressão',
                            line=dict(color='red')
                        ))
                        
                        fig_reg.update_layout(
                            title=f'Regressão Linear: {var_y} vs {var_x}',
                            xaxis_title=var_x,
                            yaxis_title=var_y,
                            height=500
                        )
                        
                        st.plotly_chart(fig_reg, use_container_width=True)
                        
                        # Previsão para valores futuros
                        st.subheader("🔮 Previsão")
                        
                        valor_x = st.number_input(
                            f"Valor futuro para {var_x}",
                            value=float(df_reg['x'].iloc[-1]),
                            step=100.0
                        )
                        
                        previsao_y = resultado['previsao'](valor_x)
                        
                        st.metric(
                            f"Previsão para {var_y}",
                            f"{previsao_y:.2f}",
                            f"{(previsao_y - df_reg['y'].iloc[-1]) / df_reg['y'].iloc[-1] * 100:.2f}%"
                        )
                        
                    else:
                        st.error("Dados insuficientes para regressão")
                        
                except Exception as e:
                    st.error(f"Erro ao calcular regressão: {e}")
            else:
                st.error("Selecione variáveis diferentes para X e Y")
        
        # 6. Análise de Risco (VaR)
        with quant_tab[5]:
            st.subheader("⚠️ Análise de Value at Risk (VaR)")
            
            # Selecionar ativos para carteira
            ativos_var = st.multiselect(
                "Selecione os ativos para a carteira", 
                ativos_disponiveis,
                default=["WIN$N", "WDO$N"]
            )
            
            if len(ativos_var) > 0:
                # Configurações
                col1, col2 = st.columns(2)
                
                with col1:
                    periodo_var = st.slider("Período histórico (dias)", 30, 252, 60)
                    nivel_confianca = st.slider("Nível de confiança", 0.9, 0.99, 0.95, 0.01)
                
                with col2:
                    metodo_var = st.radio("Método de cálculo", ["Histórico", "Paramétrico", "Monte Carlo"])
                    horizonte = st.slider("Horizonte de tempo (dias)", 1, 21, 1)
                
                # Obter dados históricos
                dados_var = {}
                pesos = {}
                
                for ativo in ativos_var:
                    try:
                        df_ativo = obter_dados_mt5(ativo, mt5.TIMEFRAME_D1, periodo_var)
                        dados_var[ativo] = df_ativo['close']
                        pesos[ativo] = 1.0 / len(ativos_var)  # Pesos iguais inicialmente
                    except:
                        st.warning(f"Não foi possível obter dados para {ativo}")
                
                if len(dados_var) > 0:
                    # Ajustar pesos
                    st.write("### Pesos dos Ativos na Carteira")
                    
                    novos_pesos = {}
                    col_pesos = st.columns(len(dados_var))
                    
                    for i, ativo in enumerate(dados_var.keys()):
                        with col_pesos[i]:
                            novos_pesos[ativo] = st.slider(
                                f"Peso de {ativo}", 
                                0.0, 1.0, pesos[ativo], 0.05
                            )
                    
                    # Normalizar pesos
                    soma_pesos = sum(novos_pesos.values())
                    if soma_pesos > 0:
                        for ativo in novos_pesos:
                            novos_pesos[ativo] /= soma_pesos
                    
                    # Calcular retornos
                    df_retornos = pd.DataFrame(dados_var).pct_change().dropna()
                    
                    # Calcular VaR
                    var_result = None
                    
                    if metodo_var == "Paramétrico":
                        # Método paramétrico (distribuição normal)
                        retornos_carteira = np.zeros(len(df_retornos))
                        for ativo, peso in novos_pesos.items():
                            retornos_carteira += df_retornos[ativo].values * peso
                        
                        var_result = var_normal(retornos_carteira, 1 - nivel_confianca) * np.sqrt(horizonte)
                    
                    elif metodo_var == "Histórico":
                        # Método histórico
                        retornos_carteira = np.zeros(len(df_retornos))
                        for ativo, peso in novos_pesos.items():
                            retornos_carteira += df_retornos[ativo].values * peso
                        
                        var_result = np.percentile(retornos_carteira, (1 - nivel_confianca) * 100) * np.sqrt(horizonte)
                    
                    else:  # Monte Carlo
                        # Método Monte Carlo
                        n_simulacoes = 10000
                        matriz_cov = df_retornos.cov()
                        media = df_retornos.mean().values
                        
                        # Simulação de Monte Carlo
                        np.random.seed(42)
                        simulacoes = np.random.multivariate_normal(
                            media, 
                            matriz_cov.values,
                            size=n_simulacoes
                        )
                        
                        # Calcular retornos da carteira para cada simulação
                        retornos_simulados = np.zeros(n_simulacoes)
                        for i, ativo in enumerate(novos_pesos.keys()):
                            retornos_simulados += simulacoes[:, i] * novos_pesos[ativo]
                        
                        # Calcular VaR
                        var_result = np.percentile(retornos_simulados, (1 - nivel_confianca) * 100) * np.sqrt(horizonte)
                    
                    if var_result is not None:
                        # Mostrar resultados
                        valor_carteira = st.number_input("Valor total da carteira (R$)", value=100000.0, step=10000.0)
                        
                        var_percentual = var_result
                        var_absoluto = valor_carteira * var_result
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                f"VaR {nivel_confianca*100:.1f}% ({horizonte} dia{'s' if horizonte > 1 else ''})",
                                f"{var_percentual*100:.2f}%",
                                delta=None
                            )
                        
                        with col2:
                            st.metric(
                                "VaR em R$",
                                f"R$ {var_absoluto:.2f}",
                                delta=None
                            )
                        
                        # Visualizar distribuição dos retornos e VaR
                        retornos_carteira = np.zeros(len(df_retornos))
                        for ativo, peso in novos_pesos.items():
                            retornos_carteira += df_retornos[ativo].values * peso
                        
                        fig_var = go.Figure()
                        
                        fig_var.add_trace(go.Histogram(
                            x=retornos_carteira,
                            nbinsx=30,
                            name='Retornos',
                            histnorm='probability',
                            marker_color='blue',
                            opacity=0.5
                        ))
                        
                        # Adicionar linha no VaR
                        fig_var.add_shape(
                            type="line",
                            x0=var_result / np.sqrt(horizonte), y0=0,
                            x1=var_result / np.sqrt(horizonte), y1=0.3,
                            line=dict(color="red", width=2, dash="dash")
                        )
                        
                        fig_var.add_annotation(
                            x=var_result / np.sqrt(horizonte),
                            y=0.05,
                            text=f"VaR {nivel_confianca*100:.1f}%: {var_result*100:.2f}%",
                            showarrow=True,
                            arrowhead=1
                        )
                        
                        fig_var.update_layout(
                            title='Distribuição dos Retornos da Carteira',
                            xaxis_title='Retorno',
                            yaxis_title='Probabilidade',
                            height=400
                        )
                        
                        st.plotly_chart(fig_var, use_container_width=True)
                else:
                    st.error("Não foi possível obter dados para os ativos selecionados")
            else:
                st.info("Selecione pelo menos um ativo para a carteira")
        
        # 7. Curva de Volatilidade
        with quant_tab[6]:
            st.subheader("🔄 Curva de Volatilidade")
            
            col1, col2 = st.columns(2)
            
            with col1:
                strikes = st.text_input(
                    "Strikes (separados por vírgula)",
                    value="105000, 110000, 115000, 120000, 125000"
                )
                
                try:
                    strikes_list = [float(x.strip())