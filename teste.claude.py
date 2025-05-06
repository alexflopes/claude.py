"""
Implementação das funções para a plataforma de trading
Autor: Claude
Data: 06/05/2025
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN
from scipy.stats import norm


def obter_dados_mt5(ativo, timeframe, n_candles=1000):
    """
    Função para obter dados históricos do MetaTrader5
    
    Parâmetros:
    - ativo: símbolo do ativo (ex: 'WIN$', 'WDO$')
    - timeframe: período dos candles (mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, etc)
    - n_candles: quantidade de candles a serem obtidos
    
    Retorno:
    - DataFrame pandas com os dados
    """
    # Inicializar conexão com MT5 se ainda não estiver conectado
    if not mt5.initialize():
        print(f"Falha ao inicializar o MetaTrader5. Erro: {mt5.last_error()}")
        return None
    
    # Definição do timezone (horário de Brasília)
    timezone = pytz.timezone("America/Sao_Paulo")
    
    # Data atual em UTC
    utc_now = datetime.now(pytz.utc)
    
    # Converter para o timezone local
    now = utc_now.astimezone(timezone)
    
    # Obter os dados do MT5
    rates = mt5.copy_rates_from(ativo, timeframe, now, n_candles)
    
    # Verificar se houve erro na obtenção dos dados
    if rates is None or len(rates) == 0:
        print(f"Falha ao obter dados do {ativo}. Erro: {mt5.last_error()}")
        return None
    
    # Converter para DataFrame
    df = pd.DataFrame(rates)
    
    # Converter timestamp para datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Renomear colunas para padrão português
    df.rename(columns={
        'time': 'data',
        'open': 'abertura',
        'high': 'maxima',
        'low': 'minima',
        'close': 'fechamento',
        'tick_volume': 'volume',
        'spread': 'spread',
        'real_volume': 'volume_real'
    }, inplace=True)
    
    # Definir a coluna de data como índice
    df.set_index('data', inplace=True)
    
    # Retornar o DataFrame
    return df


def detectar_zonas_suporte_resistencia(df, min_pts=5, janela=20, eps=50.0, sensibilidade=0.8):
    """
    Detecta zonas de suporte e resistência usando DBSCAN para agrupar extremos locais
    
    Parâmetros:
    - df: DataFrame com dados de preços
    - min_pts: mínimo de pontos para formar uma zona
    - janela: tamanho da janela para encontrar extremos locais
    - eps: parâmetro de distância para o DBSCAN
    - sensibilidade: fator para ajustar a sensibilidade da detecção
    
    Retorno:
    - dict com zonas de suporte e resistência
    """
    # Cópia do DataFrame para manipulação
    data = df.copy()
    
    # Encontrar extremos locais (máximos e mínimos)
    maximas_idx = argrelextrema(data['maxima'].values, np.greater_equal, order=janela)[0]
    minimas_idx = argrelextrema(data['minima'].values, np.less_equal, order=janela)[0]
    
    # Criar arrays para armazenar os preços extremos
    maximas_precos = data['maxima'].iloc[maximas_idx].values
    minimas_precos = data['minima'].iloc[minimas_idx].values
    
    # Ajustar o eps baseado na volatilidade do ativo
    volatilidade = (data['maxima'] - data['minima']).mean()
    eps_ajustado = eps * (volatilidade / 100) * sensibilidade
    
    # Zonas de resistência (baseadas em máximos)
    zonas_resistencia = {}
    if len(maximas_precos) > 0:
        # Reshape para formato requerido pelo DBSCAN
        maximas_array = maximas_precos.reshape(-1, 1)
        
        # Aplicar DBSCAN para clustering
        clustering = DBSCAN(eps=eps_ajustado, min_samples=min_pts).fit(maximas_array)
        
        # Extrair os clusters (ignore cluster -1, que são outliers)
        labels = clustering.labels_
        clusters_unicos = np.unique(labels)
        
        # Para cada cluster, calcular o preço médio (zona)
        for cluster in clusters_unicos:
            if cluster != -1:  # Ignorar outliers
                cluster_pontos = maximas_array[labels == cluster]
                preco_medio = np.mean(cluster_pontos)
                forca = len(cluster_pontos)  # Quantidade de toques na zona
                zonas_resistencia[preco_medio] = {
                    'forca': forca,
                    'nivel': float(preco_medio),
                    'toques': len(cluster_pontos),
                    'tipo': 'resistencia'
                }
    
    # Zonas de suporte (baseadas em mínimos)
    zonas_suporte = {}
    if len(minimas_precos) > 0:
        # Reshape para formato requerido pelo DBSCAN
        minimas_array = minimas_precos.reshape(-1, 1)
        
        # Aplicar DBSCAN para clustering
        clustering = DBSCAN(eps=eps_ajustado, min_samples=min_pts).fit(minimas_array)
        
        # Extrair os clusters (ignore cluster -1, que são outliers)
        labels = clustering.labels_
        clusters_unicos = np.unique(labels)
        
        # Para cada cluster, calcular o preço médio (zona)
        for cluster in clusters_unicos:
            if cluster != -1:  # Ignorar outliers
                cluster_pontos = minimas_array[labels == cluster]
                preco_medio = np.mean(cluster_pontos)
                forca = len(cluster_pontos)  # Quantidade de toques na zona
                zonas_suporte[preco_medio] = {
                    'forca': forca,
                    'nivel': float(preco_medio),
                    'toques': len(cluster_pontos),
                    'tipo': 'suporte'
                }
    
    # Combinar zonas
    todas_zonas = {**zonas_suporte, **zonas_resistencia}
    
    return todas_zonas


def calcular_confluencias(df, zonas_sr, indicadores):
    """
    Calcula confluências entre zonas de S/R e indicadores técnicos
    
    Parâmetros:
    - df: DataFrame com dados de preços
    - zonas_sr: dicionário com zonas de suporte e resistência
    - indicadores: dicionário com indicadores técnicos calculados
    
    Retorno:
    - dict com confluências detectadas
    """
    confluencias = {}
    preco_atual = df['fechamento'].iloc[-1]
    
    # Margem de tolerância para confluência (em pontos)
    tolerancia = 50
    
    # Verificar confluências com médias móveis
    for periodo, valor in indicadores.get('medias_moveis', {}).items():
        # Último valor da média móvel
        mm_valor = valor.iloc[-1] if not pd.isna(valor.iloc[-1]) else 0
        
        # Verificar confluências com zonas de S/R
        for nivel, zona in zonas_sr.items():
            if abs(nivel - mm_valor) <= tolerancia:
                if nivel not in confluencias:
                    confluencias[nivel] = {
                        'tipo': zona['tipo'],
                        'confluencias': []
                    }
                confluencias[nivel]['confluencias'].append(f"Média Móvel {periodo}")
    
    # Verificar confluências com Bandas de Bollinger
    if 'bollinger' in indicadores:
        bb = indicadores['bollinger']
        for nivel, zona in zonas_sr.items():
            # Confluência com banda superior
            if bb and 'upper' in bb and abs(nivel - bb['upper'].iloc[-1]) <= tolerancia:
                if nivel not in confluencias:
                    confluencias[nivel] = {
                        'tipo': zona['tipo'],
                        'confluencias': []
                    }
                confluencias[nivel]['confluencias'].append("Banda Superior de Bollinger")
            
            # Confluência com banda inferior
            if bb and 'lower' in bb and abs(nivel - bb['lower'].iloc[-1]) <= tolerancia:
                if nivel not in confluencias:
                    confluencias[nivel] = {
                        'tipo': zona['tipo'],
                        'confluencias': []
                    }
                confluencias[nivel]['confluencias'].append("Banda Inferior de Bollinger")
    
    # Verificar confluências com níveis de Fibonacci
    if 'fibonacci' in indicadores:
        fib = indicadores['fibonacci']
        for nivel_fib, valor_fib in fib.items():
            for nivel, zona in zonas_sr.items():
                if abs(nivel - valor_fib) <= tolerancia:
                    if nivel not in confluencias:
                        confluencias[nivel] = {
                            'tipo': zona['tipo'],
                            'confluencias': []
                        }
                    confluencias[nivel]['confluencias'].append(f"Fibonacci {nivel_fib}")
    
    # Adicionar informação de preço próximo
    for nivel, info in confluencias.items():
        # Calcular distância do preço atual até o nível
        distancia = abs(preco_atual - nivel)
        info['distancia_preco'] = distancia
        
        # Verificar se o preço está próximo (menos de 100 pontos)
        if distancia <= 100:
            info['confluencias'].append("Preço Atual Próximo")
    
    return confluencias


def detectar_padrao_grafico(df, janela=20):
    """
    Detecta padrões gráficos comuns como triângulos, bandeiras, etc.
    
    Parâmetros:
    - df: DataFrame com dados de preços
    - janela: tamanho da janela para análise de padrões
    
    Retorno:
    - dict com padrões detectados
    """
    padroes = {}
    
    # Análise apenas se tiver dados suficientes
    if len(df) < janela:
        return padroes
    
    # Últimos N candles para análise
    dados_recentes = df.tail(janela)
    
    # Detectar Triângulo Ascendente
    # (mínimas em tendência de alta e máximas relativamente estáveis)
    minimos = dados_recentes['minima'].values
    maximos = dados_recentes['maxima'].values
    
    # Calcular tendência linear dos mínimos e máximos
    x = np.arange(len(minimos))
    coef_min = np.polyfit(x, minimos, 1)[0]
    coef_max = np.polyfit(x, maximos, 1)[0]
    
    # Triângulo Ascendente: mínimos em alta, máximos estáveis
    if coef_min > 0.3 and abs(coef_max) < 0.1:
        padroes['triangulo_ascendente'] = {
            'confianca': min(1.0, abs(coef_min * 10)),
            'descricao': 'Triângulo Ascendente - Potencial movimento de alta'
        }
    
    # Triângulo Descendente: máximos em queda, mínimos estáveis
    elif coef_max < -0.3 and abs(coef_min) < 0.1:
        padroes['triangulo_descendente'] = {
            'confianca': min(1.0, abs(coef_max * 10)),
            'descricao': 'Triângulo Descendente - Potencial movimento de baixa'
        }
    
    # Triângulo Simétrico: máximos em queda e mínimos em alta
    elif coef_max < -0.2 and coef_min > 0.2:
        padroes['triangulo_simetrico'] = {
            'confianca': min(1.0, (abs(coef_max) + abs(coef_min)) * 5),
            'descricao': 'Triângulo Simétrico - Consolidação antes de movimento direcional'
        }
    
    # Detectar Topo Duplo
    if len(maximos) >= janela:
        # Encontrar os dois maiores picos
        sorted_idx = np.argsort(maximos)
        pico1_idx = sorted_idx[-1]
        pico2_idx = sorted_idx[-2]
        
        # Verificar se os picos estão distantes o suficiente
        if abs(pico1_idx - pico2_idx) > janela/4 and abs(maximos[pico1_idx] - maximos[pico2_idx]) < maximos[pico1_idx] * 0.01:
            padroes['topo_duplo'] = {
                'confianca': 0.7,
                'descricao': 'Topo Duplo - Potencial reversão de alta para baixa'
            }
    
    # Detectar Fundo Duplo
    if len(minimos) >= janela:
        # Encontrar os dois menores vales
        sorted_idx = np.argsort(minimos)
        vale1_idx = sorted_idx[0]
        vale2_idx = sorted_idx[1]
        
        # Verificar se os vales estão distantes o suficiente
        if abs(vale1_idx - vale2_idx) > janela/4 and abs(minimos[vale1_idx] - minimos[vale2_idx]) < minimos[vale1_idx] * 0.01:
            padroes['fundo_duplo'] = {
                'confianca': 0.7,
                'descricao': 'Fundo Duplo - Potencial reversão de baixa para alta'
            }
    
    # Detectar Bandeira de Alta (correção em tendência de alta)
    if np.mean(dados_recentes['fechamento'].pct_change().fillna(0) * 100) > 0.05:
        # Verificar se últimos N candles formam um canal de correção
        if coef_max < 0 and coef_min < 0 and abs(coef_max - coef_min) < 0.1:
            padroes['bandeira_alta'] = {
                'confianca': 0.6,
                'descricao': 'Bandeira de Alta - Potencial continuação de movimento de alta'
            }
    
    # Detectar Bandeira de Baixa (correção em tendência de baixa)
    if np.mean(dados_recentes['fechamento'].pct_change().fillna(0) * 100) < -0.05:
        # Verificar se últimos N candles formam um canal de correção
        if coef_max > 0 and coef_min > 0 and abs(coef_max - coef_min) < 0.1:
            padroes['bandeira_baixa'] = {
                'confianca': 0.6,
                'descricao': 'Bandeira de Baixa - Potencial continuação de movimento de baixa'
            }
    
    return padroes


def analisar_volume_preco(df, janela=20):
    """
    Analisa a relação entre volume e preço para identificar absorção ou exaustão
    
    Parâmetros:
    - df: DataFrame com dados de preços e volume
    - janela: tamanho da janela para análise
    
    Retorno:
    - dict com análise de volume/preço
    """
    analise = {}
    
    # Verificar se há dados suficientes
    if len(df) < janela:
        return analise
    
    # Dados recentes para análise
    dados_recentes = df.tail(janela)
    
    # Calcular média de volume
    volume_medio = dados_recentes['volume'].mean()
    
    # Identificar candles com volume acima da média
    candles_volume_alto = dados_recentes[dados_recentes['volume'] > volume_medio * 1.5]
    
    # Analisar candles de alta com volume alto (Acumulação/Distribuição)
    candles_alta_vol_alto = candles_volume_alto[candles_volume_alto['fechamento'] > candles_volume_alto['abertura']]
    
    # Analisar candles de baixa com volume alto
    candles_baixa_vol_alto = candles_volume_alto[candles_volume_alto['fechamento'] < candles_volume_alto['abertura']]
    
    # Calcular rejeição de preços (sombras dos candles)
    dados_recentes['sombra_superior'] = dados_recentes['maxima'] - dados_recentes[['abertura', 'fechamento']].max(axis=1)
    dados_recentes['sombra_inferior'] = dados_recentes[['abertura', 'fechamento']].min(axis=1) - dados_recentes['minima']
    
    # Identificar regiões com rejeição significativa e volume alto
    rejeicao_superior = dados_recentes[(dados_recentes['sombra_superior'] > dados_recentes['sombra_superior'].mean() * 1.5) & 
                                       (dados_recentes['volume'] > volume_medio * 1.2)]
    
    rejeicao_inferior = dados_recentes[(dados_recentes['sombra_inferior'] > dados_recentes['sombra_inferior'].mean() * 1.5) & 
                                       (dados_recentes['volume'] > volume_medio * 1.2)]
    
    # Analisar divergências entre preço e volume
    # Tendência de preço vs tendência de volume
    if len(dados_recentes) >= 5:
        preco_tendencia = dados_recentes['fechamento'].iloc[-1] > dados_recentes['fechamento'].iloc[-5]
        volume_tendencia = dados_recentes['volume'].iloc[-1] > dados_recentes['volume'].iloc[-5]
        
        if preco_tendencia and not volume_tendencia:
            analise['divergencia'] = {
                'tipo': 'Alta sem confirmação de volume',
                'descricao': 'Preços subindo com volume decrescente - possível exaustão de alta',
                'confianca': 0.7
            }
        elif not preco_tendencia and volume_tendencia:
            analise['divergencia'] = {
                'tipo': 'Baixa com volume crescente',
                'descricao': 'Preços caindo com volume crescente - possível aceleração de baixa',
                'confianca': 0.8
            }
    
    # Identificar regiões de absorção (preço mantém-se estável com volume alto)
    variacao_preco = dados_recentes['fechamento'].pct_change().abs()
    media_variacao = variacao_preco.mean()
    
    candles_volume_alto_baixa_variacao = dados_recentes[(dados_recentes['volume'] > volume_medio * 1.2) & 
                                                       (variacao_preco < media_variacao * 0.5)]
    
    if len(candles_volume_alto_baixa_variacao) > 0:
        if candles_volume_alto_baixa_variacao['fechamento'].iloc[-1] > candles_volume_alto_baixa_variacao['abertura'].iloc[-1]:
            analise['absorcao'] = {
                'tipo': 'Absorção de venda',
                'descricao': 'Alto volume com baixa variação de preço em candle de alta - possível absorção de vendedores',
                'confianca': 0.75
            }
        else:
            analise['absorcao'] = {
                'tipo': 'Absorção de compra',
                'descricao': 'Alto volume com baixa variação de preço em candle de baixa - possível absorção de compradores',
                'confianca': 0.75
            }
    
    # Exaustão - Alto volume com grande variação de preço seguido por reversão
    if len(dados_recentes) >= 3:
        ultimos_candles = dados_recentes.tail(3)
        if (ultimos_candles['volume'].iloc[0] > volume_medio * 1.5 and
            abs(ultimos_candles['fechamento'].iloc[0] - ultimos_candles['abertura'].iloc[0]) > 
            abs(dados_recentes['fechamento'] - dados_recentes['abertura']).mean() * 1.5):
            
            # Verificar se houve reversão nos candles seguintes
            if ((ultimos_candles['fechamento'].iloc[0] > ultimos_candles['abertura'].iloc[0] and 
                 ultimos_candles['fechamento'].iloc[-1] < ultimos_candles['abertura'].iloc[-1]) or
                (ultimos_candles['fechamento'].iloc[0] < ultimos_candles['abertura'].iloc[0] and 
                 ultimos_candles['fechamento'].iloc[-1] > ultimos_candles['abertura'].iloc[-1])):
                
                analise['exaustao'] = {
                    'tipo': 'Possível exaustão',
                    'descricao': 'Alto volume com grande variação seguido por reversão - possível exaustão de movimento',
                    'confianca': 0.85
                }
    
    # Resumo da análise de volume/preço
    if len(candles_alta_vol_alto) > len(candles_baixa_vol_alto) * 1.5:
        analise['resumo'] = {
            'tendencia': 'Acumulação',
            'descricao': 'Predominância de candles de alta com volume alto - possível acumulação',
            'confianca': min(1.0, len(candles_alta_vol_alto) / max(1, len(candles_baixa_vol_alto)) * 0.3)
        }
    elif len(candles_baixa_vol_alto) > len(candles_alta_vol_alto) * 1.5:
        analise['resumo'] = {
            'tendencia': 'Distribuição',
            'descricao': 'Predominância de candles de baixa com volume alto - possível distribuição',
            'confianca': min(1.0, len(candles_baixa_vol_alto) / max(1, len(candles_alta_vol_alto)) * 0.3)
        }
    
    # Adicionar níveis importantes identificados pela análise de volume
    analise['niveis_importantes'] = {}
    
    # Níveis de rejeição superior (resistências potenciais)
    if len(rejeicao_superior) > 0:
        for i, row in rejeicao_superior.iterrows():
            nivel = row['maxima']
            volume = row['volume']
            analise['niveis_importantes'][nivel] = {
                'tipo': 'resistencia',
                'forca': min(1.0, volume / volume_medio - 1),
                'descricao': f'Rejeição superior com volume {volume/volume_medio:.1f}x acima da média'
            }
    
    # Níveis de rejeição inferior (suportes potenciais)
    if len(rejeicao_inferior) > 0:
        for i, row in rejeicao_inferior.iterrows():
            nivel = row['minima']
            volume = row['volume']
            analise['niveis_importantes'][nivel] = {
                'tipo': 'suporte',
                'forca': min(1.0, volume / volume_medio - 1),
                'descricao': f'Rejeição inferior com volume {volume/volume_medio:.1f}x acima da média'
            }
    
    return analise


def calcular_indicadores_sentimento(df, janela=14):
    """
    Calcula indicadores de sentimento do mercado
    
    Parâmetros:
    - df: DataFrame com dados de preços
    - janela: tamanho da janela para cálculos
    
    Retorno:
    - dict com indicadores de sentimento
    """
    sentimento = {}
    
    # Verificar se há dados suficientes
    if len(df) < janela:
        return sentimento
    
    # Dados recentes
    dados_recentes = df.tail(janela+1)
    
    # 1. Índice de Força Relativa (RSI)
    delta = dados_recentes['fechamento'].diff()
    ganhos = delta.copy()
    perdas = delta.copy()
    ganhos[ganhos < 0] = 0
    perdas[perdas > 0] = 0
    perdas = abs(perdas)
    
    # Calcular médias de ganhos e perdas
    media_ganhos = ganhos.rolling(window=janela).mean()
    media_perdas = perdas.rolling(window=janela).mean()
    
    # Calcular RS e RSI
    rs = media_ganhos / media_perdas
    rsi = 100 - (100 / (1 + rs))
    
    # 2. Estocástico
    min_periodo = dados_recentes['minima'].rolling(window=janela).min()
    max_periodo = dados_recentes['maxima'].rolling(window=janela).max()
    
    k_percent = 100 * ((dados_recentes['fechamento'] - min_periodo) / (max_periodo - min_periodo))
    d_percent = k_percent.rolling(window=3).mean()
    
    # 3. MACD
    ema12 = df['fechamento'].ewm(span=12, adjust=False).mean()
    ema26 = df['fechamento'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    # 4. Análise de tendência (ADX)
    # +DI e -DI são simplificados aqui
    high_diff = df['maxima'].diff()
    low_diff = df['minima'].diff().abs() * -1  # Inverted for -DM
    
    plus_dm = high_diff.copy()
    plus_dm[plus_dm < 0] = 0
    plus_dm[(high_diff < 0) | (high_diff < low_diff)] = 0
    
    minus_dm = low_diff.copy()
    minus_dm[minus_dm > 0] = 0
    minus_dm = minus_dm.abs()
    minus_dm[(low_diff > 0) | (high_diff > abs(low_diff))] = 0
    
    tr = pd.DataFrame({
        'hl': df['maxima'] - df['minima'],
        'hc': abs(df['maxima'] - df['fechamento'].shift(1)),
        'lc': abs(df['minima'] - df['fechamento'].shift(1))
    }).max(axis=1)
    
    atr = tr.rolling(window=janela).mean()
    plus_di = 100 * (plus_dm.rolling(window=janela).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=janela).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=janela).mean()
    
    # 5. Chaikin Money Flow (CMF)
    mfm = ((df['fechamento'] - df['minima']) - (df['maxima'] - df['fechamento'])) / (df['maxima'] - df['minima'])
    mfm = mfm.replace([np.inf, -np.inf], 0)
    mfv = mfm * df['volume']
    cmf = mfv.rolling(window=janela).sum() / df['volume'].rolling(window=janela).sum()
    
    # Compilar resultados
    sentimento['rsi'] = {
        'valor': float(rsi.iloc[-1]),
        'interpretacao': 'sobrecomprado' if rsi.iloc[-1] > 70 else 'sobrevendido' if rsi.iloc[-1] < 30 else 'neutro',
        'tendencia': 'alta' if rsi.iloc[-1] > rsi.iloc[-2] else 'baixa'
    }
    
    sentimento['estocastico'] = {
        'k': float(k_percent.iloc[-1]),
        'd': float(d_percent.iloc[-1]),
        'interpretacao': 'sobrecomprado' if k_percent.iloc[-1] > 80 else 'sobrevendido' if k_percent.iloc[-1] < 20 else 'neutro',
        'tendencia': 'alta' if k_percent.iloc[-1] > d_percent.iloc[-1] else 'baixa'
    }
    
    sentimento['macd'] = {
        'linha': float(macd_line.iloc[-1]),
        'sinal': float(signal_line.iloc[-1]),
        'histograma': float(macd_hist.iloc[-1]),
        'interpretacao': 'alta' if macd_line.iloc[-1] > signal_line.iloc[-1] else 'baixa',
        'tendencia': 'fortalecendo' if macd_hist.iloc[-1] > macd_hist.iloc[-2] else 'enfraquecendo'
    }
    
    sentimento['adx'] = {
        'adx': float(adx.iloc[-1]),
        'plus_di': float(plus_di.iloc[-1]),
        'minus_di': float(minus_di.iloc[-1]),
        'forca_tendencia': 'forte' if adx.iloc[-1] > 25 else 'fr