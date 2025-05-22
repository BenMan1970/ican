import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import os
from ta.trend import adx

st.set_page_config(page_title="Scanner Simple", layout="centered")
st.title("⭐ Signaux 5 & 6 Étoiles")

# 🔐 Clés API depuis Streamlit Secrets
try:
    api_key = st.secrets["alpaca"]["alpaca_api_key"]
    secret_key = st.secrets["alpaca"]["alpaca_secret_key"]
except:
    api_key = os.getenv("alpaca_api_key", "")
    secret_key = os.getenv("alpaca_secret_key", "")

headers = {
    "Apca-Api-Key-Id": api_key,
    "Apca-Api-Secret-Key": secret_key
}

hma_length = 20
adx_threshold = 20
rsi_length = 10
adx_length = 14
ichimoku_len = 9

symbols = [
    'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD',
    'USDJPY', 'USDCAD', 'USDCHF',
    'XAUUSD', 'US30', 'NAS100', 'SPX'
]

@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def fetch_bars(symbol, timeframe='1Hour', limit=100):
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    params = {'timeframe': timeframe, 'limit': limit}
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'bars' in data and data['bars']:
                return pd.DataFrame(data['bars'])
        return None
    except Exception as e:
        st.error(f"Erreur pour {symbol}: {str(e)}")
        return None

def hma(values, length):
    if len(values) < length:
        return np.full(len(values), np.nan)
    
    half_length = max(1, int(length / 2))
    sqrt_length = max(1, int(np.sqrt(length)))
    
    # Calcul WMA simplifié avec pandas
    series = pd.Series(values)
    wma_half = series.rolling(half_length).mean()
    wma_full = series.rolling(length).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_result = raw_hma.rolling(sqrt_length).mean()
    
    return hma_result.fillna(method='bfill').values

def calculate_indicators(df):
    if len(df) < 60:
        return None
        
    # Assurer que les colonnes sont numériques
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    
    # Supprimer les lignes avec des NaN
    df = df.dropna()
    
    if len(df) < 60:
        return None
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values

    # HMA
    try:
        df['hma'] = hma(close, hma_length)
        df['hma_slope'] = np.where(df['hma'] > df['hma'].shift(1), 1, -1)
    except:
        df['hma_slope'] = 0

    # Heiken Ashi
    try:
        ha_close = (open_ + high + low + close) / 4
        ha_open = np.zeros_like(open_)
        ha_open[0] = (open_[0] + close[0]) / 2
        for i in range(1, len(ha_open)):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
        df['haSignal'] = np.where(ha_close > ha_open, 1, -1)
    except:
        df['haSignal'] = 0

    # Smoothed Heiken Ashi
    try:
        def ema(x, n): 
            return pd.Series(x).ewm(span=n, adjust=False).mean().values
        o = ema(open_, 10)
        c = ema(close, 10)
        h = ema(high, 10)
        l = ema(low, 10)
        haclose = (o + h + l + c) / 4
        haopen = np.zeros_like(o)
        haopen[0] = (o[0] + c[0]) / 2
        for i in range(1, len(haopen)):
            haopen[i] = (haopen[i - 1] + haclose[i - 1]) / 2
        df['smoothedHaSignal'] = np.where(ema(haopen, 10) < ema(haclose, 10), 1, -1)
    except:
        df['smoothedHaSignal'] = 0

    # RSI
    try:
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(rsi_length).mean()
        avg_loss = loss.rolling(rsi_length).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsiSignal'] = np.where(df['rsi'] > 50, 1, -1)
    except:
        df['rsiSignal'] = 0

    # ADX
    try:
        df['adx'] = adx(pd.Series(high), pd.Series(low), pd.Series(close), window=adx_length)
        df['adxHasMomentum'] = np.where(df['adx'] >= adx_threshold, 1, 0)
    except:
        df['adxHasMomentum'] = 0

    # Ichimoku
    try:
        tenkan = (pd.Series(high).rolling(ichimoku_len).max() + pd.Series(low).rolling(ichimoku_len).min()) / 2
        kijun = (pd.Series(high).rolling(26).max() + pd.Series(low).rolling(26).min()) / 2
        senkouA = (tenkan + kijun) / 2
        senkouB = (pd.Series(high).rolling(52).max() + pd.Series(low).rolling(52).min()) / 2
        df['ichimokuSignal'] = np.where(close > senkouA, 1, np.where(close < senkouB, -1, 0))
    except:
        df['ichimokuSignal'] = 0

    return df.iloc[-1]

def scan_market():
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols):
        status_text.text(f'Analyse de {symbol}...')
        progress_bar.progress((i + 1) / len(symbols))
        
        data = fetch_bars(symbol, '1Hour')
        if data is not None and len(data) > 60:
            try:
                row = calculate_indicators(data)
                if row is not None:
                    # Calculer les signaux
                    signal_columns = ['hma_slope', 'haSignal', 'smoothedHaSignal', 'rsiSignal', 'adxHasMomentum', 'ichimokuSignal']
                    
                    # S'assurer que toutes les colonnes existent
                    for col in signal_columns:
                        if col not in row or pd.isna(row[col]):
                            row[col] = 0
                    
                    bull = sum(1 for x in signal_columns if row[x] == 1)
                    bear = sum(1 for x in signal_columns if row[x] == -1)
                    conf = max(bull, bear)
                    direction = '🟢 ↑' if bull >= bear else '🔴 ↓'
                    
                    if conf >= 5:
                        results.append({
                            'Symbole': symbol,
                            'Direction': direction,
                            'Étoiles': '⭐' * conf,
                            'Score': conf
                        })
            except Exception as e:
                st.warning(f"Erreur lors de l'analyse de {symbol}: {str(e)}")
                continue
    
    progress_bar.empty()
    status_text.empty()
    return results

# Interface principale
if st.button("🔄 Scanner le marché", type="primary"):
    with st.spinner("Analyse en cours..."):
        signals = scan_market()
        
        if signals:
            st.success(f"✅ {len(signals)} signaux détectés!")
            
            # Afficher sous forme de tableau
            df_results = pd.DataFrame(signals)
            df_results = df_results.sort_values('Score', ascending=False)
            
            st.dataframe(
                df_results,
                hide_index=True,
                use_container_width=True
            )
            
            # Afficher aussi sous forme de liste
            st.write("### 📊 Résultats détaillés:")
            for result in signals:
                st.write(f"**{result['Symbole']}** {result['Direction']} {result['Étoiles']}")
        else:
            st.info("🔍 Aucun signal 5+ étoiles détecté actuellement.")

# Auto-refresh option
if st.checkbox("🔄 Auto-refresh (toutes les 5 minutes)"):
    time.sleep(5)  # Attendre 5 secondes au lieu de 5 minutes pour les tests
    st.rerun()

# Informations
with st.expander("ℹ️ Informations sur les indicateurs"):
    st.write("""
    **Indicateurs utilisés :**
    - **HMA (Hull Moving Average)** : Tendance
    - **Heiken Ashi** : Direction du mouvement
    - **Smoothed Heiken Ashi** : Tendance lissée
    - **RSI** : Force relative
    - **ADX** : Momentum
    - **Ichimoku** : Support/Résistance
    
    **Score :** Nombre d'indicateurs alignés dans la même direction
    """)
