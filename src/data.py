import os
import pandas as pd
import yfinance as yf
import numpy as np


def load_prices(tickers, start, end, path):
    # Prix de clôture ajustés des actions.
    # Si le fichier existe déjà, on le relit ; 
    # sinon on télécharge et on sauvegarde.

    if os.path.exists(path):
        # Le fichier existe : on le relit
        prices = pd.read_csv(path,index_col=0, parse_dates=True)


    else:
        # Le fichier n'existe pas : on télécharge
        data = yf.download(tickers, start=start, end=end ,auto_adjust=True )
        prices = data['Close']    # on garde seulement les prix de clôture
        prices = prices.dropna()  # on enlève les lignes avec des valeurs manquantes
        prices.to_csv(path)      # on sauvegarde dans le fichier

    return prices


def compute_returns(prices):
    returns=np.log(prices/prices.shift(1))
    returns=returns.dropna()
    return returns

