import numpy as np


def hill(losses, k):
    # Estimateur de Hill moyenne des ln(pire / seuil) sur les k pires pertes
    pertes_triees = losses.sort_values(ascending=False)
    pires = pertes_triees.iloc[:k]
    seuil = pertes_triees.iloc[k]
    return np.log(pires / seuil).mean()


def pickands(losses, k):
    # Estimateur de Pickhands qui compare les marches entre les rangs k, 2k et 4k
    pertes_triees = losses.sort_values(ascending=False)
    x1 = pertes_triees.iloc[k]
    x2 = pertes_triees.iloc[2 * k]
    x4 = pertes_triees.iloc[4 * k]
    return np.log((x1 - x2) / (x2 - x4)) / np.log(2)


def var_evt(losses, alpha, k):
    # VaR EVT (Hill) seuil * (k / (n * (1 - alpha)))^xi
    pertes_triees = losses.sort_values(ascending=False)
    seuil = pertes_triees.iloc[k]
    xi = hill(losses, k)
    n = len(losses)
    return seuil * (k / (n * (1 - alpha))) ** xi


def es_evt(losses, alpha, k):
    # ES EVT > VaR / (1 - xi), valable seulement si xi < 1
    xi = hill(losses, k)
    if xi >= 1:
        return np.inf      # l'ES n'existe pas (moyenne infinie)
    return var_evt(losses, alpha, k) / (1 - xi)