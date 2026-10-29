import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# VAR/ES historique
def var_hist(losses, alpha):
    Var=np.quantile(losses,alpha)
    return Var

def es_hist(losses,alpha):
    var=var_hist(losses,alpha)
    pires=[]
    for element in losses:
        if element >= var:
            pires.append(element)
    es=np.mean(pires) # car pires est une liste
    return es # voir cahier ya une autre methode plus simple avec pandas avc return losses[losses>=var].mean()

#VAR/ES gaussienne

from scipy.stats import norm

def var_gaussien(losses,alpha):
    mu=losses.mean()
    sigma=losses.std()
    z=norm.ppf(alpha)  # quantile de la cloche standard
    var=mu+sigma*z
    return var

def es_gaussien(losses,alpha):
    mu=losses.mean()
    sigma=losses.std()
    z=norm.ppf(alpha)
    es=mu+sigma*(norm.pdf(z)/(1-alpha))
    return es


#KDE

def silverman_bandwidth(losses):  # pour trouver h , si le temp  faire la validation croisée : 
    # on teste plein de valeurs de h et on garde celle qui colle le mieux aux données
    n = len(losses) 
    sigma = losses.std()
    iqr = np.quantile(losses, 0.75) - np.quantile(losses, 0.25)
    return 0.9 * min(sigma, iqr / 1.34) * n ** (-1/5)

def kde_cdf(x, losses, h):
    # sa donne la proba que la perte L a un jour soit inf ou egale a x P(L <= x) estimée par la KDE : moyenne des Phi((x - L_i) / h) 
    return norm.cdf((x - losses) / h).mean()


def var_kde(losses, alpha, h=None):
    # VaR KDE : le x tel que P(L <= x) = alpha 
    if h is None:
        h = silverman_bandwidth(losses)

    # On cherche le x où kde_cdf(x) - alpha = 0
    def ecart(x):
        return kde_cdf(x, losses, h) - alpha

    debut = losses.min() - 5 * h
    fin = losses.max() + 5 * h
    return brentq(ecart, debut, fin)


def es_kde(losses, alpha, h=None):
    # ES KDE : la perte moyenne au-delà de la VaR, sous la courbe KDE
    if h is None:
        h = silverman_bandwidth(losses)
    v = var_kde(losses, alpha, h)
    d = (v - losses) / h
    queue = losses * (1 - norm.cdf(d)) + h * norm.pdf(d)
    return queue.mean() / (1 - alpha)