import numpy as np
from scipy.stats import norm

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
    return es # voir cahier un autre methode plus simple avec pandas avc return losses[losses>=var].mean()

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