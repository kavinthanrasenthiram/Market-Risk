import numpy as np
from scipy.stats import norm, t
from arch import arch_model


def fit_garch(port_returns, dist="t"):
    # Estime un GARCH(1,1) sur les rendements (en % pour aider l'optimiseur)
    modele = arch_model(100 * port_returns, vol="GARCH", p=1, q=1, dist=dist)
    resultat = modele.fit(disp="off")
    return resultat


def garch_var_es(resultat, alpha):
    # 1. sigma( lecart type en gros le volatilite) de demain (prévision), en décimal
    prevision = resultat.forecast(horizon=1)
    variance_demain = prevision.variance.iloc[-1, 0]
    sigma_demain = np.sqrt(variance_demain) / 100

    # 2. perte moyenne (perte = - rendement), en décimal
    mu_pertes = -resultat.params["mu"] / 100

    # 3. quantile z et ES de la loi des chocs (variance 1)
    if "nu" in resultat.params:
        # loi de Student
        nu = resultat.params["nu"]
        s = np.sqrt((nu - 2) / nu)          # correction pour avoir une variance de 1
        q = t.ppf(alpha, nu)
        z = q * s
        es_z = t.pdf(q, nu) / (1 - alpha) * (nu + q**2) / (nu - 1) * s
    else:
        # loi normale
        z = norm.ppf(alpha)
        es_z = norm.pdf(z) / (1 - alpha)

    # 4. VaR et ES
    var = mu_pertes + sigma_demain * z
    es = mu_pertes + sigma_demain * es_z
    return var, es



def var_fhs(resultat, alpha):
    # FHS : sigma de demain (GARCH) x quantile des chocs standardisés réels (au lieu de la loi de Student)
    prevision = resultat.forecast(horizon=1)
    sigma_demain = np.sqrt(prevision.variance.iloc[-1, 0]) / 100
    mu_pertes = -resultat.params["mu"] / 100
    z = -resultat.std_resid.dropna()          # chocs standardisés, côté pertes
    return mu_pertes + sigma_demain * np.quantile(z, alpha)