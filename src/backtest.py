import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import chi2
from scipy.special import xlogy
from src.risk_measures import var_hist, var_gaussien, var_kde
from src.evt import var_evt
from src.garch import garch_var_es, var_fhs


def rolling_var(losses, window=500, start="2008", refit_every=20, k_evt=25):
    # Chaque jour t : VaR 95 % et 99 % de chaque méthode, calculée avec les 500 jours AVANT t
    lignes = []
    for i, t in enumerate(range(window, len(losses))):
        fenetre = losses.iloc[t - window : t]
        rendements = -100 * fenetre

        # GARCH : réestimation tous les 20 jours, sinon on garde les paramètres
        modele_n = arch_model(rendements, vol="GARCH", p=1, q=1, dist="normal")
        modele_t = arch_model(rendements, vol="GARCH", p=1, q=1, dist="t")
        if i % refit_every == 0:
            params_n = modele_n.fit(disp="off").params
            params_t = modele_t.fit(disp="off").params
        res_n = modele_n.fix(params_n)
        res_t = modele_t.fix(params_t)

        ligne = {"date": losses.index[t], "perte": losses.iloc[t]}
        for alpha in [0.95, 0.99]:
            niv = str(int(alpha * 100))
            ligne["hist_" + niv] = var_hist(fenetre, alpha)
            ligne["gauss_" + niv] = var_gaussien(fenetre, alpha)
            ligne["kde_" + niv] = var_kde(fenetre, alpha)
            ligne["evt_" + niv] = var_evt(fenetre, alpha, k_evt)
            ligne["garch_n_" + niv] = garch_var_es(res_n, alpha)[0]
            ligne["garch_t_" + niv] = garch_var_es(res_t, alpha)[0]
    
        lignes.append(ligne)

        if i % 500 == 0:
            print("jour", i, "-", losses.index[t].date())

    resultats = pd.DataFrame(lignes).set_index("date")
    return resultats.loc[start:]


def hits(pertes, var):
    # 1 si la perte dépasse la VaR, 0 sinon
    return (pertes > var).astype(int)


def kupiec(h, alpha):
    # H0 : taux de dépassement = 1-alpha Renvoie la p-value
    p = 1 - alpha
    n = len(h)
    x = h.sum()
    p_obs = x / n
    log_l0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    log_l1 = xlogy(n - x, 1 - p_obs) + xlogy(x, p_obs)
    lr = -2 * (log_l0 - log_l1)
    return 1 - chi2.cdf(lr, 1)


def christoffersen(h):
    # H0 : dépassements indépendants (pi0 = pi1). Renvoie pi0, pi1, p-value
    h = np.asarray(h)
    hier, auj = h[:-1], h[1:]
    n00 = ((hier == 0) & (auj == 0)).sum()
    n01 = ((hier == 0) & (auj == 1)).sum()
    n10 = ((hier == 1) & (auj == 0)).sum()
    n11 = ((hier == 1) & (auj == 1)).sum()

    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    log_l0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_l1 = xlogy(n00, 1 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    lr = -2 * (log_l0 - log_l1)
    return pi0, pi1, 1 - chi2.cdf(lr, 1)


def basel_zones(h, window=250):
    # Dépassements sur les 250 derniers jours, chaque jour -> % du temps en vert / jaune / rouge
    nb_250 = h.rolling(window).sum().dropna()
    return {
        "vert (%)": 100 * (nb_250 <= 4).mean(),
        "jaune (%)": 100 * ((nb_250 >= 5) & (nb_250 <= 9)).mean(),
        "rouge (%)": 100 * (nb_250 >= 10).mean(),
        "pire année": nb_250.max(),
    }


def backtest_table(resultats, methodes, alpha):
    # Tableau récapitulatif : une ligne par méthode
    niv = str(int(alpha * 100))
    lignes = []
    for m in methodes:
        var = resultats[m + "_" + niv]
        h = hits(resultats["perte"], var)
        pi0, pi1, p_ind = christoffersen(h)
        ligne = {
            "méthode": m,
            "dépassements": h.sum(),
            "attendus": round(len(h) * (1 - alpha)),
            "p Kupiec": kupiec(h, alpha),
            "pi0 (%)": 100 * pi0,
            "pi1 (%)": 100 * pi1,
            "p Christoffersen": p_ind,
            "VaR moyenne (%)": 100 * var.mean(),
        }
        if niv == "99":
            ligne.update(basel_zones(h))
        lignes.append(ligne)
    return pd.DataFrame(lignes).set_index("méthode").round(3)

def rolling_fhs(losses, window=500, start="2008", refit_every=20):
    # VaR FHS jour par jour : même GARCH-t que rolling_var, mais quantile des chocs standardisés réels
    lignes = []
    for i, t in enumerate(range(window, len(losses))):
        fenetre = losses.iloc[t - window : t]
        modele = arch_model(-100 * fenetre, vol="GARCH", p=1, q=1, dist="t")
        if i % refit_every == 0:
            params = modele.fit(disp="off").params
        res = modele.fix(params)
        lignes.append({
            "date": losses.index[t],
            "fhs_95": var_fhs(res, 0.95),
            "fhs_99": var_fhs(res, 0.99),
        })
        if i % 500 == 0:
            print("jour", i, "-", losses.index[t].date())
    return pd.DataFrame(lignes).set_index("date").loc[start:]