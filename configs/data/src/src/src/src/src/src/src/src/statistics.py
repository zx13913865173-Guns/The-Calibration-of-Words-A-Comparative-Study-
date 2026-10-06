import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.inter_rater import fleiss_kappa

def anova_two_way(df, dv, factor1, factor2):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    model = ols(f"{dv} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})", data=df).fit()
    return sm.stats.anova_lm(model, typ=2)

def paired_ttest(a, b):
    t, p = stats.ttest_rel(a, b)
    return t, p

def fleiss_kappa_from_matrix(matrix):
    # matrix: n_items x n_categories
    return fleiss_kappa(matrix)
