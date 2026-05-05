import os
import pickle
import logging
import math
import re
import warnings
import numpy as np
import pandas as pd
import scipy
import numba
import pymc as pm
#import powerlaw
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from dataclasses import dataclass


def round_p_value(p: float) -> str:
    """
    Round a p-value to 2 decimal places if above 0.01, 3 if above 0.001, and zero if below.
    """
    if p == 0:
        return "0"
    if p > .01:
        return f"{p:.2f}"
    if p > .001:
        return f"{p:.3f}"
    return "$10^{" f"{int(math.log10(p))}" "}$"


def estimate_tail_index(y, tail='left', frac=0.05):
    """
    Estimate the tail index of a return series using the Hill estimator.

    Parameters:
        y (pd.Series): The series of returns.
        tail (str): 'left' for negative tail, 'right' for positive tail, 'both' for both tails.
        frac (float): Fraction of extreme values to use (e.g., 0.05 for 5%).

    Returns:
        dict: Tail indices for the selected tails.
    """
    if isinstance( y, pd.Series ):
        y = y.values
    y = y[ np.isfinite(y) ]
    n = len(y)
    k = int(frac * n)

    result = {}

    if tail in ['left', 'both']:
        # Hill on the upper tail of (-y), i.e. the most negative returns (same indexing as right tail).
        neg_returns_sorted = np.sort(-y)[::-1]
        if k > 0 and len(neg_returns_sorted) >= k:
            top_k = neg_returns_sorted[:k]
            x_min = top_k[-1]
            hill = np.mean(np.log(top_k / x_min))
            result['left'] = 1 / hill if hill != 0 else np.nan
        else:
            result['left'] = np.nan

    if tail in ['right', 'both']:
        pos_returns_sorted = np.sort(y)[::-1]  # right tail
        if k > 0 and len(pos_returns_sorted) >= k:
            top_k = pos_returns_sorted[:k]
            x_min = top_k[-1]
            hill = np.mean(np.log(top_k / x_min))
            result['right'] = 1 / hill if hill != 0 else np.nan
        else:
            result['right'] = np.nan

    return result


@dataclass
class GARCHParameters:
    SR: float
    SR_p: float
    mu: float
    sigma: float
    sigma_p: float
    skew: float
    kurtosis: float
    alpha: float
    beta: float
    phi: float
    left_tail: float
    right_tail: float
    tail_index: float
    T: int
    p: float
    denominator: float

    gjr_omega: float
    gjr_alpha: float
    gjr_gamma: float
    gjr_beta: float
    egarch_omega: float
    egarch_alpha: float
    egarch_gamma: float
    egarch_beta: float
    sv_mu: float
    sv_phi: float
    sv_sigma_v: float

    su_a: float
    su_b: float
    su_loc: float
    su_scale: float
    nct_df: float
    nct_nc: float
    nct_loc: float
    nct_scale: float
    skew_t_a: float  # df = a + b if a=b;  negatively skewed if a < b
    skew_t_b: float
    skew_t_loc: float
    skew_t_scale: float
    arch_lm_pvalue: float
    f_pvalue: float
    ljung_box_pvalue: float


# Global variable to store the pretrained model to estimate the stochastic volatility parameters from the statistics of the returns, if it is available.
# See the notebook "unused/Stochastic_Volatility.ipynb" for more details.
stochastic_volatility_features_to_parameters = "not loaded yet"

def estimate_parameters( r: np.ndarray | pd.Series, p = 1.5 ) -> GARCHParameters:
    "Estimate the GARCH parameters (and many others) from the returns"

    # Moments
    if isinstance( r, pd.Series ):
        r = r.values
    T = len(r)
    mu = r.mean()
    y = r - mu
    sigma = y.std()
    SR = mu / sigma
    skew = scipy.stats.skew( y )  # Skewness of the returns

    # GARCH parameters
    model = arch_model(y, vol='Garch', p=1, q=1, dist='skewt', mean = 'Zero', rescale=True)
    res = model.fit(disp="off")
    innovations = res.resid / res.conditional_volatility
    alpha = res.params['alpha[1]']
    beta = res.params['beta[1]']
    phi = alpha + beta
    kurtosis = scipy.stats.kurtosis( innovations ) + 3  # Kurtosis of the innovations, not of the returns
    denominator = ( 1 - alpha**2 * kurtosis - 2 * alpha * beta - beta**2 )

    # Fractional moments 
    sigma_p = np.mean( np.abs( mu - r ) ** p ) ** ( 1 / p )
    SR_p = mu / sigma_p

    # Different models for the innovations: Johnson SU, non-central t, skewed t
    su_a, su_b, su_loc, su_scale = scipy.stats.johnsonsu.fit(innovations)
    nct_params = scipy.stats.nct.fit(innovations)
    skew_t_params = scipy.stats.jf_skew_t.fit(innovations)

    # GJR-GARCH model: leverage effect (negative returns increase the volatility more than positive returns)
    gjr_model = arch_model(y, p=1, o=1, q=1, vol='GARCH', dist='skewt', mean = 'Zero', rescale=True)  # o = 'asymmetry'
    gjr_result = gjr_model.fit(disp='off')

    # EGARCH model: model the log of the volatility (and add a leverage effect)
    egarch_model = arch_model(y, p=1, o=1, q=1, vol='EGARCH', dist='skewt', mean = 'Zero', rescale=True)
    egarch_result = egarch_model.fit(disp='off')

    # Stochastic volatility model
    sv_result = { 
        'mu': np.nan,
        'phi': np.nan,
        'sigma_v': np.nan,
    }
    if False:   # 10 minutes per time series! 
        with pm.Model() as sv_model:
            phi = pm.Uniform("phi", lower=-1, upper=1)      # Persistence
            sigma_v = pm.Exponential("sigma_v", lam=1)      # Vol of vol
            mu = pm.Normal("mu", mu=0, sigma=1)             # Mean log-vol
            h = pm.AR(name = "h", ar_order = 1, rho=[mu,phi], sigma=sigma_v, constant=True, shape=len(y))  # Latent volatility (log-variance) modeled as an AR(1) process
            y = pm.Normal("y", mu=0, sigma=pm.math.exp(h / 2), observed=y)  # Observation equation: returns ~ N(0, exp(h))
            trace = pm.sample(1000, tune=1000, target_accept=0.9, progressbar=False)
            sv_result = { 
                'mu': trace.posterior.mu.values.mean(),
                'phi': trace.posterior.phi.values.mean(),
                'sigma_v': trace.posterior.sigma_v.values.mean(),
            }

    global stochastic_volatility_features_to_parameters
    if isinstance( stochastic_volatility_features_to_parameters, str ):
        filename = f'cache/stochastic_volatility_features_to_parameters_500_10000_100.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                stochastic_volatility_features_to_parameters = pickle.load(f)
        else: 
            stochastic_volatility_features_to_parameters = None
    if stochastic_volatility_features_to_parameters is not None:
        sv_result = stochastic_volatility_features_to_parameters.predict( sv_features(y).reshape( 1, -1 ) )[0]
        sv_result = { 'mu': sv_result[0], 'phi': sv_result[1], 'sigma_v': sv_result[2] }
    
    # Tail index
    if False:
        left_tail = powerlaw.Fit( -y[ y < 0 ], xmin=None, verbose=0).alpha
        right_tail = powerlaw.Fit( y[ y > 0 ], xmin=None, verbose=0).alpha
        tail_index = ( left_tail + right_tail ) / 2
    tail_index = estimate_tail_index( y, tail = 'both' )

    # Test for (G)ARCH effects.
    # H₀: the squared returns have no autocorrelation, i.e., there are no (G)ARCH effects.
    # (These are tests on the squared returns, not the returns.)
    nlags = 5
    arch_lm_stat, arch_lm_pvalue, f_stat, f_pvalue = het_arch(y - y.mean(), nlags=nlags)
    ljung_box_stat, ljung_box_pvalue = acorr_ljungbox((y - y.mean())**2, lags = [nlags], return_df = True).loc[nlags,:]
    
    return GARCHParameters( 
        SR = SR, 
        SR_p = SR_p, 
        mu = mu, 
        sigma = sigma, 
        sigma_p = sigma_p, 
        skew = skew, 
        kurtosis = kurtosis, 
        alpha = alpha, 
        beta = beta, 
        phi = phi, 
        left_tail = tail_index['left'],
        right_tail = tail_index['right'],
        tail_index = (tail_index['left'] + tail_index['right']) / 2,
        T = T,
        p = p,
        denominator = denominator,

        gjr_omega = gjr_result.params['omega'],
        gjr_alpha = gjr_result.params['alpha[1]'],
        gjr_gamma = gjr_result.params['gamma[1]'],
        gjr_beta = gjr_result.params['beta[1]'],

        egarch_omega = egarch_result.params['omega'],
        egarch_alpha = egarch_result.params['alpha[1]'],
        egarch_gamma = egarch_result.params['gamma[1]'],
        egarch_beta = egarch_result.params['beta[1]'],

        sv_mu = sv_result['mu'],
        sv_phi = sv_result['phi'],
        sv_sigma_v = sv_result['sigma_v'],

        # Not used
        su_a = su_a,
        su_b = su_b,
        su_loc = su_loc,
        su_scale = su_scale,
        
        # Not used
        nct_df = nct_params[0],
        nct_nc = nct_params[1],
        nct_loc = nct_params[2],
        nct_scale = nct_params[3],

        skew_t_a = skew_t_params[0],
        skew_t_b = skew_t_params[1],
        skew_t_loc = skew_t_params[2],
        skew_t_scale = skew_t_params[3],

        arch_lm_pvalue = arch_lm_pvalue,
        f_pvalue = f_pvalue,
        ljung_box_pvalue = ljung_box_pvalue,
    )



def estimate_parameters_fast( r: np.ndarray | pd.Series, p = 1.5 ) -> GARCHParameters:
    "Estimate the GARCH parameters from the returns (only those: not the other models)"
    if isinstance( r, pd.Series ):
        r = r.values
    T = len(r)
    mu = r.mean()
    y = r - mu
    sigma = y.std()
    SR = mu / sigma
    skew = scipy.stats.skew( y )  # Skewness of the returns

    model = arch_model(y, vol='Garch', p=1, q=1, dist='skewt', mean = 'Zero', rescale=True)
    res = model.fit(disp="off")
    innovations = res.resid / res.conditional_volatility
    alpha = res.params['alpha[1]']
    beta = res.params['beta[1]']
    kurtosis = scipy.stats.kurtosis( innovations ) + 3  # Kurtosis of the innovations, not of the returns

    return GARCHParameters( 
        SR = SR, 
        SR_p = np.nan, 
        mu = mu, 
        sigma = sigma, 
        sigma_p = np.nan, 
        skew = skew, 
        kurtosis = kurtosis, 
        alpha = alpha, 
        beta = beta, 
        phi = np.nan, 
        left_tail = np.nan,
        right_tail = np.nan,
        tail_index = np.nan,
        T = T,
        p = p,
        denominator = np.nan,

        gjr_omega = np.nan,
        gjr_alpha = np.nan,
        gjr_gamma = np.nan,
        gjr_beta = np.nan,

        egarch_omega = np.nan,
        egarch_alpha = np.nan,
        egarch_gamma = np.nan,
        egarch_beta = np.nan,

        sv_mu = np.nan,
        sv_phi = np.nan,
        sv_sigma_v = np.nan,

        # Not used
        su_a = np.nan,
        su_b = np.nan,
        su_loc = np.nan,
        su_scale = np.nan,
        
        # Not used
        nct_df = np.nan,
        nct_nc = np.nan,
        nct_loc = np.nan,
        nct_scale = np.nan,

        skew_t_a = np.nan,
        skew_t_b = np.nan,
        skew_t_loc = np.nan,
        skew_t_scale = np.nan,

        arch_lm_pvalue = np.nan,
        f_pvalue = np.nan,
        ljung_box_pvalue = np.nan,
    )


def formula_15( SR: float, skew: float, kurtosis: float, alpha: float, beta: float, T: int ) -> float:
    """
    Variance of the Sharpe ratio (formula (15) from the paper)

    The skew is the skew of the returns.
    The kurtosis is the kurtosis of the innovations.
    """
    phi = alpha + beta
    var = (
        1
        - SR  * ( 1 - beta ) / ( 1 - phi ) * skew
        + SR**2 * ( kurtosis - 1 ) / 4 *
        ( 1 - beta ) ** 2 * ( 1 + phi ) / ( 1 - phi ) / ( 1 - alpha**2 * kurtosis - 2 * alpha * beta - beta**2 )
    ) / T 
    return var


def standardized_student( size: int, df: float = 4 ) -> np.ndarray:
    """
    Generate samples from a Student distribution, with unit variance

    The moments, if defined, are:
    - Mean = 0
    - Variance = 1
    - Skewness = 0
    - Kurtosis = 3 + 6 / ( df - 4 )  (this is the non-excess kurtosis) (only finite if df>4)
    """
    assert df > 2, f"df must be greater than 2, otherwise the variance is infinite; got {df}"
    xs = np.random.standard_t( size = size, df = df )
    variance = df / ( df - 2 )
    return xs / np.sqrt( variance )


def standardized_student_test(): 
    np.random.seed(0)
    n = 1_000_000
    df = 3
    xs = standardized_student( size = n, df = df )
    assert np.abs( xs.mean() - 0 ) < 1e-3
    assert np.abs( xs.std() - 1 ) < 1e-2

    n = 1_000_000
    df = 8
    xs = standardized_student( size = n, df = df )
    sample_kurtosis = 3 + scipy.stats.kurtosis( xs )
    theoretical_kurtosis = 3 + 6 / ( df - 4 )
    assert np.abs( sample_kurtosis - theoretical_kurtosis ) < 1e-1


@numba.njit
def garch_returns( 
    size: int, 
    mu, sigma, alpha, beta,
    innovations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate GARCH(1,1) returns with Student innovations
    Return returns and volatility
    """
    assert alpha >= 0
    assert beta >= 0
    assert alpha + beta < 1
    assert size <= len(innovations)
    n = len(innovations)
    phi = alpha + beta
    omega = ( 1 - phi ) * sigma ** 2
    zs = innovations
    vs = np.zeros( shape = n ) + sigma ** 2
    ys = np.zeros( shape = n )
    ys[0] = sigma * zs[0]
    for t in range( 1, n ):
        vs[t] = omega + alpha * ys[t-1] ** 2 + beta * vs[t-1]
        ys[t] = zs[t] * np.sqrt( vs[t] )
    return (ys + mu)[-size:], np.sqrt(vs[-size:])


def garch_returns_test():
    np.random.seed(0)
    n = 1_000_000
    df = 5
    sigma = .15
    mu = .08
    alpha = .1
    beta = .85
    kurtosis = 3 + 6 / ( df - 4 )
    innovations = standardized_student( size = n + 100, df = df )
    ys, vs = garch_returns( 
        size = n, 
        mu = mu, sigma = sigma, alpha = alpha, beta = beta,
        innovations = innovations,
    )
    assert np.abs( ys.mean() - mu ) < 1e-3
    assert np.abs( ys.std() - sigma ) < 1e-3
    assert np.abs( scipy.stats.skew( ys ) - 0 ) < 1e-1
    assert np.abs( 3 + scipy.stats.kurtosis( innovations )  -  kurtosis ) < 1  # Not very precise: 8.4 instead of 9

    model = arch_model(ys, vol='Garch', p=1, q=1, dist='skewt', mean = 'Constant', rescale=True)
    res = model.fit(disp="off")
    res.summary()
    assert np.abs( res.params['alpha[1]'] - alpha ) < 1e-3
    assert np.abs( res.params['beta[1]'] - beta ) < 1e-3
    assert np.abs( res.params['mu'] / res.scale - mu ) < 1e-3
    innovations = res.resid / res.conditional_volatility
    assert np.abs( 3 + scipy.stats.kurtosis( innovations )  -  kurtosis ) < 1


@numba.njit
def gjr_garch_returns( 
    size: int, 
    mu, sigma,
    omega, alpha, gamma, beta,
    innovations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate GJR-GARCH returns with Student innovations
    Return returns and volatility
    """
    assert alpha >= 0
    assert beta >= 0
    assert alpha + beta + gamma/2< 1
    assert size < len(innovations)  # Includes a burn-in period
    n = len(innovations)
    y_gjr, h_gjr = np.zeros(n), np.zeros(n)
    h_gjr[0] = omega / (1 - alpha - gamma/2 - beta)
    for t in range(1, n):
        I = 1 if y_gjr[t-1] < 0 else 0
        h_gjr[t] = omega + (alpha + gamma * I) * y_gjr[t-1]**2 + beta * h_gjr[t-1]
        y_gjr[t] = np.sqrt(h_gjr[t]) * innovations[t]

    #h_gjr = h_gjr / y_gjr.var()
    #y_gjr = (y_gjr - y_gjr.mean()) / y_gjr.std()  # Is this needed? No.
    
    h_gjr = h_gjr * sigma**2
    y_gjr = y_gjr * sigma + mu
    return y_gjr[-size:], np.sqrt(h_gjr[-size:])


def gjr_garch_returns_test():
    np.random.seed(0)
    n = 1_000
    burnin = 1000
    df = 5
    innovations = standardized_student( size = n + burnin, df = df )
    ys, _ = gjr_garch_returns( 
        size = n, 
        mu = 0, sigma = 1,
        omega = 0.05, alpha = 0.05, gamma = 0.1, beta = 0.8,
        innovations = innovations,
    )
    #return ys


@numba.njit
def egarch_returns(
    size: int,
    mu, sigma,
    omega, alpha, gamma, beta,
    innovations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate EGARCH returns with Student innovations
    Return returns and volatility
    """
    assert alpha >= 0
    assert beta >= 0
    assert beta < 1
    #assert alpha + beta < 1
    assert size < len(innovations)  # Includes a burn-in period
    n = len(innovations)
    y_eg, log_h = np.zeros(n), np.zeros(n)
    log_h[0] = ( omega + alpha * np.mean(np.abs(innovations)) + gamma * np.mean(innovations) ) / ( 1 - beta )
    for t in range(1, n):
        log_h[t] = omega + alpha * np.abs(innovations[t-1]) + gamma * innovations[t-1] + beta * log_h[t-1]
        y_eg[t] = np.exp(log_h[t]/2) * innovations[t]

    #log_h = log_h - np.log(y_eg.var())
    #y_eg = (y_eg - y_eg.mean()) / y_eg.std()  # Is this needed? No.
    
    log_h = log_h + 2 * np.log(sigma)
    y_eg = y_eg * sigma + mu

    return y_eg[-size:], np.exp(log_h/2)[-size:]


def egarch_returns_test():
    np.random.seed(0)
    n = 1_000
    burnin = 1000
    df = 5
    innovations = standardized_student( size = n + burnin, df = df )
    ys, _ = egarch_returns( 
        size = n,
        mu = 0, sigma = 1,
        omega = -0.1, alpha = 0.1, gamma = -0.1, beta = 0.9,
        innovations = innovations,
    )
    #return ys


def sv_features(r: np.ndarray, K: int = 5) -> np.ndarray:
    """
    Compute the features for the stochastic volatility model
    """
    r = r - np.nanmean(r)
    r2 = r**2
    abs_r = np.abs(r)
    log_r2 = np.log(r2 + 1e-8)

    return np.r_[
        np.log(np.var(r) + 1e-12),
        np.mean(log_r2),
        np.var(log_r2),
        np.mean(r2**2) / (np.mean(r2)**2 + 1e-12),
        acf(r2, nlags=K, fft=True)[1:],
        acf(abs_r, nlags=K, fft=True)[1:],
        acf(log_r2, nlags=K, fft=True)[1:]
    ]

@numba.njit
def stochastic_volatility_returns(
    size: int,
    mu, sigma,
    mu_v, phi, sigma_v,
    innovations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate stochastic volatility returns with Student innovations
    Return returns and volatility
    """
    assert size < len(innovations)  # Includes a burn-in period
    n = len(innovations)
    y_sv, h_sv = np.zeros(n), np.zeros(n)
    v = np.random.normal(0, sigma_v, n)
    h_sv[0] = mu_v
    for t in range(1, n):
        h_sv[t] = mu_v + phi * (h_sv[t-1] - mu_v) + v[t]
        y_sv[t] = np.exp(h_sv[t]/2) * innovations[t]

    #h_sv = h_sv - np.log(y_sv.var())
    #y_sv = (y_sv - y_sv.mean()) / y_sv.std()  # Is this needed? No.
    
    h_sv = h_sv + 2 * np.log(sigma)
    y_sv = y_sv * sigma + mu

    return y_sv[-size:], np.exp(h_sv/2)[-size:]

def stochastic_volatility_returns_test():
    np.random.seed(0)
    n = 1_000
    burnin = 1000
    df = 5
    innovations = standardized_student( size = n + burnin, df = df )
    ys, _ = stochastic_volatility_returns( 
        size = n, 
        mu = 0, sigma = 1,
        mu_v = -1.0, phi = 0.95, sigma_v = 0.2,
        innovations = innovations,
    )
    #return ys


def plot_parameters_garch( parameters ):

    fig, axs = plt.subplots( 2, 2, figsize = (6,6), layout = 'constrained', dpi = 300 )
    fig.suptitle("GARCH(1,1) parameters" )

    ax = axs[1,0]
    ax.scatter( parameters['alpha'], parameters['beta'], alpha = .1, s = 10 )
    ax.axline( (0,1), slope = -1, color = 'black', linestyle = ':', linewidth = 1 )
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_aspect(1)

    ax = axs[1,1]
    ax.hist( 
        parameters['phi'], 
        density = True, 
        bins = np.linspace(0, 1, 21),
        facecolor = 'lightblue',
        edgecolor = 'tab:blue',
    )
    ax.set_xlabel(r'$\phi = \alpha + \beta$')
    ax.set_title(r"Distribution of $\phi$")
    ax.set_yticks([])
    ax.set_xlim(-.02,1.02)

    ax = axs[0,0]
    ax.hist( 
        parameters['alpha'], 
        density = True, 
        bins = np.linspace(0,1,21),
        facecolor = 'lightblue',
        edgecolor = 'tab:blue',
    )
    ax.set_xlabel(r'$\alpha$')
    #ax.set_title(r"Distribution of $\alpha$")
    ax.set_yticks([])
    ax.set_xlim(-.02,1.02)

    ax = axs[0,1]
    ax.hist( 
        parameters['beta'], 
        density = True, 
        bins = np.linspace(0,1,21),
        facecolor = 'lightblue',
        edgecolor = 'tab:blue',
    )
    ax.set_xlabel(r'$\beta$')
    #ax.set_title(r"Distribution of $\beta$")
    ax.set_yticks([])
    ax.set_xlim(-.02,1.02)

    for ax in axs.flatten()[[0,1,3]]:
        for side in ['left', 'right', 'top']:
            ax.spines[side].set_visible(False)
    plt.show()


def plot_parameters_gjr( parameters ):
    X = pd.DataFrame( { 
        'alpha': parameters['gjr_alpha'],
        'beta': parameters['gjr_beta'],
        'gamma': parameters['gjr_gamma'],
        'alpha + gamma': parameters['gjr_alpha'] + parameters['gjr_gamma'],
    })
    fig, axs = plt.subplots( 4,4, figsize = (12,12), layout = 'constrained', dpi = 100 )
    fig.suptitle("GJR-GARCH(1,1) parameters" )
    for i, ylabel in enumerate( X.columns ):
        for j, xlabel in enumerate( X.columns ):
            ax = axs[i,j]
            if i != j: 
                ax.scatter( X[xlabel], X[ylabel], alpha = .1, s = 10 )
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                if not 'gamma' in xlabel and not 'gamma' in ylabel:
                    ax.set_aspect(1)
                if 'gamma' != xlabel and 'gamma' != ylabel:
                    ax.axline( (0,1), slope = -1, color = 'black', linestyle = ':', linewidth = 1 )
            else: 
                ax.hist( X[xlabel], density = True, bins = np.linspace(0,1,21), facecolor = 'lightblue', edgecolor = 'tab:blue' )
                ax.set_xlabel(xlabel)
                ax.set_yticks([])
                ax.set_xlim(-.02,1.02)
    plt.show()   


def plot_parameters_egarch( parameters, clip = 2 ):
    # The values of alpha and gamma have a lot of outliers.
    X = pd.DataFrame( { 
        'alpha': np.clip( parameters['egarch_alpha'], -clip, clip ),
        'beta': parameters['egarch_beta'],
        'gamma': np.clip( parameters['egarch_gamma'], -clip, clip ),
        'alpha + gamma': np.clip(parameters['egarch_alpha'] + parameters['egarch_gamma'], -clip, clip),
        'alpha - gamma': np.clip(parameters['egarch_alpha'] - parameters['egarch_gamma'], -clip, clip),
    })
    fig, axs = plt.subplots( 5,5, figsize = (15,15), layout = 'constrained', dpi = 100 )
    fig.suptitle("EGARCH parameters" )
    for i, ylabel in enumerate( X.columns ):
        for j, xlabel in enumerate( X.columns ):
            ax = axs[i,j]
            if i != j: 
                ax.scatter( X[xlabel], X[ylabel], alpha = .1, s = 10 )
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                if not 'gamma' in xlabel and not 'gamma' in ylabel and xlabel != 'alpha' and ylabel != 'alpha':
                    ax.set_aspect(1)
            else: 
                ax.hist( 
                    X[xlabel], density = True, 
                    bins = np.linspace( min(0,X[xlabel].min()), max(1,X[xlabel].max()), 21), 
                    facecolor = 'lightblue', edgecolor = 'tab:blue',
                )
                ax.set_xlabel(xlabel)
                ax.set_yticks([])
                #ax.set_xlim(-.02,1.02)
    plt.show()   
    

def plot_parameters_sv( parameters ):
    X = pd.DataFrame( { 
        'mu': parameters['sv_mu'],
        'phi': parameters['sv_phi'],
        'sigma_v': parameters['sv_sigma_v'],
    } )
    fig, axs = plt.subplots( 3,3, figsize = (9,9), layout = 'constrained', dpi = 100 )
    fig.suptitle("Stochastic volatility parameters" )
    for i, ylabel in enumerate( X.columns ):
        for j, xlabel in enumerate( X.columns ):
            ax = axs[i,j]
            if i != j: 
                ax.scatter( X[xlabel], X[ylabel], alpha = .1, s = 10 )
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
            else: 
                ax.hist( 
                    X[xlabel], density = True, 
                    #bins = np.linspace( min(0,X[xlabel].min()), max(1,X[xlabel].max()), 21), 
                    facecolor = 'lightblue', edgecolor = 'tab:blue',
                )
                ax.set_xlabel(xlabel)
                ax.set_yticks([])
                #ax.set_xlim(-.02,1.02)
    plt.show()   


def plot_parameters_tail_index( parameters ):
    fig, ax = plt.subplots( figsize = (6, 2.5), layout = 'constrained', dpi = 300 )
    _, bin_edges, patches = ax.hist( 
        parameters['tail_index'], 
        bins = np.linspace(0, 20, 41), 
        facecolor = 'lightblue', 
        edgecolor = 'white',
    )
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge < 4:
            patch.set_facecolor('pink')
        else:
            patch.set_facecolor('lightblue')

    for side in ['left', 'right', 'top']:
        ax.spines[side].set_visible(False)
    ax.set_xlabel(r'tail index')
    ax.set_yticks([])
    plt.show()


def plot_parameters_johnson_su( parameters ):
    fig, ax = plt.subplots( figsize = (4, 3), layout = 'constrained' )
    ax.scatter( parameters['su_a'], parameters['su_b'], s = 1, alpha = .1 )
    ax.set_xlim( -100, 50 )
    ax.set_ylim( -10, 100 )
    ax.set_xlabel( r"$a$")
    ax.set_ylabel( r"$b$")
    ax.set_title( "Johnson SU parameters" )
    plt.show()


def plot_parameters_non_central_t( parameters ):
    fig, axs = plt.subplots( 1, 3, figsize = (9, 3), layout = 'constrained' )
    ax = axs[0]
    ax.hist( np.log( parameters['nct_df'] ), bins = 100 )
    xticks = [1, 2, 4, 10, 100]
    ax.set_xlim( np.log(.1), np.log( 1_000_000 ))
    ax.set_xticks( np.log( xticks ), xticks )
    ax.set_title( 'df' )
    ax = axs[1]
    ax.hist( parameters['nct_nc'], bins = 200, density = True )
    ax.set_xlim( -20, 50)
    ax.set_title( 'nc' )
    ax = axs[2]
    ax.scatter( np.log( parameters['nct_df'] ), parameters['nct_nc'], s = 1, alpha = .1 )
    ax.set_xticks( np.log( xticks ), xticks )
    ax.set_ylim( -20, 50 )
    ax.set_xlabel( "Degrees of freedom")
    ax.set_ylabel( "Non-centrality parameter" )
    #ax.scatter( parameters['nct_loc'], parameters['nct_scale'], s = 1, alpha = .1 )
    fig.suptitle( "Parameters of the non-central t-distribution" )
    plt.show()


def plot_parameters_skewed_t( parameters ): 
    fig, ax = plt.subplots( figsize = (3,3), layout = 'constrained', dpi = 300 )
    ax.scatter( 
        parameters['skew_t_a'], parameters['skew_t_b'], 
        s = 1, 
        alpha = 1,
        color = [ 
            'tab:blue' if positive 
            else 'tab:red' 
            for positive in ( parameters['skew_t_a'] > parameters['skew_t_b'] ).values
        ],
    )
    ax.set_xlabel( r"$a$")
    ax.set_ylabel( r"$b$")
    ax.set_title( "Skewed T distribution parameters" )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim( 1, 1000 )
    ax.set_ylim( 1, 1000 )
    ax.set_aspect(1)
    plt.show()


def plot_parameters_moments( parameters ):       
    fig, axs = plt.subplots( 1, 3, figsize = (9,2.5), layout = 'constrained', dpi = 300 )

    ax = axs[0]
    ax.hist( parameters['skew'], bins = 100 )
    ax.set_xlim( -1, 1 )
    ax.set_xlabel( "return skewness" )

    ax = axs[1]
    ax.hist( parameters['kurtosis'], bins = 200)
    ax.axvline( 3, color = 'white', linewidth = 1 )
    ax.set_xlim( 0, 6 )
    ax.set_xlabel( "innovation non-excess kurtosis")

    ax = axs[2]
    ax.hist( parameters['denominator'], bins = np.linspace( -.1, 1, 23 ) )
    ax.axvline( 0, color = 'white', linewidth = 1 )
    ax.set_xlim( -.1, 1.05 )
    ax.set_xlabel( r"$1-\alpha^2 \kappa - 2 \alpha \beta - \beta^2$" )

    for ax in axs: 
        for side in ['left', 'right', 'top']:
            ax.spines[side].set_visible(False)
        ax.set_yticks([])
    plt.show()


def plot_p_values( parameters ):
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.5), layout='constrained', dpi=300, sharey=True)
    for column, ax in zip(['arch_lm_pvalue', 'f_pvalue', 'ljung_box_pvalue'], axs):
        ax.hist( parameters[column], bins = np.linspace(0, 1, 21), density = True)
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
        ax.set_yticks([])
        ax.set_xlabel(column)
    fig.suptitle( 
        "p-values of GARCH(1,1) tests\nH₀: no (G)ARCH effect\n" 
        f"p<0.05 for {int(100 * np.mean( parameters['arch_lm_pvalue'] < .05 ))}% of time series"
    )
    plt.show()


def plot_parameters_all( parameters ):
    plot_parameters_garch( parameters )
    plot_parameters_gjr( parameters )
    plot_parameters_egarch( parameters )
    plot_parameters_sv( parameters )
    plot_parameters_tail_index( parameters )
    plot_parameters_johnson_su( parameters )
    plot_parameters_non_central_t( parameters )
    plot_parameters_skewed_t( parameters )
    plot_parameters_moments( parameters )
    plot_p_values( parameters )


def hplot(x, y=None, ax = None, **kwargs):
    """
    Plot vertical bars

    Contrary to plt.bar, the width is specified as with plt.plot: the bars will not disappear if there is too much data.
    This is similar to plot(..., type='h') in R.

    Inputs: x, y: data to plot; if y is missing, x can have two columns (otherwise, range(n) is used for the x axis)
            ax: where to add the plot
            kwargs: passed to plt.plot
    Output: None

    Example:
        hplot( np.random.normal(size=100), linewidth=1, color='grey' )
    """

    ax_was_None = ax is None

    if isinstance( x, list ):
        x = np.array(x)
    if isinstance( y, list ):
        y = np.array(y)
    # If there is only one argument, either split it in two, or use range(n)
    if y is None:
        if isinstance(x, pd.Series):
            x, y = x.index, x.values
        elif isinstance( x, pd.DataFrame ) and x.shape[1] > 1:
            x, y = x.values[:,0], x.values[:,1]
        elif isinstance( x, pd.DataFrame ):
            x, y = x.index, x.values[:,0]
        elif isinstance( x, np.ndarray ) and len(x.shape) == 2 and x.shape[1] > 1:
            x, y = x[:,0], x[:,1]
        else:
            x, y = np.arange(len(x)), x

    # Convert Pandas objects to Numpy; if they are 2-dimensional, only keep the first column
    if isinstance( x, pd.Series ):
        x = x.values
    if isinstance( y, pd.Series ):
        y = y.values
    if isinstance( x, pd.DataFrame ):
        x = x.values[:,0]
    if isinstance( y, pd.DataFrame ):
        y = y.values[:,0]

    # If we were given 2-dimensional Numpy objects, only keep the first column
    if isinstance( x, np.ndarray ) and len(x.shape) == 2:
        x = x[:,0]
    if isinstance( y, np.ndarray ) and len(y.shape) == 2:
        y = y[:,0]

    # LineCollection only accepts floats, not dates: convert them if needed
    xmin, xmax = x.min(), x.max()
    if isinstance( x, pd.DatetimeIndex ):
        x = matplotlib.dates.date2num( x.to_pydatetime() )

    assert len(x) == len(y), f"{len(x)=} ≠ {len(y)=}"

    if ax_was_None:
        fig, ax = plt.subplots()
    
    segments = [ [ (u,0), (u,v) ] for (u,v) in zip(x,y) ]
    coll = LineCollection( segments, **kwargs )
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.set_xlim( xmin - .02*(xmax-xmin), xmax + .02*(xmax-xmin) )

    if ax_was_None:
        fig.tight_layout()
        plt.show()


def remove_scientific_notation_from_vertical_axis(ax, deprecated_argument=None):
    """
    Remove the scientific notation from the vertical axis tick labels.
    If the scale is logarithmic but spans less than one or two orders of magnitude.
    """

    if deprecated_argument is None:
        fig = ax.get_figure()
    else:
        # The old version of this function was taking fig, ax as argument...
        # TODO: issue a deprecation warning
        fig, ax = ax, deprecated_argument

    fig.canvas.draw()

    def remove_scientific_notation(text = '$\\mathdefault{2\\times10^{-2}}$'):
        if text == '':
            return text
        expr = r'\$\\mathdefault\{((.*)\\times)?10\^\{(.*)\}\}\$'
        mantissa = re.sub( expr, r'\2', text )
        exponent = re.sub( expr, r'\3', text )
        if mantissa == '':
            mantissa = 1
        mantissa = float(mantissa)
        exponent = float(exponent)
        result = mantissa * 10 ** exponent
        return f'{float(f"{result:.4g}"):g}'

    labels = ax.yaxis.get_ticklabels()
    for label in labels:
        a = label.get_text()
        b = remove_scientific_notation(a)
        label.set_text(b)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message = "FixedFormatter should only be used together with FixedLocator" )
        warnings.filterwarnings('ignore', message = "set_ticklabels" )
        ax.yaxis.set_ticklabels(labels)

    labels = ax.yaxis.get_minorticklabels()
    for label in labels:
        a = label.get_text()
        b = remove_scientific_notation(a)
        label.set_text(b)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message = "FixedFormatter should only be used together with FixedLocator" )
        warnings.filterwarnings('ignore', message = "set_ticklabels" )
        ax.yaxis.set_ticklabels(labels, minor=True)


def legend_thick(ax, *args, **kwargs):
    leg = ax.legend(*args, **kwargs)
    for i in leg.legend_handles:
        i.set_linewidth(7)
        i.set_solid_capstyle('butt')

if __name__ == "__main__":
    garch_returns_test()
    standardized_student_test()
    gjr_garch_returns_test()
    egarch_returns_test()
    stochastic_volatility_returns_test()
    print( "All tests passed" )

