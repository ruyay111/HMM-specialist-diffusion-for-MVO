import numpy as np
import cvxpy as cp


def global_minimum_variance_portfolio(returns, min_weight=0.0):
    cov_matrix = np.cov(returns, rowvar=False)
    n_assets = returns.shape[1]
    cov_matrix += 1e-4 * np.eye(n_assets)

    weights = cp.Variable(n_assets)
    objective = cp.Minimize(cp.quad_form(weights, cov_matrix))
    constraints = [cp.sum(weights) == 1, weights >= min_weight]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return weights.value


def mean_variance_portfolio(returns, risk_aversion=1.0, min_weight=0.0):
    mu = returns.mean(axis=0)
    cov_matrix = np.cov(returns, rowvar=False)
    n_assets = returns.shape[1]
    cov_matrix += 1e-4 * np.eye(n_assets)

    weights = cp.Variable(n_assets)
    objective = cp.Minimize(-mu @ weights + risk_aversion * cp.quad_form(weights, cov_matrix))
    constraints = [cp.sum(weights) == 1, weights >= min_weight]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return weights.value


def maximize_sharpe_ratio_portfolio(returns, risk_free_rate=0.0, min_weight=0.0):
    mu = returns.mean(axis=0) - risk_free_rate
    cov = np.cov(returns, rowvar=False)
    n_assets = returns.shape[1]
    cov += 1e-4 * np.eye(n_assets)

    L = np.linalg.cholesky(cov)

    weights = cp.Variable(n_assets)
    t = cp.Variable()

    constraints = [
        cp.sum(weights) == 1,
        weights >= min_weight,
        cp.norm(L @ weights, 2) <= t
    ]

    objective = cp.Maximize(mu @ weights)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return weights.value


def minimize_cvar_portfolio(returns, alpha=0.95, min_weight=0.0):
    n_samples = returns.shape[0]
    n_assets = returns.shape[1]

    weights = cp.Variable(n_assets)
    z = cp.Variable(n_samples)
    VaR = cp.Variable()

    portfolio_returns = returns @ weights
    losses = -portfolio_returns

    constraints = [
        cp.sum(weights) == 1,
        weights >= min_weight,
        z >= 0,
        losses - VaR <= z
    ]

    cvar_objective = VaR + (1 / (1 - alpha)) * cp.sum(z) / n_samples
    objective = cp.Minimize(cvar_objective)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return weights.value
