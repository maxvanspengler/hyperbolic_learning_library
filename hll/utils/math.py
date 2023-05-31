from math import exp, lgamma


def beta_func(a: int, b: int) -> float:
    log_beta = lgamma(a) + lgamma(b) - lgamma(a + b)
    return exp(log_beta)
