import numpy as np


def estimate_number_of_source_mdl(R, N):
    w, _ = np.linalg.eigh(R)

    w = np.sort(w)[::-1]

    M = R.shape[0]
    mdl_criteron = []
    for i in range(M):
        coef = 1 / (M - i)
        a = coef * np.sum(w[i:])
        g = np.pow(np.prod(w[i:]), coef)
        mdl_criteron.append(
            -np.log(np.pow((g / a), N * (M - i))) + 0.5 * i * (2 * M - i) * np.log(N)
        )

    return np.argmin(mdl_criteron)


def estimate_number_of_source_aic(R, N):
    w, _ = np.linalg.eigh(R)

    w = np.sort(w)[::-1]

    M = R.shape[0]
    aic_criteron = []
    for i in range(M):
        coef = 1 / (M - i)
        a = coef * np.sum(w[i:])
        g = np.pow(np.prod(w[i:]), coef)
        aic_criteron.append(
            (M - i) * N * np.log((coef * a) / np.pow(g, coef)) + i * (2 * M - i)
        )

    return np.argmin(aic_criteron)


def estimate_number_of_source_mdl_freqs(R, N):
    w, _ = np.linalg.eigh(R)

    w = np.sort(w)[::-1]

    M = R.shape[-1]
    mdl_criteron = []
    for i in range(M):
        coef = 1 / (M - i)
        a = coef * np.sum(w[i:])
        g = np.pow(np.prod(w[i:]), coef)
        mdl_criteron.append(
            -np.log(np.pow((g / a), N * (M - i))) + 0.5 * i * (2 * M - i) * np.log(N)
        )

    return np.argmin(mdl_criteron)


def estimate_number_of_source_aic_freqs(R, N):
    w, _ = np.linalg.eigh(R)

    w = np.sort(w)[::-1]

    M = R.shape[-1]
    aic_criteron = []
    for i in range(M):
        coef = 1 / (M - i)
        a = coef * np.sum(w[i:])
        g = np.pow(np.prod(w[i:]), coef)
        aic_criteron.append(
            (M - i) * N * np.log((coef * a) / np.pow(g, coef)) + i * (2 * M - i)
        )

    return np.argmin(aic_criteron)
