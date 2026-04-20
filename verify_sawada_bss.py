from __future__ import annotations

import importlib.util
from pathlib import Path
from importlib.machinery import SourceFileLoader

import numpy as np

from signal_class import NSpectrogram


def _load_sawada_module():
    module_path = Path(__file__).resolve().parent / "Sawada_separation"
    loader = SourceFileLoader("sawada_module", str(module_path))
    spec = importlib.util.spec_from_loader("sawada_module", loader)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger le module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _best_binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    acc_direct = np.mean(y_true == y_pred)
    acc_flip = np.mean(y_true == (1 - y_pred))
    return max(acc_direct, acc_flip)


def verify_em_clustering(seed: int = 0) -> float:
    """
    Vérifie que le coeur EM du SawadaBSS sépare correctement 2 directions
    de source synthétiques sur la sphère complexe.
    """
    rng = np.random.default_rng(seed)
    sawada_module = _load_sawada_module()
    EMClustering = sawada_module.EMClustering

    e_num = 3
    n_points = 1800

    a1 = np.array([1.0 + 0.0j, 0.6 + 0.2j, -0.3 + 0.4j])
    a2 = np.array([0.1 + 0.8j, -0.9 + 0.0j, 0.5 - 0.1j])
    a1 = a1 / np.linalg.norm(a1)
    a2 = a2 / np.linalg.norm(a2)

    labels = rng.integers(0, 2, size=n_points)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=n_points))
    noise = 0.08 * (
        rng.standard_normal((e_num, n_points)) + 1j * rng.standard_normal((e_num, n_points))
    )

    X = np.zeros((e_num, n_points), dtype=np.complex128)
    for n in range(n_points):
        steer = a1 if labels[n] == 0 else a2
        X[:, n] = steer * phases[n] + noise[:, n]

    X = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-12)

    model = EMClustering(n_sources=2, n_iter=30, phi=1.0)
    model.fit(X)
    masks = model.predict(X)
    pred = np.argmax(masks, axis=0)
    return _best_binary_accuracy(labels, pred)


def verify_whitening(seed: int = 1) -> float:
    """
    Vérifie que le blanchiment de NSpectrogram rend la corrélation spatiale
    proche de l'identité.
    """
    rng = np.random.default_rng(seed)
    e_num, f_num, t_num = 3, 24, 30
    X = (
        rng.standard_normal((e_num, f_num, t_num))
        + 1j * rng.standard_normal((e_num, f_num, t_num))
    )

    mixing = np.array(
        [
            [1.0 + 0j, 0.3 - 0.1j, 0.2 + 0.4j],
            [0.1 + 0.5j, 1.2 + 0j, -0.3 + 0.2j],
            [0.2 - 0.2j, 0.4 + 0.2j, 0.9 + 0j],
        ]
    )
    X_corr = np.einsum("ij,jft->ift", mixing, X)

    spectro = NSpectrogram(
        f=np.arange(f_num),
        t=np.arange(t_num),
        Sxx=X_corr,
        fs=16000,
        window="hann",
        nperseg=256,
    )

    eigvals, eigvecs = spectro.decompose_spatial_correlation()
    W = spectro.compute_whitening_matrix(eigvals, eigvecs)
    X_white = spectro.apply_transformation(W)

    X_flat = X_white.Sxx.reshape(e_num, -1)
    R = (X_flat @ X_flat.conj().T) / X_flat.shape[1]
    err = np.linalg.norm(R - np.eye(e_num))
    return float(err)


def main():
    accuracy = verify_em_clustering(seed=42)
    whitening_error = verify_whitening(seed=7)

    print(f"EM clustering accuracy (best permutation): {accuracy:.4f}")
    print(f"Whitening Frobenius error to identity: {whitening_error:.4e}")

    if accuracy < 0.90:
        raise SystemExit("ECHEC: précision de séparation trop faible (< 0.90).")
    if whitening_error > 2e-1:
        raise SystemExit("ECHEC: blanchiment trop éloigné de l'identité (> 2e-1).")

    print("OK: vérification SawadaBSS réussie.")


if __name__ == "__main__":
    main()
