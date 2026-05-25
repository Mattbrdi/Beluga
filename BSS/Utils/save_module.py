
"""
Ce module sert à sauvegarder les résultats d'une séparation de source et représente la manière standard de le faire ici
Lire une séparation de source se fait donc de la manière inverse avec ces noms de fichier etc
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .associated_dataclasses import (
    EMClusteringParameters,
    SawadaBssParameters,
    StftParameters,
)
from .signal_class import Mixture, MultiSignal, Signal

if TYPE_CHECKING:
    from ..Algo_Separation.Sawada_separation import SawadaBSS


@dataclass
class SeparationResultBundle:
    original_sources: list[MultiSignal]
    mixture: MultiSignal
    parameters: SawadaBssParameters
    separated_sources: list[MultiSignal]
    applied_mixture: Mixture | None = None
    metadata: dict[str, Any] | None = None
    sawada_model: "SawadaBSS | None" = None


def _normalize_json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize_json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _rebuild_sawada_parameters(raw_parameters: dict[str, Any]) -> SawadaBssParameters:
    return SawadaBssParameters(
        n_sources=raw_parameters["n_sources"],
        stft_parameters=StftParameters(**raw_parameters["stft_parameters"]),
        em_clustering_parameters=EMClusteringParameters(
            **raw_parameters["em_clustering_parameters"]
        ),
        whitening=raw_parameters["whitening"],
    )


def _validate_output_dir(output_dir: str | Path, overwrite: bool) -> Path:
    target_dir = Path(output_dir)
    if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Le dossier '{target_dir}' existe deja et n'est pas vide. "
            "Utilise overwrite=True pour l'ecraser."
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def save_signal(signal: Signal, output_path: str | Path) -> Path:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target_path, data=signal.data, freq=signal.freq)
    return target_path


def load_signal(input_path: str | Path) -> Signal:
    with np.load(Path(input_path), allow_pickle=False) as payload:
        data = np.asarray(payload["data"])
        freq = float(payload["freq"])
    return Signal(data=data, freq=freq)


def save_multisignal(multisignal: MultiSignal, output_path: str | Path) -> Path:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(signal.data) for signal in multisignal.signals], dtype=int)
    np.savez(
        target_path,
        data=multisignal.data,
        freq=multisignal.freq,
        lengths=lengths,
    )
    return target_path


def load_multisignal(input_path: str | Path) -> MultiSignal:
    with np.load(Path(input_path), allow_pickle=False) as payload:
        data = np.asarray(payload["data"])
        freq = float(payload["freq"])
        lengths = np.asarray(payload["lengths"], dtype=int)

    signals = [Signal(data=row[:length], freq=freq) for row, length in zip(data, lengths)]
    return MultiSignal(signals)


def save_mixture(mixture: Mixture, output_path: str | Path) -> Path:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target_path,
        E=mixture.E,
        S=mixture.S,
        L=mixture.L,
        filters=mixture.filters,
    )
    return target_path


def load_mixture(input_path: str | Path) -> Mixture:
    with np.load(Path(input_path), allow_pickle=False) as payload:
        E = int(payload["E"])
        S = int(payload["S"])
        L = int(payload["L"])
        filters = np.asarray(payload["filters"])

    mixture = Mixture(E=E, S=S, L=L)
    mixture.filters = filters
    return mixture


def _save_multisignal_collection(
    multisignals: list[MultiSignal],
    output_dir: Path,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, multisignal in enumerate(multisignals):
        save_multisignal(multisignal, output_dir / f"{prefix}_{idx}.npz")


def _load_multisignal_collection(
    input_dir: Path,
    prefix: str,
    expected_count: int,
) -> list[MultiSignal]:
    return [
        load_multisignal(input_dir / f"{prefix}_{idx}.npz")
        for idx in range(expected_count)
    ]


def _save_nspectrogram(npz_path: Path, nspectrogram: Any) -> None:
    np.savez(
        npz_path,
        f=nspectrogram.f,
        t=nspectrogram.t,
        Sxx=nspectrogram.Sxx,
        fs=nspectrogram.fs,
        window=np.array(nspectrogram.window),
        nperseg=nspectrogram.nperseg,
        noverlap=-1 if nspectrogram.noverlap is None else nspectrogram.noverlap,
        nfft=-1 if nspectrogram.nfft is None else nspectrogram.nfft,
        boundary=np.array("" if nspectrogram.boundary is None else nspectrogram.boundary),
        padded=nspectrogram.padded,
        signal_lengths=(
            np.array([], dtype=int)
            if nspectrogram.signal_lengths is None
            else nspectrogram.signal_lengths
        ),
    )


def _load_nspectrogram(npz_path: Path) -> Any:
    from .signal_class import NSpectrogram

    with np.load(npz_path, allow_pickle=False) as payload:
        noverlap_raw = int(payload["noverlap"])
        nfft_raw = int(payload["nfft"])
        boundary_raw = str(payload["boundary"])
        signal_lengths = np.asarray(payload["signal_lengths"], dtype=int)

        return NSpectrogram(
            f=np.asarray(payload["f"]),
            t=np.asarray(payload["t"]),
            Sxx=np.asarray(payload["Sxx"]),
            fs=float(payload["fs"]),
            window=str(payload["window"]),
            nperseg=int(payload["nperseg"]),
            noverlap=None if noverlap_raw < 0 else noverlap_raw,
            nfft=None if nfft_raw < 0 else nfft_raw,
            boundary=None if boundary_raw == "" else boundary_raw,
            padded=bool(payload["padded"]),
            signal_lengths=None if signal_lengths.size == 0 else signal_lengths,
        )


def _save_sawada_model(model_dir: Path, sawada_model: "SawadaBSS") -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    model_state: dict[str, Any] = {
        "has_nspectro_preprocessed": sawada_model.nspectro_preprocessed is not None,
        "has_eigenvalues_matrix": sawada_model.eigenvalues_matrix is not None,
        "has_eigenvector_matrix": sawada_model.eigenvector_matrix is not None,
        "has_signal": sawada_model.signal is not None,
        "bin_model_indices": sorted(int(idx) for idx in sawada_model.bin_models.keys()),
        "bin_mask_indices": sorted(int(idx) for idx in sawada_model.bin_masks.keys()),
    }

    if sawada_model.nspectro_preprocessed is not None:
        _save_nspectrogram(model_dir / "nspectro_preprocessed.npz", sawada_model.nspectro_preprocessed)

    if sawada_model.signal is not None:
        save_multisignal(sawada_model.signal, model_dir / "signal.npz")

    if sawada_model.eigenvalues_matrix is not None:
        np.save(model_dir / "eigenvalues_matrix.npy", sawada_model.eigenvalues_matrix)

    if sawada_model.eigenvector_matrix is not None:
        np.save(model_dir / "eigenvector_matrix.npy", sawada_model.eigenvector_matrix)

    if sawada_model.bin_masks:
        sorted_mask_indices = sorted(int(idx) for idx in sawada_model.bin_masks.keys())
        mask_stack = np.stack([sawada_model.bin_masks[idx] for idx in sorted_mask_indices], axis=0)
        np.savez(
            model_dir / "bin_masks.npz",
            indices=np.array(sorted_mask_indices, dtype=int),
            masks=mask_stack,
        )

    if sawada_model.bin_models:
        sorted_model_indices = sorted(int(idx) for idx in sawada_model.bin_models.keys())
        centroids = np.stack(
            [sawada_model.bin_models[idx].centroids for idx in sorted_model_indices],
            axis=0,
        )
        variances = np.stack(
            [sawada_model.bin_models[idx].variances for idx in sorted_model_indices],
            axis=0,
        )
        weights = np.stack(
            [sawada_model.bin_models[idx].weights for idx in sorted_model_indices],
            axis=0,
        )
        np.savez(
            model_dir / "bin_models.npz",
            indices=np.array(sorted_model_indices, dtype=int),
            centroids=centroids,
            variances=variances,
            weights=weights,
        )

    state_path = model_dir / "state.json"
    state_path.write_text(
        json.dumps(_normalize_json_value(model_state), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _load_sawada_model(model_dir: Path, parameters: SawadaBssParameters) -> "SawadaBSS":
    from ..Algo_Separation.Sawada_separation import EMClustering, SawadaBSS

    state_path = model_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))

    sawada_model = SawadaBSS(
        n_sources=parameters.n_sources,
        stft_parameters=parameters.stft_parameters,
        em_clustering_parameters=parameters.em_clustering_parameters,
        whitening=parameters.whitening,
    )

    if state.get("has_signal"):
        sawada_model.signal = load_multisignal(model_dir / "signal.npz")

    if state.get("has_nspectro_preprocessed"):
        sawada_model.nspectro_preprocessed = _load_nspectrogram(
            model_dir / "nspectro_preprocessed.npz"
        )

    if state.get("has_eigenvalues_matrix"):
        sawada_model.eigenvalues_matrix = np.load(
            model_dir / "eigenvalues_matrix.npy",
            allow_pickle=False,
        )

    if state.get("has_eigenvector_matrix"):
        sawada_model.eigenvector_matrix = np.load(
            model_dir / "eigenvector_matrix.npy",
            allow_pickle=False,
        )

    bin_masks_path = model_dir / "bin_masks.npz"
    if bin_masks_path.exists():
        with np.load(bin_masks_path, allow_pickle=False) as payload:
            indices = np.asarray(payload["indices"], dtype=int)
            masks = np.asarray(payload["masks"])
        sawada_model.bin_masks = {
            int(idx): masks[pos]
            for pos, idx in enumerate(indices)
        }

    bin_models_path = model_dir / "bin_models.npz"
    if bin_models_path.exists():
        with np.load(bin_models_path, allow_pickle=False) as payload:
            indices = np.asarray(payload["indices"], dtype=int)
            centroids = np.asarray(payload["centroids"])
            variances = np.asarray(payload["variances"])
            weights = np.asarray(payload["weights"])

        reconstructed_models = {}
        for pos, idx in enumerate(indices):
            model = EMClustering(
                n_sources=parameters.n_sources,
                n_iter=parameters.em_clustering_parameters.n_iter,
                phi=parameters.em_clustering_parameters.phi,
                eps=parameters.em_clustering_parameters.eps,
            )
            model.centroids = centroids[pos]
            model.variances = variances[pos]
            model.weights = weights[pos]
            reconstructed_models[int(idx)] = model
        sawada_model.bin_models = reconstructed_models

    return sawada_model


def save_separation_result(
    output_dir: str | Path,
    original_sources: list[MultiSignal],
    mixture: MultiSignal,
    separated_sources: list[MultiSignal],
    parameters: SawadaBssParameters,
    applied_mixture: Mixture | None = None,
    sawada_model: "SawadaBSS | None" = None,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    target_dir = _validate_output_dir(output_dir=output_dir, overwrite=overwrite)

    save_multisignal(mixture, target_dir / "mixture.npz")
    _save_multisignal_collection(
        multisignals=original_sources,
        output_dir=target_dir / "original_sources",
        prefix="original_source",
    )
    _save_multisignal_collection(
        multisignals=separated_sources,
        output_dir=target_dir / "separated_sources",
        prefix="separated_source",
    )
    if applied_mixture is not None:
        save_mixture(applied_mixture, target_dir / "applied_mixture.npz")
    if sawada_model is not None:
        _save_sawada_model(target_dir / "sawada_model", sawada_model)

    manifest = {
        "format": "bss_separation_result",
        "version": 2,
        "original_sources_count": len(original_sources),
        "separated_sources_count": len(separated_sources),
        "has_applied_mixture": applied_mixture is not None,
        "has_sawada_model": sawada_model is not None,
        "parameters": _normalize_json_value(parameters),
        "metadata": _normalize_json_value(metadata or {}),
    }

    manifest_path = target_dir / "metadata.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return target_dir


def save_sawada_result(
    sawada_model: "SawadaBSS",
    output_dir: str | Path,
    original_sources: list[MultiSignal],
    separated_sources: list[MultiSignal] | None = None,
    mixture: MultiSignal | None = None,
    applied_mixture: Mixture | None = None,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    mixture_to_save = mixture if mixture is not None else sawada_model.signal
    if mixture_to_save is None:
        raise ValueError(
            "Aucun melange disponible. Passe mixture=... ou execute process_signal avant save_sawada_result."
        )

    separated_to_save = (
        separated_sources if separated_sources is not None else sawada_model.separate_source()
    )

    return save_separation_result(
        output_dir=output_dir,
        original_sources=original_sources,
        mixture=mixture_to_save,
        separated_sources=separated_to_save,
        parameters=sawada_model.parameters,
        applied_mixture=applied_mixture,
        sawada_model=sawada_model,
        metadata=metadata,
        overwrite=overwrite,
    )


def load_separation_result(input_dir: str | Path) -> SeparationResultBundle:
    source_dir = Path(input_dir)
    manifest_path = source_dir / "metadata.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("format") != "bss_separation_result":
        raise ValueError("Format de sauvegarde non supporte.")

    parameters = _rebuild_sawada_parameters(manifest["parameters"])
    mixture = load_multisignal(source_dir / "mixture.npz")
    original_sources = _load_multisignal_collection(
        input_dir=source_dir / "original_sources",
        prefix="original_source",
        expected_count=int(manifest["original_sources_count"]),
    )
    separated_sources = _load_multisignal_collection(
        input_dir=source_dir / "separated_sources",
        prefix="separated_source",
        expected_count=int(manifest["separated_sources_count"]),
    )
    applied_mixture = None
    if manifest.get("has_applied_mixture"):
        applied_mixture = load_mixture(source_dir / "applied_mixture.npz")
    sawada_model = None
    if manifest.get("has_sawada_model"):
        sawada_model = _load_sawada_model(source_dir / "sawada_model", parameters)

    return SeparationResultBundle(
        original_sources=original_sources,
        mixture=mixture,
        applied_mixture=applied_mixture,
        separated_sources=separated_sources,
        parameters=parameters,
        metadata=manifest.get("metadata", {}),
        sawada_model=sawada_model,
    )
