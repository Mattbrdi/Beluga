from __future__ import annotations

import json
import shutil
from pathlib import Path
from ..Utils.signal_generation import AudioSceneGenerator
from .config import DatasetConfig
from .io import metadata_to_dict, save_scene, write_json
from .scenarios import get_scenario_factory


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Le dossier {output_dir} n'est pas vide. "
                "Choisis un autre dossier ou utilise overwrite=True."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def build_dataset(
    config: DatasetConfig,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Genere un dataset fixe; une seed unique est affectee a chaque scene.

    Le scenario nomme dans la configuration est resolu via SCENARIO_FACTORIES.
    """
    root = Path(output_dir)
    spec_factory = get_scenario_factory(config.scenario)
    _prepare_output_dir(root, overwrite=overwrite)
    write_json(root / "dataset_config.json", config.to_dict())

    generator = AudioSceneGenerator(**config.generator.__dict__)
    global_index = 0

    for split_name, split_size in config.splits.items():
        split_dir = root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = split_dir / "manifest.jsonl"

        with manifest_path.open("w", encoding="utf-8") as manifest:
            for split_index in range(split_size):
                seed = config.base_seed + global_index
                spec = spec_factory(split_name, split_index, seed)
                scene = generator.generate(spec=spec, seed=seed)
                filename = f"scene_{split_index:06d}.npz"
                save_scene(scene, split_dir / filename, compressed=config.compressed)

                record = {
                    "id": f"{split_name}_{split_index:06d}",
                    "path": filename,
                    "split": split_name,
                    "seed": seed,
                    "metadata": metadata_to_dict(scene),
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                global_index += 1

    return root
