# Dataset synthetique BSS

Ce dossier transforme `BSS.Utils.signal_generation.AudioSceneGenerator` en un
dataset fixe et reproductible. La logique de synthese reste dans `Utils`; ce
module gere uniquement la configuration, les splits et le stockage.

## Generation

Depuis le dossier `Beluga` :

```powershell
python -m BSS.Dataset.generate_dataset `
  --config BSS/Dataset/default_config.json `
  --output BSS/Dataset/generated/benchmark_v1
```

Le dossier de sortie doit etre vide. `--overwrite` autorise explicitement son
remplacement.

## Format produit

```text
benchmark_v1/
|-- dataset_config.json
|-- train/
|   |-- manifest.jsonl
|   `-- scene_000000.npz
|-- validation/
`-- test/
```

Chaque fichier de scene contient :

- `mixed` : entree du module de separation, shape `(n_mics, n_samples)` ;
- `sources` : references propres, shape `(n_sources, n_samples)` ;
- `clean_mixed` et `noise` : signaux utiles aux analyses par niveau de bruit ;
- `mixing_filters` : filtres/retards reels du melange ;
- `fs` : frequence d'echantillonnage ;
- `metadata_json` : metadonnees necessaires pour reconstruire une `AudioScene`.

Le manifeste contient le chemin, le split, la seed et les metadonnees de chaque
scene. Une configuration et une seed identiques reproduisent le meme dataset.

Pour charger une scene :

```python
from BSS.Dataset import load_scene

scene = load_scene("BSS/Dataset/generated/benchmark_v1/test/scene_000000.npz")
mixture = scene.mixed          # MultiSignal donne a l'algorithme BSS
references = scene.sources     # MultiSignal de reference pour l'evaluation
```

`load_scene_arrays(path)` reste disponible lorsqu'un traitement a besoin des
tableaux NumPy bruts plutot que des objets `AudioScene` et `MultiSignal`.

`build_dataset(..., spec_factory=...)` accepte aussi une fonction qui retourne
un `AudioSceneSpec` par scene. C'est le point d'extension prevu pour construire
des scenarios `easy`, `nominal`, `hard` ou hors distribution.
