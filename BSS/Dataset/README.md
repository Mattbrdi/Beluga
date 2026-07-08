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
Le fichier `dataset_config.json` produit contient aussi un instantane des cles
des registres effectivement charges, la version du format, le commit Git et un
indicateur signalant si le code comportait des modifications non commitees.

Le champ `scenario` de la configuration selectionne une fabrique enregistree
dans `SCENARIO_FACTORIES`. Le scenario `default` applique la generation
aleatoire sans contrainte supplementaire. Le scenario `whistles_only` limite
les sources utiles aux sifflements, sans modifier la generation des bruits.

Pour charger une scene :

```python
from BSS.Dataset import load_scene

scene = load_scene("BSS/Dataset/generated/benchmark_v1/test/scene_000000.npz")
mixture = scene.mixed          # MultiSignal donne a l'algorithme BSS
references = scene.sources     # MultiSignal de reference pour l'evaluation
```

`load_scene_arrays(path)` reste disponible lorsqu'un traitement a besoin des
tableaux NumPy bruts plutot que des objets `AudioScene` et `MultiSignal`.

Pour ajouter un scenario reproductible dans `BSS/Dataset/scenarios.py` :

```python
@register_scenario("hard")
def hard_scenario(split: str, index: int, seed: int) -> AudioSceneSpec:
    return AudioSceneSpec(snr_db=0.0)
```

Le JSON peut ensuite utiliser `"scenario": "hard"`. Toute decision aleatoire
de la fabrique doit etre derivee de la `seed` recue. Le champ `snr_db` fixe le
rapport, en decibels, entre l'energie totale du melange propre sur les micros et
celle des bruits continus; les bruits locaux ne participent pas a ce calcul.
