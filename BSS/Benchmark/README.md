# Benchmark BSS

Ce module lance Sawada et/ou ICA sur les scenes d'un dataset genere avec
`BSS.Dataset`, sauvegarde les sources separees et compare les TDOA estimes aux
retards de reference.

## Verification sur une scene

```bash
python3 -m BSS.Benchmark.run_benchmark \
  --dataset BSS/Dataset/generated/boat_and_whistle_v2 \
  --split test \
  --output BSS/Benchmark/results/boat_and_whistle_v2_check \
  --algorithms sawada ica \
  --reference-microphone 0 \
  --limit 1
```

## Split complet

```bash
python3 -m BSS.Benchmark.run_benchmark \
 --dataset BSS/Dataset/generated/boat_and_whistle_v2 \
  --split test \
  --output BSS/Benchmark/results/boat_and_whistle_v2 \
  --algorithms sawada ica \
  --reference-microphone 0
```

## Sorties

Pour chaque scene :

```text
results/<split>/<scene_id>/
|-- sawada_sources.npz
|-- sawada_metrics.json
|-- sawada_model.npz
|-- ica_sources.npz
`-- ica_metrics.json
```

A la racine de sortie :

```text
summary.csv
```

Les TDOA pairwise suivent l'ordre :

```text
M1M2, M1M3, M1M4, M2M3, M2M4, M3M4
```

avec la convention `M_iM_j = delay(M_j) - delay(M_i)`.

## Visualisation des resultats

```bash
python3 -m BSS.Benchmark.visualize_results \
  --results BSS/Benchmark/results/boat_and_whistle_v2 \
  --dataset BSS/Dataset/generated/boat_and_whistle_v2
```

L'interface Dash permet de parcourir les splits, scenes et algorithmes, puis de
visualiser les sources originales, les melanges, les estimations, leurs
spectrogrammes, l'audio associe et les TDOA estimes/alignees face aux valeurs de
reference. Pour Sawada, elle affiche aussi les masques temps-frequence si
`sawada_model.npz` a ete genere par le benchmark.
