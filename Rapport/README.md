# Rapport LaTeX

Le fichier principal est `main.tex`. Il s'agit d'un rapport LaTeX classique au
format article A4 qui reprend la structure suivante :

- introduction du pipeline ;
- creation du masque temps-frequence ;
- separation de sources par clustering EM et RANSAC circulaire ;
- calcul/estimation des TDOA sur bins isoles ;
- validation sur dataset synthetique.

Compilation :

```bash
cd Rapport
pdflatex -interaction=nonstopmode main.tex
```

Pour completer la table de resultats, generer ou reutiliser un dataset, puis
lancer le benchmark :

```bash
python -m BSS.Dataset.generate_dataset \
  --config BSS/Dataset/default_config.json \
  --output BSS/Dataset/generated/benchmark_v1

python -m BSS.Benchmark.run_benchmark \
  --dataset BSS/Dataset/generated/benchmark_v1 \
  --split test \
  --output BSS/Benchmark/results/benchmark_v1 \
  --algorithms sawada ica \
  --reference-microphone 0
```

Les valeurs globales a reporter se trouvent dans
`BSS/Benchmark/results/benchmark_v1/summary.csv`.
