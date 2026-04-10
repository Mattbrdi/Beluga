import pandas as pd
import os

# Fichiers d'entrée
file_8295 = r"\\lisse-tache.uqo.ca\Commun\Pour Irene\4-canaux Cacouna\Output\detections_8295.csv"
file_8296 = r"\\lisse-tache.uqo.ca\Commun\Pour Irene\4-canaux Cacouna\Output\detections_8296.csv"

# Fichier de sortie
output_file = r"\\lisse-tache.uqo.ca\Commun\Pour Irene\4-canaux Cacouna\Output\detections_common_8295_8296.csv"

# Lecture
df_8295 = pd.read_csv(file_8295)
df_8296 = pd.read_csv(file_8296)

# S'assurer que Timestamp est bien comparable
df_8295["Timestamp"] = pd.to_datetime(df_8295["Timestamp"])
df_8296["Timestamp"] = pd.to_datetime(df_8296["Timestamp"])

# Trouver timestamps communs
common_timestamps = sorted(
    set(df_8295["Timestamp"]).intersection(set(df_8296["Timestamp"]))
)

rows = []

for ts in common_timestamps:
    row_8296 = df_8296[df_8296["Timestamp"] == ts]
    row_8295 = df_8295[df_8295["Timestamp"] == ts]

    if not row_8296.empty:
        rows.append(row_8296.iloc[0])
    if not row_8295.empty:
        rows.append(row_8295.iloc[0])

# Créer DataFrame final
if rows:
    df_common = pd.DataFrame(rows)
else:
    df_common = pd.DataFrame(columns=df_8295.columns)

# Écriture (écrase toujours)
df_common.to_csv(output_file, index=False)

print(f"[OK] CSV commun écrit : {output_file}")
print(f"Nombre de timestamps communs : {len(common_timestamps)}")
