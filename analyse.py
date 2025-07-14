import pandas as pd
import matplotlib.pyplot as plt
import collections

# Charger le fichier
df = pd.read_csv("nouveau.csv", sep=';')

# Les phases à analyser
phases = ['VEIN', 'TARD', 'PORT', 'ART']

# Pour chaque classe, compter le nombre de patients ayant 1, 2 ou 3 phases manquantes
classes = sorted(df['classe_name'].unique())
x = [1, 2, 3]

plt.figure(figsize=(10, 6))

for classe in classes:
    df_classe = df[df['classe_name'] == classe].copy()
    manquantes = []
    for idx, row in df_classe.iterrows():
        nb_nan = 0
        for phase in phases:
            col = f"{phase}_original_firstorder_10Percentile"
            if col in df_classe.columns:
                if pd.isna(row[col]):
                    nb_nan += 1
            else:
                nb_nan += 1
        manquantes.append(nb_nan)
    compte = collections.Counter(manquantes)
    y = [compte.get(i, 0) for i in x]
    plt.bar([i + 0.2*classe for i in x], y, width=0.2, label=f"Classe {classe}")

plt.xlabel("Nombre de phases manquantes")
plt.ylabel("Nombre de patients")
plt.title("Nombre de patients selon le nombre de phases manquantes (par classe)")
plt.xticks(x)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', which='both', linestyle='--', alpha=0.7)
plt.yticks(range(0, 16, 1))  # graduation de 0 à 15, un trait par entier
plt.ylim(0, 15)
plt.show()