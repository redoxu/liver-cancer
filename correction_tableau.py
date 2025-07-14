import pandas as pd
import numpy as np
# Charger le fichier Excel (remplace 'chemin_vers_fichier.xlsx' par ton fichier)
fichier = 'Relectures_imageries.xlsx'
df = pd.read_excel(fichier)

# Vérifier qu'on a bien 21 colonnes
if df.shape[1] != 21:
    raise ValueError(f"Le fichier doit avoir exactement 21 colonnes, mais en a {df.shape[1]}.")

# Remplacer les valeurs vides (NaN) par False
df_filled = df.fillna(False)

# Optionnel : afficher les premières lignes du tableau
print(df_filled.head())

# Pour exporter vers un nouveau fichier Excel (optionnel)
df_filled.to_excel("donnees_patients_proprees.xlsx", index=False)

print(df["Diffusion_targetoid"].value_counts())

print(df["Diffusion_targetoid"].apply(type).value_counts())
print(df["Diffusion_targetoid"].unique())

import pandas as pd
import numpy as np

def remplir_colonne_binaire(df, colonne):
    df = df.copy()

    if colonne not in df.columns:
        raise ValueError(f"La colonne '{colonne}' n'existe pas dans le DataFrame.")

    # Masque plus strict : uniquement les booléens False, pas 0
    mask_false = df[colonne].apply(lambda x: x is False)

    print(f"Nombre de False exacts dans '{colonne}': {mask_false.sum()}")

    indices_a_remplir = df.index[mask_false]

    # Les valeurs valides (tout sauf ces False), converties en float
    valeurs_valides = df.loc[~mask_false, colonne].astype(float)

    print(f"Valeurs valides pour calcul de la probabilité :")
    print(valeurs_valides.value_counts())

    # Calcul de la proba 1
    if len(valeurs_valides.unique()) == 1:
        proba_1 = 0.5
        print("Toutes les valeurs valides sont identiques, proba_1 fixée à 0.5")
    else:
        proba_1 = valeurs_valides.mean()
        print(f"Probabilité calculée de 1 : {proba_1}")

    # Tirage aléatoire pour remplacer uniquement les False
    valeurs_generees = np.random.choice([0, 1], size=len(indices_a_remplir), p=[1 - proba_1, proba_1])

    print(f"Valeurs générées pour remplissage : {np.bincount(valeurs_generees)}")

    df.loc[indices_a_remplir, colonne] = valeurs_generees
    print(df[colonne].apply(type).value_counts())
    print(df.loc[df[colonne].apply(lambda x: x is False), colonne])
    print(df.loc[df[colonne] == 0, colonne])

    return df

# Supposons que df_filled est déjà ton DataFrame avec des False pour les valeurs manquantes
df_corrigé = remplir_colonne_binaire(df_filled, "Diffusion_targetoid")
# Optionnel : afficher les premières lignes du tableau
print(df_corrigé.head())
df_corrigé.to_excel("/Users/amram/Desktop/centrale1A/EI cancer/EI-ST4/final_tableau.xlsx", index=False, engine='openpyxl')
