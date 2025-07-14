import pandas as pd

# Lire le fichier CSV complet
df = pd.read_csv("radiomiques_multislice.csv", sep=';')

# Supprimer les colonnes commençant par "diagnostics"
df = df.loc[:, ~df.columns.str.startswith("diagnostics")]
df = pd.concat([df["classe_name"],df["temps_inj"],df.select_dtypes(include=["number"])], axis=1)

# Ajouter une colonne pour l'ordre d'origine
df['ordre_origine'] = range(len(df))

# Créer un identifiant patient unique basé sur patient_num + classe_name
df['patient_id'] = df['patient_num'].astype(str) + "_" + df['classe_name'].astype(str)+ "_" +df['slice_num'].astype(str)

# Liste des colonnes à pivoter (toutes sauf patient_num, classe_name, temps_inj)
cols_to_pivot = [col for col in df.columns if col not in ['patient_num', 'classe_name','slice_num', 'temps_inj', 'patient_id', 'ordre_origine']]

# On garde le premier ordre d'apparition de chaque patient_id
ordre_patients = df.groupby('patient_id')['ordre_origine'].min().sort_values().index.tolist()

# Pivot par phase (temps_inj), index = patient_id, valeurs = variables à pivoter
df_pivot = df.pivot(index='patient_id', columns='temps_inj', values=cols_to_pivot)

# Réorganiser les colonnes pour regrouper par temps_inj (ART, VEIN, TARD, PORT)
# On suppose que l'ordre souhaité est : ART, VEIN, TARD, PORT
ordre_phases = ['VEIN', 'TARD', 'PORT', 'ART']
new_cols = []
for phase in ordre_phases:
    new_cols += [col for col in df_pivot.columns if col[1] == phase]

# Réindexer les colonnes dans le bon ordre
df_pivot = df_pivot[new_cols]

# Renommer les colonnes (phase d'abord, puis variable)
df_pivot.columns = [f"{phase}_{feature}" for feature, phase in df_pivot.columns]

# Réindexer les patients dans l'ordre d'origine
df_pivot = df_pivot.loc[ordre_patients].reset_index()

# Extraire à nouveau patient_num et classe_name
df_pivot[['patient_num', 'classe_name','slice_num']] = df_pivot['patient_id'].str.split('_', expand=True)
df_pivot.drop(columns=['patient_id'], inplace=True)

# Encoder les classes en 0, 1, 2
df_pivot['classe_name'], _ = pd.factorize(df_pivot['classe_name'])

# Réordonner les colonnes pour mettre patient_num et classe_name au début
cols = ['patient_num', 'classe_name','slice_num'] + [col for col in df_pivot.columns if col not in ['patient_num', 'classe_name','slice_num']]
df_pivot = df_pivot[cols]

# Enregistrer dans un fichier CSV
df_pivot.to_csv("nouveauu.csv", index=False, sep=";")
"""valeur = df_pivot.loc[0, 'ART_original_firstorder_10Percentile']
print(valeur)
"""
