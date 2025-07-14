import pandas as pd
import impute
import ACP



def long_to_wide(df):
    # Supprimer les colonnes commençant par "diagnostics"
    df = df.loc[:, ~df.columns.str.startswith("diagnostics")]
    df = pd.concat([df["classe_name"],df["temps_inj"],df.select_dtypes(include=["number"])], axis=1)

    # Ajouter une colonne pour l'ordre d'origine
    df['ordre_origine'] = range(len(df))

    # Créer un identifiant patient unique basé sur patient_num + classe_name
    df['patient_id'] = df['patient_num'].astype(str) + "_" + df['classe_name'].astype(str)

    # Liste des colonnes à pivoter (toutes sauf patient_num, classe_name, temps_inj)
    cols_to_pivot = [col for col in df.columns if col not in ['patient_num', 'classe_name', 'temps_inj', 'patient_id', 'ordre_origine']]

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
    df_pivot[['patient_num', 'classe_name']] = df_pivot['patient_id'].str.split('_', expand=True)
    df_pivot.drop(columns=['patient_id'], inplace=True)

    # Réordonner les colonnes pour mettre patient_num et classe_name au début
    cols = ['patient_num', 'classe_name'] + [col for col in df_pivot.columns if col not in ['patient_num', 'classe_name']]
    df_pivot = df_pivot[cols]

    # Enregistrer dans un fichier CSV
    df_pivot.to_csv("wide.csv", index=False,sep=";")
    return df_pivot



def wide_to_long(df, injection_types=['VEIN', 'TARD', 'PORT', 'ART']):
    id_vars = ['patient_num', 'classe_name']
    value_vars = [col for col in df.columns if any(col.startswith(inj + '_') for inj in injection_types)]

    melted = df.melt(id_vars=id_vars, value_vars=value_vars,
                     var_name='temps_inj_measure', value_name='value')

    # Use keyword arguments to ensure compatibility with all pandas versions
    melted[['temps_inj', 'measure']] = melted['temps_inj_measure'].str.split(pat='_', n=1, expand=True)

    reshaped = melted.pivot(index=['patient_num', 'classe_name', 'temps_inj'],
                            columns='measure', values='value').reset_index()

    reshaped.to_csv("long.csv", index=False, sep=";")
    return reshaped


"""def long_to_wide_multi(df):
    # Supprimer les colonnes commençant par "diagnostics"
    df = df.loc[:, ~df.columns.str.startswith("diagnostics")]
    df = pd.concat([df["classe_name"],df["temps_inj"],df.select_dtypes(include=["number"])], axis=1)

    # Ajouter une colonne pour l'ordre d'origine
    df['ordre_origine'] = range(len(df))

    # Créer un identifiant patient unique basé sur patient_num + classe_name
    df['patient_id'] = df['patient_num'].astype(str) + "_" + df['classe_name'].astype(str)+ "_" +df['slice_num'].astype(str)

    # Liste des colonnes à pivoter (toutes sauf patient_num, classe_name, temps_inj)
    cols_to_pivot = [col for col in df.columns if col not in ['patient_num', 'classe_name','slide_num', 'temps_inj', 'patient_id', 'ordre_origine']]

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
    df_pivot.to_csv("wide.csv", index=False, sep=";")
    return df_pivot

def wide_to_long_multi(df, injection_types=['VEIN', 'TARD', 'PORT', 'ART']):
    id_vars = ['patient_num', 'classe_name','slice_num']
    value_vars = [col for col in df.columns if any(col.startswith(inj + '_') for inj in injection_types)]

    melted = df.melt(id_vars=id_vars, value_vars=value_vars,
                     var_name='temps_inj_measure', value_name='value')

    melted[['temps_inj', 'measure']] = melted['temps_inj_measure'].str.split(pat='_', n=1, expand=True)

    reshaped = melted.pivot(index=['patient_num', 'classe_name','slice_num', 'temps_inj'],
                            columns='measure', values='value').reset_index()
    reshaped.to_csv("long.csv", index=False, sep=";")
    return reshaped
"""
def long_to_wide_multi(df):
    id_vars = ['slice_num', 'classe_name', 'patient_num']
    df = df.loc[:, ~df.columns.str.startswith("diagnostics")]
    df = pd.concat([df["classe_name"],df["temps_inj"],df.select_dtypes(include=["number"])], axis=1)


    # Liste des colonnes variables (toutes sauf id_vars + temps_inj)
    value_vars = [col for col in df.columns if col not in id_vars + ['temps_inj']]

    # On pivote : temps_inj devient le préfixe des colonnes
    df_wide = df.pivot(index=id_vars, columns='temps_inj', values=value_vars)

    # On a maintenant un MultiIndex sur les colonnes (valeur, temps_inj), il faut aplatir ça
    df_wide.columns = [f"{inj}_{var}" for var, inj in df_wide.columns]

    # On remet à plat l'index
    df_wide = df_wide.reset_index()
    df_wide = df_wide.sort_values(by=['patient_num', 'classe_name', 'slice_num']).reset_index(drop=True)
    df_wide.to_csv("wide.csv", index=False, sep=";")
    return df_wide


def wide_to_long_multi(df, injection_types=['VEIN', 'TARD', 'PORT', 'ART']):

    # Colonnes de base
    id_vars = ['slice_num', 'classe_name', 'patient_num']

    # Trouver toutes les colonnes contenant une injection
    value_vars = [col for col in df.columns if any(col.startswith(inj + '_') for inj in injection_types)]

    # Melt pour passer en format long
    df_long = df.melt(id_vars=id_vars, value_vars=value_vars,
                    var_name='temps_inj_var', value_name='value')

    # Séparer temps_inj et le nom de la variable
    df_long[['temps_inj', 'var_name']] = df_long['temps_inj_var'].str.split('_', n=1, expand=True)

    # Pivot pour avoir chaque variable comme colonne
    df_final = df_long.pivot(index=['slice_num', 'classe_name', 'patient_num', 'temps_inj'],
                            columns='var_name', values='value').reset_index()

    df_final.to_csv("long.csv", index=False, sep=";")

    return df_final


df = pd.read_csv("radiomiques_multislice.csv", sep=";")

print("Initial DataFrame shape:", df.shape)

wide = long_to_wide_multi(df)

print("Wide DataFrame created with shape:", wide.shape)
wide_imp = impute.main(wide)

print("Imputation completed. DataFrame shape after imputation:", wide_imp.shape)
reshaped = wide_to_long_multi(wide_imp)

print("Reshaped DataFrame created with shape:", reshaped.shape)

"""
ACP.main(reshaped)
"""