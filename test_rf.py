import var_matrice as varmat
import randomforest
import pandas as pd

df = pd.read_csv('global_par_patient_imp.csv', sep=";")

print("Initialement:")

print(randomforest.main_sans_temps_inj(df))

val_seuil = [0.95, 0.9, 0.85, 0.8]

print("Avec le nouveau dataset:")
for seuil in val_seuil:
    print(f"\nAvec seuil de corrélation {seuil}:")
    df_grouped = varmat.main(df, seuil_corr=seuil)
    print(randomforest.main_sans_temps_inj(df_grouped))

#! NOTES SUR LE CODE
""" 
En ce moment:
- Dans le initialement, on utilise les 3 types de cancers alors qu'il faut drop le mixte
- Ensuite, on drop les mixtes
- On utilise global_par_patient.csv parce que c'est le dataset qui va donner une ligne <-> un couple (cancer, patient)
"""

#TODO: mettre remplacer les valeurs manquantes par la moyenne, ou moyenne + (random.uniform(0, 1) * std)
#TODO: ajouter des gens dans la classe minoritaire
#TODO: comparer les méthodes avec l'AUC (?)
