import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement du fichier CSV
df = pd.read_csv('descriptif_patients.csv', sep=';')

# Nettoyage et conversions
df['Alpha_foetoprotein'] = pd.to_numeric(df['Alpha_foetoprotein'], errors='coerce')
df['Age_at_disease'] = pd.to_numeric(df['Age_at_disease'], errors='coerce')
df['Gender'] = df['Gender'].astype(str)


#  1. Histogramme des catégories d’âge par type de cancer

# Créer les groupes d'âge
age_bins = [0, 30, 45, 60, 75, 100]
age_labels = ['<30', '30-45', '45-60', '60-75', '>75']
df['Age_group'] = pd.cut(df['Age_at_disease'], bins=age_bins, labels=age_labels)

df_age = df.dropna(subset=['classe_name', 'Age_group'])

# Tracé
plt.figure(figsize=(10, 6))
sns.countplot(data=df_age, x='Age_group', hue='classe_name', palette='Set2')
plt.title("Répartition des classes d'âge par type de cancer")
plt.xlabel("Classe d'âge")
plt.ylabel("Nombre de patients")
plt.tight_layout()
plt.show()

# 2. Histogramme de la répartition par genre selon le type de cancer

df_gender = df.dropna(subset=['classe_name', 'Gender'])

plt.figure(figsize=(10, 6))
sns.countplot(data=df_gender, x='Gender', hue='classe_name', palette='Set2')
plt.title("Répartition par genre selon le type de cancer")
plt.xlabel("Genre")
plt.ylabel("Nombre de patients")
plt.tight_layout()
plt.show()
