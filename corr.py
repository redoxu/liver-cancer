import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("radiomiques_multislice.csv", sep=';')
categories = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
df_num = df.select_dtypes(include='number')

for cat in categories:
    #Sélection colonne dans une catégorie
    df_temp = df_num.filter(like=cat)
    corr_matrix = df_temp.corr()

    #Affichage
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, fmt=".2f")
    title = "Matrice de corrélation (Pearson):" + cat
    plt.title(title)
    plt.tight_layout()
    plt.show()
    

