import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main(df):
    X = df.drop(columns=['temps_inj','classe_name','patient_num'], errors='ignore')
    X_imputed = SimpleImputer(strategy='mean').fit_transform(X)

    X_scaled = StandardScaler().fit_transform(X_imputed)
    pca = PCA(n_components=15)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_

    # Créer un DataFrame avec les composantes principales
    df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])

    # Ajouter la classe
    df_pca["classe_name"] = df["classe_name"].values
    df_pca = df_pca[df_pca["classe_name"] != "Mixtes"]
    # Sauvegarder dans un fichier CSV
    df_pca.to_csv("resultat_pca.csv", index=False)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(explained_var) + 1), explained_var * 100)
    plt.xlabel("Composantes principales")
    plt.ylabel("Variance expliquée (%)")
    plt.title("Variance expliquée par composante")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
