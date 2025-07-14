import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import networkx as nx

def main(df, seuil_corr=0.9):
    df_numeric = df.select_dtypes(include='number')

    df_filtered = df_numeric.loc[:, df.nunique() > 1]

    corr_matrix = df_filtered.corr().abs()
    corr_matrix = corr_matrix.fillna(0).clip(0, 1)

    G = nx.Graph()

    G.add_nodes_from(corr_matrix.columns)

    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i != j and corr_matrix.loc[i, j] >= seuil_corr:
                G.add_edge(i, j)

    groupes = list(nx.connected_components(G))

    df_grouped = pd.DataFrame(index=df.index)

    for i, group in enumerate(groupes):
        cols_in_group = [col for col in group if col in df.columns]
        
        if len(cols_in_group) == 0:
            continue
        df_grouped[f'group_{i+1}'] = df[cols_in_group].mean(axis=1)

    df_grouped["classe_name"] = df.iloc[:, 0]
    df_grouped["temps_inj"] = df.iloc[:, 1]
    df_grouped["patient_num"] = df.iloc[:, 2]

    return df_grouped

