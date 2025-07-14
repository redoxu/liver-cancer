import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def main(df, seuil_corr=0.9):
    df_cleaned = df.dropna()

    df_numeric = df_cleaned.select_dtypes(include='number')

    df_filtered = df_numeric.loc[:, df_cleaned.nunique() > 1]
    corr_matrix = df_filtered.corr().abs()
    corr_matrix = corr_matrix.fillna(0).clip(0, 1)

    G = nx.Graph()

    G.add_nodes_from(corr_matrix.columns)

    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i != j and corr_matrix.loc[i, j] >= seuil_corr:
                G.add_edge(i, j)

    groupes = list(nx.connected_components(G))
    if {'patient_num'} in groupes:
        groupes.remove({'patient_num'})
    i = len(groupes)

    df_grouped = pd.DataFrame(index=df_cleaned.index)

    group_cols = {}
    for i, group in enumerate(groupes):
        cols_in_group = [col for col in group if col in df_cleaned.columns]
        if len(cols_in_group) == 0:
            continue
        group_cols[f'group_{i+1}'] = df_cleaned[cols_in_group].mean(axis=1)

    df_grouped = pd.concat(group_cols.values(), axis=1)
    df_grouped.columns = group_cols.keys()
    df_grouped.index = df_cleaned.index  # pour garder le même index

    df_grouped['cancer_type'] = df_cleaned.iloc[:, 1]
    df_grouped = df_grouped[df_grouped["cancer_type"] != 'Mixtes']

    df_grouped["classe_name"] = df.iloc[:, 1]
    df_grouped["patient_num"] = df.iloc[:, 0]


    return df_grouped
