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
from imblearn.over_sampling import SMOTE


df = pd.read_csv("EI-ST4/global_par_patient_imp.csv", sep=';')
df_cleaned = df.dropna()

df_numeric = df_cleaned.select_dtypes(include='number')

print(df_numeric.sum())

df_filtered = df_numeric.loc[:, df_cleaned.nunique() > 1]
df_filtré = df_cleaned[df_cleaned.iloc[:, 0] != 'Mixte']
corr_matrix = df_filtered.corr().abs()
corr_matrix = corr_matrix.fillna(0).clip(0, 1)

seuil_corr = 0.75

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
print(i)
df_grouped = pd.DataFrame(index=df_cleaned.index)

for i, group in enumerate(groupes):
    cols_in_group = [col for col in group if col in df_cleaned.columns]
    
    if len(cols_in_group) == 0:
        continue
    df_grouped[f'group_{i+1}'] = df_cleaned[cols_in_group].mean(axis=1)

df_grouped['cancer_type'] = df_cleaned["classe_name"]
df_grouped = df_grouped[df_grouped["cancer_type"] != 'Mixtes']


X = df_grouped[[f'group_{i}' for i in range(1, i+1)]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = df_grouped['cancer_type'] 
print(X,y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced')
model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))



"""dist_matrix = 1 - corr_matrix
dist_array = squareform(dist_matrix.values)

linkage_matrix = linkage(dist_array, method='average')
cluster_ids = fcluster(linkage_matrix, t=0.15, criterion='distance')
groupes = pd.DataFrame({'variable': corr_matrix.columns, 'groupe': cluster_ids})"""






#plt.figure(figsize=(15, 12))
#sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt='.2f')
#plt.title("Matrice de corrélation")
#plt.show()


"""mask = np.triu(corr_matrix, k=1)  
corr_pairs = corr_matrix.where(mask != 0).stack().sort_values(ascending=False)

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]


df_reduit = df_filtered.drop(columns=to_drop)
corr_matrix_red = df_reduit.corr().abs()

print(corr_matrix_red)"""

