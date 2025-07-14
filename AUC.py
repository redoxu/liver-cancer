import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

df = pd.read_csv("EI-ST4/global_par_patient_imp.csv", sep=';')
df_cleaned = df.dropna()
df_numeric = df_cleaned.select_dtypes(include='number')
df_filtered = df_numeric.loc[:, df_numeric.nunique() > 1]
df_cleaned = df_cleaned[df_cleaned.iloc[:, 0] != 'Mixte']  # filtre bien le 'Mixte'

corr_matrix = df_filtered.corr().abs().fillna(0).clip(0, 1)
seuil_corr = 0.75

G = nx.Graph()
G.add_nodes_from(corr_matrix.columns)

for i in corr_matrix.columns:
    for j in corr_matrix.columns:
        if i != j and corr_matrix.loc[i, j] >= seuil_corr:
            G.add_edge(i, j)

groupes = list(nx.connected_components(G))


groupes.remove({'patient_num'})

print(len(groupes))

df_grouped = pd.DataFrame(index=df_cleaned.index)
for i, group in enumerate(groupes):
    cols_in_group = [col for col in group if col in df_cleaned.columns]
    if len(cols_in_group) == 0:
        continue
    df_grouped[f'group_{i+1}'] = df_cleaned[cols_in_group].mean(axis=1)

df_grouped['cancer_type'] = df_cleaned.iloc[:, 1]
df_grouped = df_grouped[df_grouped["cancer_type"] != 'Mixtes']  # attention à l'orthographe


X = df_grouped.drop("cancer_type", axis=1)
y = df_grouped["cancer_type"]


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

y_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
print(f"AUC ROC : {auc:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Faux Positifs (FPR)")
plt.ylabel("Vrais Positifs (TPR)")
plt.title("Courbe ROC")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(class_weight="balanced", random_state=42))
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_scaled, y_encoded, cv=skf, scoring='roc_auc')
print("AUC CV avec SMOTE dans pipeline :", scores)
print("Moyenne AUC :", scores.mean())

print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")