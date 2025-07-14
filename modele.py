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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("C:/Users/alexi/Downloads/EI_ST4/EI-ST4/radiomiques_global.csv", sep=';')
data = pd.read_csv("C:/Users/alexi/Downloads/EI_ST4/EI-ST4/resultat_pca.csv", sep=',')
data["patient_num"] = df["patient_num"]


def modele_regression_logistique(data):

    print(data['classe_name'])
    y = data['classe_name']
    groups = data['patient_num']
    X = data.drop(columns=['classe_name', 'patient_num'], errors='ignore')
    
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
   
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced', max_iter=1000)
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title("Matrice de confusion - Régression logistique")
    plt.show()

def modele_AUC(data) :
    y = data['classe_name']
    groups = data['patient_num']
    X = data.drop(columns=['classe_name', 'patient_num'], errors='ignore')
    
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    smote = SMOTE(random_state=42)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test) 
    y_encoded = le.transform(y)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train_enc)
    model = RandomForestClassifier(random_state=42,class_weight='balanced')
    model.fit(X_resampled, y_resampled)

    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test_enc, y_proba)
    print(f"AUC ROC : {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test_enc, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Faux Positifs (FPR)")
    plt.ylabel("Vrais Positifs (TPR)")
    plt.title("Courbe ROC")
    plt.legend()
    plt.grid(True)
    plt.show()   
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', RandomForestClassifier(class_weight="balanced", random_state=42))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y_encoded, cv=skf, scoring='roc_auc')
    print("AUC CV avec SMOTE dans pipeline :", scores)
    print("Moyenne AUC :", scores.mean())
 
def modele_randomforest(file_path) :
    print(data['classe_name'])
    y = data['classe_name']
    groups = data['patient_num']
    X = data.drop(columns=['classe_name', 'patient_num'], errors='ignore')
    
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)   
    clf = RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced')
    clf.fit(X_resampled, y_resampled)
    print(classification_report(y_test, clf.predict(X_test)))


print("== Régression logistique ==")
print(modele_regression_logistique(data))


print("== AUC Random Forest ==")
print(modele_AUC(data))

print("== Random Forest Classique ==")
print(modele_randomforest(data))
