import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Charger le fichier CSV
df = pd.read_csv("nouveau.csv", sep=';')
df = df.dropna()

# Supprimer les lignes où la classe est 0
df = df[df["classe_name"] != 0]

# Séparer variables explicatives et cible
X = df.drop(columns=["classe_name","patient_num"])  
y = df["classe_name"]

# Séparation train/test avec stratification pour garder la proportion des classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardiser les variables sur le train puis appliquer au test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Régression logistique avec pénalisation L1 
model = LogisticRegressionCV(
    Cs=np.logspace(-1, 3, 20),
    cv=5,
    penalty='l1',
    solver='saga',
    random_state=0,
    max_iter=20000,
    scoring='accuracy',
    multi_class='multinomial'
)
model.fit(X_train_scaled, y_train)

# Prédiction et calcul de l'accuracy sur le test
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy sur le jeu de test : {accuracy:.3f}")

# Moyenne absolue des coefficients pour chaque variable de X_train (multiclasse = moyenne sur classes)
coef = pd.Series(np.mean(np.abs(model.coef_), axis=0), index=X_train.columns)

# Afficher les variables sélectionnées
selected_features = coef[coef > 0]
print("Variables sélectionnées :\n", selected_features)
print(f"Nombre de variables sélectionnées : {selected_features.shape[0]}")

# Visualisation
if not selected_features.empty:
    selected_features.sort_values().plot(kind='barh', figsize=(8, 5))
    plt.title("Variables sélectionnées par régression logistique L1")
    plt.xlabel("Importance (|coefficient| moyen)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Aucune variable sélectionnée.")

