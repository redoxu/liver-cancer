import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Charger le fichier CSV
df = pd.read_csv("nouveau.csv", sep=';')
df.iloc[:, 2:] = df.iloc[:, 2:].fillna(df.iloc[:, 2:].mean())
# Séparer variables explicatives et cible
X = df.drop(columns=["classe_name"])  
y = df["classe_name"]

# Standardiser les variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer Lasso avec validation croisée
lasso = LassoCV(cv=5, random_state=0)
lasso.fit(X_scaled, y)

# Récupérer les coefficients
coef = pd.Series(lasso.coef_, index=X.columns)

# Afficher les variables sélectionnées
selected_features = coef[coef != 0]
print("Variables sélectionnées :\n", selected_features)

# Visualiser
selected_features.sort_values().plot(kind='barh', figsize=(8, 5))
plt.title("Variables sélectionnées par Lasso")
plt.xlabel("Coefficient")
plt.grid(True)
plt.tight_layout()
plt.show()
