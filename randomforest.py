from sklearn.ensemble import RandomForestClassifier  # ou RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def main_multi():

    # 1. Lecture du fichier
    df = pd.read_csv("radiomiques_multislice.csv", sep=";")

    # 2. Encodage des catégories
    df = df.dropna(subset=['classe_name', 'temps_inj', 'patient_num'])  # Enlève les NaN critiques
    df['classe_encoded'] = LabelEncoder().fit_transform(df['classe_name'])
    df['temps_inj_encoded'] = LabelEncoder().fit_transform(df['temps_inj'])

    # 3. On garde toutes les colonnes numériques sauf les identifiants
    exclure = ['slice_num', 'classe_name', 'temps_inj', 'patient_num']
    features = df.drop(columns=exclure).select_dtypes(include='number').columns.tolist()

    # 4. Agrégation des slices par patient + phase d'injection
    agg_df = df.groupby(['patient_num', 'temps_inj_encoded', 'classe_encoded'], as_index=False)[features].mean()

    # 5. Features (X) et target (y)
    X = agg_df.drop(columns=['patient_num', 'classe_encoded'])  # On garde `temps_inj_encoded` comme feature
    y = agg_df['classe_encoded']

    # 6. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

    # 7. Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 8. Résultats
    print(classification_report(y_test, clf.predict(X_test)))

    label_encoder = LabelEncoder()
    df['classe_encoded'] = label_encoder.fit_transform(df['classe_name'])

    print("Correspondance classe_encoded → classe_name :")
    for code, label in enumerate(label_encoder.classes_):
        print(f"{code} → {label}")


def main_global(df):
    df = df.dropna(subset=['classe_name', 'temps_inj', 'patient_num'])  # Enlève les NaN critiques
    df['classe_encoded'] = LabelEncoder().fit_transform(df['classe_name'])
    df['temps_inj_encoded'] = LabelEncoder().fit_transform(df['temps_inj'])

    # 3. On garde toutes les colonnes numériques sauf les identifiants
    exclure = ['classe_name', 'temps_inj', 'patient_num']
    features = df.drop(columns=exclure).select_dtypes(include='number').columns.tolist()

    # 4. Agrégation des slices par patient + phase d'injection
    agg_df = df.groupby(['patient_num', 'temps_inj_encoded', 'classe_encoded'], as_index=False)[features].mean()

    # 5. Features (X) et target (y)
    X = agg_df.drop(columns=['patient_num', 'classe_encoded'])  # On garde `temps_inj_encoded` comme feature
    y = agg_df['classe_encoded']

    # 6. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

    # 7. Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 8. Résultats
    print(classification_report(y_test, clf.predict(X_test)))

    label_encoder = LabelEncoder()
    df['classe_encoded'] = label_encoder.fit_transform(df['classe_name'])

    print("Correspondance classe_encoded → classe_name :")
    for code, label in enumerate(label_encoder.classes_):
        print(f"{code} → {label}")

def main_sans_temps_inj(df):
    df = df.dropna(subset=['classe_name', 'patient_num'])  # Enlève les NaN critiques
    label_encoder = LabelEncoder()
    df['classe_encoded'] = label_encoder.fit_transform(df['classe_name'])
    # 3. On garde toutes les colonnes numériques sauf les identifiants
    exclure = ['classe_name', 'patient_num']
    features = df.drop(columns=exclure).select_dtypes(include='number').columns.tolist()

    # 4. Agrégation des slices par patient + phase d'injection
    agg_df = df.groupby(['patient_num', 'classe_encoded'], as_index=False)[features].mean()

    # 5. Features (X) et target (y)
    X = agg_df.drop(columns=['patient_num', 'classe_encoded'])  # On garde `temps_inj_encoded` comme feature
    y = agg_df['classe_encoded']

    # 6. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

    # 7. Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 8. Résultats
    print(classification_report(y_test, clf.predict(X_test)))

    print("Correspondance classe_encoded → classe_name :")
    for code, label in enumerate(label_encoder.classes_):
        print(f"{code} → {label}")