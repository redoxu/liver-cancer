# Détection du type de cancer du foie et prédiction de la survie

Projet réalisé dans le cadre du ST4 Big Data et Santé en partenariat avec le service de radiologie de l’hôpital Henri Mondor.

## 👥 Équipe projet
- Reda Hamama  
- Marilou Bernard de Courville  
- Amram El Bazis  
- Alexis Le Parco  
- Adiel Van Hecke  

## 🗓 Date de rendu : 6 juin 2025

---

## 🧠 Objectif du projet

Développer un outil d’aide à la décision basé sur des données radiomiques, cliniques et visuelles, pour :
- Classer le type de cancer du foie (CHC, CCK, Mixte)
- Prédire la survie des patients à partir de métriques radiomiques multi-phases

---

## 📁 Données utilisées

- **Patients** : 148 au total (majorité hommes, principalement CHC)
- **Types de données** :
  - Données radiomiques globales
  - Données radiomiques multislice
  - Observations visuelles des radiologues
- **Phases d'injection** : ART (artérielle), PORT (portale), VEIN (veineuse), TARD (tardive)

---

## ⚙️ Pipeline de traitement

### 1. **Prétraitement des données**
- Nettoyage : suppression des colonnes non exploitables
- Extraction de variables statistiques par coupe (moyenne, écart-type, skewness, énergie)
- Standardisation des variables

### 2. **Multislicing**
- Analyse slice par slice des tumeurs
- Création de courbes pour suivre l’évolution des paramètres selon les slices

### 3. **Réduction de dimension**
- Utilisation de l'ACP (Analyse en Composantes Principales) pour sélectionner les variables les plus discriminantes

---

## 📊 Modèles de classification

### 🔍 Méthodes utilisées
- **Régression logistique**
- **Random Forest**
- **LASSO** pour la sélection interprétable des variables

### 📈 Résultats AUC
| Dataset                                   | AUC      |
|-------------------------------------------|----------|
| Multislice                                | 0.61     |
| Fusion multislice + global                | 0.55     |
| Global seul                               | 0.86     |
| Global + visuel (radiologues)             | 0.99 ✅   |

Le modèle le plus performant combine les données globales avec les observations visuelles.

---

## ⏱ Prédiction de la survie

- Étude comparative entre les patients vivants et décédés (suivis ≥ 1 an)
- Métriques utilisées :
  - Moyenne, variance, skewness
  - Nombre de slices
  - Indicateurs S+ / S− (variation directionnelle des valeurs)

### 📌 Variables discriminantes par phase :

| Phase | Métriques clés      | Variables discriminantes |
|-------|---------------------|---------------------------|
| ART   | S+, variance        | `glcm_Idn`, `SumEntropy` |
| PORT  | variance            | `glszm_SmallAreaEmphasis` |
| TARD  | skewness            | `firstorder_Energy`       |
| VEIN  | skewness, S+        | `firstorder_Median`       |

---

## ✅ Conclusion

- Les modèles basés sur les données globales et visuelles sont très performants pour la classification des types de cancer.
- L’analyse de survie montre que certaines variables radiomiques peuvent distinguer efficacement les patients selon leur statut vital.
- Perspectives : élargir la base de données, intégrer une modélisation de la durée de survie (survie continue).



