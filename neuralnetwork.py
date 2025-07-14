import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from logistical import selected_features,X_train, X_test, y_train, y_test, scaler
# Construction du modèle
# Étape 1 : Extraire uniquement les colonnes sélectionnées
X_train_sel = X_train[selected_features.index]
X_test_sel = X_test[selected_features.index]

# Étape 2 : Re-standardiser avec les bonnes colonnes
X_train_sel_scaled = scaler.fit_transform(X_train_sel)
X_test_sel_scaled = scaler.transform(X_test_sel)

# Étape 3 : Créer le réseau de neurones
model = Sequential()
model.add(Dense(256, input_dim=X_train_sel_scaled.shape[1]))
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(LeakyReLU())
model.add(Dropout(0.4))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Étape 4 : Entraîner le modèle
history = model.fit(
    X_train_sel_scaled, y_train,
    batch_size=200,
    epochs=100,
    validation_split=0.2,
    verbose=1
)

# Étape 5 : Évaluer sur le jeu de test
test_loss, test_acc = model.evaluate(X_test_sel_scaled, y_test, verbose=0)
print(f"Test Accuracy (réseau de neurones) : {test_acc:.3f}")
