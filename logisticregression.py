# Progetto Data Mining - Analisi OASIS Dataset
# Nome e Cognome: Victor Hugo Cascone
# Data: 12/11/2025
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# 1) CARICAMENTO DATASET
data = pd.read_csv("C:/Users/victo/OneDrive/Documenti/dataminingproj/oasis_longitudinal.csv")


# 2) PULIZIA E PREPARAZIONE DATI
data = data.dropna(subset=['MMSE','CDR', 'eTIV', 'nWBV', 'ASF', 'SES', 'Group'])

# Includo anche Converted nel training come classe 1
data_model = data[data['Group'].isin(['Nondemented', 'Demented', 'Converted'])].copy()
data_model['Group'] = data_model['Group'].replace({'Nondemented': 0, 'Demented': 1, 'Converted': 1}).astype(int)

# 3) DEFINIZIONE FEATURE E TARGET
features = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
X = data_model[features]
y = data_model['Group']

# 4) REGRESSIONE LOGISTICA + LOO
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
loo = LeaveOneOut()

y_pred = cross_val_predict(log_reg, X, y, cv=loo)

# 5) VALUTAZIONE DEL MODELLO
print("\n=== CLASSIFICATION REPORT (LOO) ===")
print(classification_report(y, y_pred, target_names=['Nondemented', 'Demented/Converted']))

cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,4))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix — Logistic Regression (LOO)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0,1], ['Nondemented', 'Demented'])
plt.yticks([0,1], ['Nondemented', 'Demented'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.show()

log_reg.fit(X, y)

# Calcola le probabilità di classe 1 (Demented) su X
prob_demented = log_reg.predict_proba(X)[:, 1]

# Istogramma della distribuzione
plt.figure(figsize=(7,5))
plt.hist(prob_demented, bins=10, edgecolor='black')
plt.xlabel("Probabilità modello: Demented")
plt.ylabel("Numero di soggetti")
plt.title("Distribuzione delle probabilità - Logistic Regression (solo training set)")
plt.tight_layout()

plt.show()

