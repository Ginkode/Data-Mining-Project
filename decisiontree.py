# Progetto Data Mining - Analisi OASIS Dataset
# Nome e Cognome: Victor Hugo Cascone
# Data: 12/11/2025
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

# Carica il dataset
data = pd.read_csv('C:/Users/victo/OneDrive/Documenti/dataminingproj/oasis_longitudinal.csv')

# Rimuove righe con valori mancanti nelle colonne critiche
data = data.dropna(subset=['MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF', 'SES', 'Group'])

# ---------------------------
# Preparazione dei dati
# ---------------------------

# Usa anche i “Converted” come classe = 1 (Demented)
data_model = data.copy()
data_model = data_model[data_model['Group'].isin(['Nondemented','Demented','Converted'])].copy()
data_model['Target'] = data_model['Group'].replace({'Nondemented': 0, 'Demented': 1, 'Converted': 1}).astype(int)

features = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
X = data_model[features]
y = data_model['Target']

# ---------------------------
# Costruzione del modello
# ---------------------------
clf = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # 👉 ATTIVATO
    random_state=42
)

# Non separo training/test, uso Leave‑One‑Out per tutti gli esempi
loo = LeaveOneOut()
y_pred = cross_val_predict(clf, X, y, cv=loo)

# ---------------------------
# Valutazione: classification report e confusion matrix
# ---------------------------
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=['Nondemented','Demented_or_Converted']))

cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(6,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix — LOO (Converted inclusi)')
plt.colorbar()
tick_marks = [0,1]
plt.xticks(tick_marks, ['Nondemented','Demented_or_Converted'], rotation=45)
plt.yticks(tick_marks, ['Nondemented','Demented_or_Converted'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2. else "black")
plt.tight_layout()
plt.show()

clf.fit(X, y)  # addestra su tutto il dataset per visualizzare l'albero completo

plt.figure(figsize=(20,10))
tree.plot_tree(clf,
               feature_names=features,
               class_names=['Nondemented', 'Demented'],
               filled=True,
               rounded=True,
               fontsize=10)
plt.title("Decision Tree - Visualizzazione Finale")
plt.tight_layout()
plt.show()