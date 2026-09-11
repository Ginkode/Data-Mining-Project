# Progetto Data Mining - Analisi OASIS Dataset
# Nome e Cognome: Victor Hugo Cascone
# Data: 12/11/2025
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1) CARICAMENTO DATASET
data = pd.read_csv("C:/Users/victo/OneDrive/Documenti/dataminingproj/oasis_longitudinal.csv")

# 2) PULIZIA E PREPARAZIONE
data = data.dropna(subset=['MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF', 'SES', 'Group'])

# Includo anche i Converted come Demented = 1
data_model = data[data['Group'].isin(['Nondemented', 'Demented', 'Converted'])].copy()
data_model['Group'] = data_model['Group'].replace({'Nondemented': 0, 'Demented': 1, 'Converted': 1}).astype(int)

# Definizione delle feature
features = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
X = data_model[features]
y = data_model['Group']

# 3) SVM CON SCALING E CLASS WEIGHT
# SVM con kernel RBF (non lineare) + bilanciamento classi
svm_model = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', random_state=42)
)

# 4) LEAVE-ONE-OUT CROSS VALIDATION
loo = LeaveOneOut()
y_pred = cross_val_predict(svm_model, X, y, cv=loo)

# 5) RISULTATI GLOBALI
print("\n=== CLASSIFICATION REPORT (SVM RBF, LOOCV) ===")
print(classification_report(y, y_pred, target_names=['Nondemented', 'Demented/Converted']))

cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Nondemented','Demented'],
            yticklabels=['Nondemented','Demented'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix — SVM RBF (Converted inclusi, LOO)")
plt.tight_layout()
plt.show()

accuracy = accuracy_score(y, y_pred)
print(f"\nAccuracy complessiva (LOOCV): {round(accuracy, 4)}")

svm_linear = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='linear', C=1, class_weight='balanced', random_state=42))
])

y_pred_linear = cross_val_predict(svm_linear, X, y, cv=loo)

print("\n=== CLASSIFICATION REPORT (SVM LINEARE, LOOCV) ===")
print(classification_report(y, y_pred_linear, target_names=['Nondemented', 'Demented/Converted']))

cm_linear = confusion_matrix(y, y_pred_linear)
plt.figure(figsize=(6,4))
sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Nondemented','Demented'],
            yticklabels=['Nondemented','Demented'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix — SVM Lineare (Converted inclusi, LOO)")
plt.tight_layout()
plt.show()

accuracy_linear = accuracy_score(y, y_pred_linear)
print(f"\nAccuracy complessiva (SVM Lineare, LOO): {round(accuracy_linear, 4)}")


C_values = [0.01, 0.1, 1, 10, 100, 500]
train_acc = []
test_acc = []
# Faccio uno split semplice per valutare C (non LOOCV)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

for C in C_values:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='linear', C=C, class_weight='balanced', random_state=42))
    ])
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc.append(accuracy_score(y_train, y_pred_train))
    test_acc.append(accuracy_score(y_test, y_pred_test))

# Grafico
plt.figure(figsize=(8,5))
plt.plot(C_values, train_acc, 'ro-', label='Training Accuracy')
plt.plot(C_values, test_acc, 'bv--', label='Test Accuracy')
plt.xscale('log')
plt.xlabel('C (scala log)')
plt.ylabel('Accuracy')
plt.title('SVM Lineare: Effetto del parametro C')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()