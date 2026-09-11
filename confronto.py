# Progetto Data Mining - Analisi OASIS Dataset
# Nome e Cognome: Victor Hugo Cascone
# Data: 12/11/2025
# --- 1. Import librerie ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 2. Caricamento e pulizia dati ---
data = pd.read_csv("C:/Users/victo/OneDrive/Documenti/dataminingproj/oasis_longitudinal.csv")
data = data.dropna(subset=['MMSE','CDR','eTIV','nWBV','ASF','SES','Group'])
data = data[data['Group'].isin(['Nondemented','Demented','Converted'])].copy()
data['Target'] = data['Group'].replace({'Nondemented':0,'Demented':1,'Converted':1}).astype(int)
features = ['Age','EDUC','SES','MMSE','eTIV','nWBV','ASF']
X = data[features]
y = data['Target']

loo = LeaveOneOut()

# --- 3. Decision Tree ---
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', random_state=42)
y_pred_tree = cross_val_predict(clf, X, y, cv=loo)
acc_tree = accuracy_score(y, y_pred_tree)

# --- 4. Logistic Regression ---
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
y_pred_log = cross_val_predict(log_reg, X, y, cv=loo)
acc_log = accuracy_score(y, y_pred_log)

# --- 5. SVM ---
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', random_state=42))
y_pred_svm = cross_val_predict(svm_model, X, y, cv=loo)
acc_svm = accuracy_score(y, y_pred_svm)

# --- 6. Tabella comparativa ---
results = pd.DataFrame({
    'Modello': ['Decision Tree', 'Logistic Regression', 'SVM (RBF)'],
    'Accuracy': [acc_tree, acc_log, acc_svm],
    'F1 Demented': [
        classification_report(y, y_pred_tree, output_dict=True)['1']['f1-score'],
        classification_report(y, y_pred_log, output_dict=True)['1']['f1-score'],
        classification_report(y, y_pred_svm, output_dict=True)['1']['f1-score']
    ]
})

print("\n=== RISULTATI COMPARATIVI (Leave-One-Out) ===")
print(results.round(3))

#heatmap accuracies
sns.barplot(data=results, x='Modello', y='Accuracy', palette='viridis')
plt.title("Confronto Accuracy - Modelli OASIS")
plt.tight_layout()
plt.show()

