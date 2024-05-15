!pip install scikit-plot

# Google Drive ile bağlantı kurma
from google.colab import drive
drive.mount('/content/drive')

# Gerekli kütüphaneleri yükleme
import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("********************************************************************************************")

# Veri setini okuma
data_path = "/content/drive/MyDrive/yapayZekaProjeVeri/mobile_price_dataset.csv"  # Veri setinin yolu
data = pd.read_csv(data_path)

# Veri setindeki 21 özelliğin isimlerini yazdır
print("Features:", data.columns[:-1])

# Veri etiket türlerini yazdır
print("Labels:", data.columns[-1])

# Veri özellik şeklini yazdır
print("Data Shape:", data.shape)

# Veri setini gözlemleme
print(data.head())  # Veri setinin ilk birkaç (5) gözlemi

# Ön işleme adımları
# Eksik değerleri kontrol etme
print(data.isnull().sum())  # Eksik değerleri sayma

# Bağımlı ve bağımsız değişkenleri belirleme
X = data.drop("price_range", axis=1)  # Bağımsız değişkenler
y = data["price_range"]  # Bağımlı değişken

# Verileri ölçeklendirme veya normalize etme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veri setini train ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Sınıflandırma algoritmalarını tanımlama
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(probability=True),  # ROC eğrisi için probability=True
    "Naive Bayes": GaussianNB(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "AdaBoost Classifier": AdaBoostClassifier(),
    "MLP Classifier": MLPClassifier(max_iter=300)
}

# Her bir modeli eğitme ve değerlendirme
for name, model in models.items():
    model.fit(X_train, y_train)  # Modeli eğitme
    accuracy = model.score(X_test, y_test)  # Doğruluk değerini hesaplama
    print(f"{name}: Doğruluk = {accuracy}")

# Her bir model için performans metriklerini hesaplama ve yazdırma
for name, model in models.items():
    y_pred = model.predict(X_test)  # Tahmin yapma

    # Performans metriklerini hesaplama
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Performans metriklerini yazdırma
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print()

# En iyi performans gösteren modeli seçme
best_model_name = "Random Forest"  # En iyi modelin adı
best_model = models[best_model_name]  # En iyi modeli seçme

# Yeni veri seti üzerinde tahmin yapma (Test veri seti üzerinden aynı veri setinde)
y_new_pred = best_model.predict(X_test)  # Tahmin yapma

# Tahminleri değerlendirme
# Örneğin, tahminler üzerinde bir performans metriği hesaplayabilirsiniz veya tahminlerinizi görselleştirebilirsiniz.

# Confusion matrix çizimi
skplt.metrics.plot_confusion_matrix(y_test, y_new_pred, normalize=True)
plt.title(f"{best_model_name} Confusion Matrix")
plt.show()

# ROC eğrisi çizimi
if not isinstance(best_model, SVC):
    y_prob = best_model.predict_proba(X_test)  # Sınıflandırma olasılıklarını al
    skplt.metrics.plot_roc(y_test, y_prob, title=f"{best_model_name} ROC Curve", plot_micro=False, plot_macro=False)
    plt.show()

# Feature Importance Grafiği çizimi
if isinstance(best_model, RandomForestClassifier):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()
else:
    print("Feature importance grafiği sadece Random Forest modelleri için desteklenmektedir.")
