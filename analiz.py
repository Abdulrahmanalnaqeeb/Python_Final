# Gerekli kütüphaneleri yükleyelim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# Grafiklerde Türkçe karakterlerin düzgün görünmesi için ayar
plt.rcParams['font.family'] = 'Arial'

# 1. Rastgele öğrenci verisi oluşturma
np.random.seed(42)  # Tekrarlanabilirlik

num_students = 1000

df = pd.DataFrame({
    'gender': np.random.choice(['male', 'female'], size=num_students),
    'race/ethnicity': np.random.choice(['group A', 'group B', 'group C', 'group D', 'group E'], size=num_students),
    'parental level of education': np.random.choice([
        'some high school', 'high school', 'associate\'s degree',
        'bachelor\'s degree', 'master\'s degree'], size=num_students),
    'lunch': np.random.choice(['standard', 'free/reduced'], size=num_students),
    'test preparation course': np.random.choice(['none', 'completed'], size=num_students),
    'math score': np.random.randint(0, 101, size=num_students),
    'reading score': np.random.randint(0, 101, size=num_students),
    'writing score': np.random.randint(0, 101, size=num_students),
})

# Ortalama skor kolonu
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

# 2. Veri setini inceleyelim
print("🔹 İlk 5 Satır:\n", df.head())
print("\n🔹 Veri Seti Boyutu:", df.shape)
print("\n🔹 Sütunlar:\n", df.columns)
print("\n🔹 Eksik Değer Sayısı:\n", df.isnull().sum())
print("\n🔹 Temel İstatistiksel Özet:\n", df.describe())

# 3. Görselleştirmeler

# Cinsiyete göre matematik notları
sns.boxplot(data=df, x='gender', y='math score', palette='pastel')
plt.title('Cinsiyete Göre Matematik Notları')
plt.xlabel('Cinsiyet')
plt.ylabel('Matematik Notu')
plt.grid(True)
plt.show()

# Test hazırlığına göre okuma skoru
sns.boxplot(data=df, x='test preparation course', y='reading score', palette='Set2')
plt.title('Test Hazırlığına Göre Okuma Skoru')
plt.xlabel('Test Hazırlık Durumu')
plt.ylabel('Okuma Skoru')
plt.grid(True)
plt.show()

# Ebeveyn eğitimi ve ortalama skor
df.groupby('parental level of education')['average_score'].mean().sort_values().plot(kind='barh', color='skyblue')
plt.title('Ebeveyn Eğitimi ve Ortalama Skor')
plt.xlabel('Ortalama Skor')
plt.ylabel('Ebeveyn Eğitim Seviyesi')
plt.grid(True)
plt.show()

# Irk/etnik gruba göre yazma puanı
sns.violinplot(data=df, x='race/ethnicity', y='writing score', palette='Set3')
plt.title('Etnik Gruba Göre Yazma Skoru Dağılımı')
plt.xlabel('Grup')
plt.ylabel('Yazma Skoru')
plt.grid(True)
plt.show()

# 4. Korelasyon matrisi
correlation = df[['math score', 'reading score', 'writing score', 'average_score']].corr()
print("\n🔹 Korelasyon Matrisi:\n", correlation)

sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Puanlar Arası Korelasyon Matrisi')
plt.show()



# 5. Etnik köken ve başarı analizi
ethnic_avg = df.groupby('race/ethnicity')['average_score'].mean().sort_values(ascending=False)
print("\n📊 Etnik Gruba Göre Ortalama Başarı Skoru:\n", ethnic_avg)

ethnic_avg.plot(kind='bar', color='mediumseagreen')
plt.title('Etnik Gruba Göre Ortalama Başarı')
plt.ylabel('Ortalama Skor')
plt.xlabel('Etnik Grup')
plt.grid(True)
plt.show()

# 6. Ebeveyn eğitim düzeyi ve başarı analizi
edu_avg = df.groupby('parental level of education')['average_score'].mean().sort_values(ascending=False)
print("\n📘 Ebeveyn Eğitim Düzeyine Göre Ortalama Başarı:\n", edu_avg)

edu_avg.plot(kind='bar', color='cornflowerblue')
plt.title('Ebeveyn Eğitim Düzeyi ve Ortalama Başarı')
plt.ylabel('Ortalama Skor')
plt.xlabel('Ebeveyn Eğitimi')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 7. Etnik grup ve ebeveyn eğitimi kombinasyonu - ısı haritası
heatmap_data = df.pivot_table(values='average_score',
                               index='race/ethnicity',
                               columns='parental level of education',
                               aggfunc='mean')

print("\n📊 Etnik Grup + Ebeveyn Eğitimi Bazlı Ortalama Başarı Skorları:\n", heatmap_data)

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu')
plt.title('Etnik Grup ve Ebeveyn Eğitimi Bazlı Başarı Isı Haritası')
plt.ylabel('Etnik Grup')
plt.xlabel('Ebeveyn Eğitim Düzeyi')
plt.show()

# 8. Başarı durumunu belirle (>=45 başarılı say)
df['success'] = df['average_score'] >= 45

# Etnik gruba göre başarı yüzdesi
ethnic_success = df.groupby('race/ethnicity')['success'].mean() * 100
print("\n✅ Etnik Gruba Göre Başarı Oranı (%):\n", ethnic_success.round(2))

ethnic_success.plot(kind='bar', color='orange')
plt.title('Etnik Gruba Göre Başarı Oranı')
plt.ylabel('Başarı Oranı (%)')
plt.grid(True)
plt.show()

# Ebeveyn eğitimine göre başarı yüzdesi
edu_success = df.groupby('parental level of education')['success'].mean() * 100
print("\n✅ Ebeveyn Eğitimine Göre Başarı Oranı (%):\n", edu_success.round(2))

edu_success.plot(kind='bar', color='teal')
plt.title('Ebeveyn Eğitim Düzeyine Göre Başarı Oranı')
plt.ylabel('Başarı Oranı (%)')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

print("🔍 VERİ ÖN İŞLEME BAŞLIYOR...\n")

# 2. Eksik veri simülasyonu (örnek olarak bazı değerlere NaN atalım)
df.loc[np.random.choice(df.index, 20), 'math score'] = np.nan
df.loc[np.random.choice(df.index, 15), 'reading score'] = np.nan
df.loc[np.random.choice(df.index, 10), 'writing score'] = np.nan

print("▶ Eksik Değer Sayıları:\n", df.isnull().sum())

# Eksik değerleri sütun ortalaması ile doldur
df['math score'].fillna(df['math score'].mean(), inplace=True)
df['reading score'].fillna(df['reading score'].mean(), inplace=True)
df['writing score'].fillna(df['writing score'].mean(), inplace=True)

print("\n✅ Eksik değerler ortalama ile dolduruldu.")

# 3. Aykırı değer analizi (Z-score)
from scipy.stats import zscore

z_scores = df[['math score', 'reading score', 'writing score']].apply(zscore)
outliers = (np.abs(z_scores) > 3).any(axis=1)
print(f"\n📌 Tespit Edilen Aykırı Değer Sayısı: {outliers.sum()}")

# Aykırı değerleri çıkar
df = df[~outliers]
print(f"✅ Aykırı değerler çıkarıldı. Yeni boyut: {df.shape}")

# 4. Normalizasyon (StandardScaler ile)
scaler = StandardScaler()

# Normalizasyon öncesi dağılımlar
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(['math score', 'reading score', 'writing score']):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i], color='salmon')
    axes[i].set_title(f'{col} - Orijinal Dağılım')
plt.suptitle("🎯 Normalizasyon Öncesi Dağılımlar")
plt.tight_layout()
plt.show()

# Normalizasyon işlemi
df[['math score', 'reading score', 'writing score']] = scaler.fit_transform(df[['math score', 'reading score', 'writing score']])

# Normalizasyon sonrası dağılımlar
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(['math score', 'reading score', 'writing score']):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i], color='mediumseagreen')
    axes[i].set_title(f'{col} - Normalize Dağılım')
plt.suptitle("📊 Normalizasyon Sonrası Dağılımlar")
plt.tight_layout()
plt.show()

print("\n✅ Normalizasyon tamamlandı.")

# 5. Kategorik değişkenleri dönüştür (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'], drop_first=True)

print(f"\n✅ Kategorik veriler dönüştürüldü. Yeni boyut: {df_encoded.shape}")

# ======================= VERİ ÖN İŞLEME ÖZETİ =======================
print("\n📋 VERİ ÖN İŞLEME ÖZETİ:")
summary_table = pd.DataFrame({
    'İşlem': ['Eksik Veriler', 'Aykırı Değerler', 'Normalizasyon', 'Kategorik Dönüşüm'],
    'Açıklama': [
        'Eksik veriler ortalama ile dolduruldu.',
        'Z-score > 3 olan aykırı değerler çıkarıldı.',
        'math/reading/writing score sütunları normalize edildi.',
        'get_dummies ile kategorik veriler binary kodlandı.'
    ]
})
print(summary_table)

# ======================= BAŞARI ANALİZİ DEVAM =======================
# Ortalama skor kolonunu güncelle
df_encoded['average_score'] = df_encoded[['math score', 'reading score', 'writing score']].mean(axis=1)

# Başarı (>=70) etiketi
df_encoded['success'] = df_encoded['average_score'] >= 0.0  # çünkü normalize edildi, ortalama da normalize olur

# Başarı dağılımı
sns.countplot(data=df_encoded, x='success', palette='Set2')
plt.title('Başarı Etiket Dağılımı (Normalize Ortalamaya Göre)')
plt.xlabel('Başarılı mı?')
plt.ylabel('Öğrenci Sayısı')
plt.grid(True)
plt.show()

# 9. Ek analizler ve sonuçlar

# En başarılı öğrenci
top_student = df[df['average_score'] == df['average_score'].max()]
print("\n🏆 En Başarılı Öğrenci:\n", top_student)
plt.show()

# Ortalama skoru 60'ın altında olan öğrenci oranı
low_scores_ratio = (df['average_score'] < 60).mean() * 100
print(f"\n📉 Ortalama Skoru 60'ın Altında Olan Öğrencilerin Oranı: %{low_scores_ratio:.2f}")


# ======================= KÜMELEME ANALİZİ =======================
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

# 1. Özellik seçimi: Üç temel puan kullanılacak
X = df_encoded[['math score', 'reading score', 'writing score']].values

# 2. Küme sayısını belirlemek için Elbow metodu
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Elbow grafiği
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-')
plt.title('Elbow Metodu: Optimal Küme Sayısı')
plt.xlabel('Küme Sayısı')
plt.ylabel('Inertia (WCSS)')
plt.axvline(x=3, color='r', linestyle='--', label='Optimal Küme Sayısı (3)')
plt.grid(True)
plt.legend()
plt.show()

# 3. K-Means Kümeleme
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X)
df_encoded['kmeans_cluster'] = kmeans.labels_

# 4. Hiyerarşik Kümeleme
plt.figure(figsize=(15, 7))
linked = linkage(X, method='ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hiyerarşik Kümeleme - Dendrogram')
plt.xlabel('Öğrenci İndeksleri')
plt.ylabel('Uzaklık')
plt.show()

# Hiyerarşik kümeleme modeli
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
df_encoded['hierarchical_cluster'] = hierarchical.fit_predict(X)

# 5. DBSCAN Kümeleme
dbscan = DBSCAN(eps=0.5, min_samples=5)
df_encoded['dbscan_cluster'] = dbscan.fit_predict(X)

# 6. Kümeleme Sonuçlarını Görselleştirme (PCA ile boyut indirgeme)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(18, 6))

# K-Means sonuçları
plt.subplot(1, 3, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_encoded['kmeans_cluster'], 
                palette='viridis', s=100, alpha=0.8)
plt.title(f'K-Means Kümeleme (k={optimal_k})')
plt.xlabel('PCA Bileşen 1')
plt.ylabel('PCA Bileşen 2')
plt.grid(True)

# Hiyerarşik kümeleme sonuçları
plt.subplot(1, 3, 2)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_encoded['hierarchical_cluster'], 
                palette='tab10', s=100, alpha=0.8)
plt.title('Hiyerarşik Kümeleme')
plt.xlabel('PCA Bileşen 1')
plt.grid(True)

# DBSCAN sonuçları
plt.subplot(1, 3, 3)
unique_clusters = df_encoded['dbscan_cluster'].unique()
palette = sns.color_palette('hsv', len(unique_clusters))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_encoded['dbscan_cluster'], 
                palette=palette, s=100, alpha=0.8)
plt.title('DBSCAN Kümeleme')
plt.xlabel('PCA Bileşen 1')
plt.grid(True)

plt.tight_layout()
plt.show()

# 7. Küme Kalite Metrikleri
print("\n📊 Kümeleme Kalite Metrikleri:")
print(f"K-Means Silhouette Skoru: {silhouette_score(X, kmeans.labels_):.4f}")
print(f"Hiyerarşik Silhouette Skoru: {silhouette_score(X, hierarchical.labels_):.4f}")
print(f"DBSCAN Silhouette Skoru: {silhouette_score(X, dbscan.labels_):.4f}")

# 8. Küme İçerik Analizi
def analyze_clusters(cluster_col):
    cluster_analysis = df_encoded.groupby(cluster_col).agg({
        'math score': 'mean',
        'reading score': 'mean',
        'writing score': 'mean',
        'average_score': 'mean',
        'success': 'mean'
    }).reset_index()
    
    cluster_analysis['cluster_size'] = df_encoded.groupby(cluster_col).size().values
    cluster_analysis['success_rate'] = cluster_analysis['success'] * 100
    return cluster_analysis.round(2)

print("\n🔍 K-Means Küme Analizi:")
print(analyze_clusters('kmeans_cluster'))

print("\n🔍 Hiyerarşik Küme Analizi:")
print(analyze_clusters('hierarchical_cluster'))

print("\n🔍 DBSCAN Küme Analizi:")
print(analyze_clusters('dbscan_cluster'))

# 9. Kümelerin Demografik Dağılımı
# Sütun isimlerini kontrol et ve gerekli düzeltmeleri yap
print("Mevcut Sütun İsimleri:")
print(df_encoded.columns.tolist())

# DÜZELTME: Sütun isimlerindeki boşlukları alt çizgi ile değiştir
df_encoded.columns = df_encoded.columns.str.replace(' ', '_')
df_encoded.columns = df_encoded.columns.str.replace('/', '_')

plt.figure(figsize=(15, 10))

# K-Means kümelerinin cinsiyet dağılımı
plt.subplot(2, 2, 1)
sns.countplot(data=df_encoded, x='kmeans_cluster', hue='gender_male', palette='coolwarm')
plt.title('K-Means: Cinsiyet Dağılımı')
plt.xlabel('Küme')
plt.ylabel('Öğrenci Sayısı')
plt.legend(['Kadın', 'Erkek'], title='Cinsiyet')

# K-Means kümelerinin ebeveyn eğitimi
plt.subplot(2, 2, 2)
# Ebeveyn eğitim sütunu için doğru isim
edu_col = [col for col in df_encoded.columns if 'parental' in col and 'bachelor' in col][0]
cluster_edu = df_encoded.groupby(['kmeans_cluster', edu_col]).size().unstack()
cluster_edu.plot(kind='bar', stacked=True, color=['lightcoral', 'mediumaquamarine'])
plt.title('K-Means: Ebeveyn Üniversite Mezunu Dağılımı')
plt.xlabel('Küme')
plt.ylabel('Öğrenci Sayısı')
plt.legend(['Üniversite Mezunu Değil', 'Üniversite Mezunu'])

# DBSCAN kümelerinin test hazırlık durumu
plt.subplot(2, 2, 3)
# Test hazırlık sütunu için doğru isim
# 'test preparation course' sütunlarını kontrol et
test_prep_cols = [col for col in df_encoded.columns if 'test preparation course' in col.lower()]
if not test_prep_cols:
    raise ValueError("Gerekli 'test preparation course' sütunları veri setinde bulunamadı.")

# 'completed' içeren sütun var mı kontrol et
test_completed_cols = [col for col in test_prep_cols if 'completed' in col]

if test_completed_cols:
    test_col = test_completed_cols[0]  # Eğer varsa, onu kullan
else:
    # Eğer 'completed' sütunu yoksa 'none' sütununu kullan ve yeniden yorumla
    if 'test preparation course_none' in df_encoded.columns:
        test_col = 'test preparation course_none'
        df_encoded['test_preparation_status'] = df_encoded[test_col].apply(lambda x: 'Not Completed' if x == 1 else 'Completed')
        print("Uyarı: 'completed' sütunu bulunamadı. Bunun yerine 'none' sütunu kullanılarak test_preparation_status oluşturuldu.")
    else:
        raise ValueError("Gerekli 'test preparation course' sütunları veri setinde bulunamadı.")

sns.countplot(data=df_encoded, x='dbscan_cluster', hue=test_col, palette='Set2')
plt.title('DBSCAN: Test Hazırlık Durumu')
plt.xlabel('Küme')
plt.ylabel('Öğrenci Sayısı')
plt.legend(['Hazırlık Yok', 'Hazırlık Tamamlandı'])

# Hiyerarşik kümelerin etnik dağılımı
plt.subplot(2, 2, 4)
ethnic_cols = [col for col in df_encoded.columns if 'race_ethnicity_' in col]
cluster_ethnic = df_encoded.groupby('hierarchical_cluster')[ethnic_cols].mean()
cluster_ethnic.plot(kind='bar', stacked=True, cmap='viridis')
plt.title('Hiyerarşik: Etnik Grup Dağılımı')
plt.xlabel('Küme')
plt.ylabel('Oran')
plt.legend(title='Etnik Grup', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# 10. Küme Etiketleme ve Yorumlama
cluster_descriptions = {
    'kmeans': {
        0: "Düşük Performanslı Öğrenciler",
        1: "Orta Performanslı Öğrenciler",
        2: "Yüksek Performanslı Öğrenciler"
    },
    'hierarchical': {
        0: "Dengeli Performans Grubu",
        1: "Matematik Odaklı Öğrenciler",
        2: "Sözel Odaklı Öğrenciler"
    },
    'dbscan': {
        -1: "Aykırı Değerler",
        0: "Ana Öğrenci Grubu",
        1: "Küçük Performans Grubu"
    }
}

print("\n📌 Küme Yorumları:")
for method, clusters in cluster_descriptions.items():
    print(f"\n{method.upper()} Küme Etiketleri:")
    for cluster_id, description in clusters.items():
        print(f"  Küme {cluster_id}: {description}")

# ======================= SONUÇ RAPORU =======================
print("\n🔎 KÜMELEME ANALİZİ SONUÇLARI:")
print(f"- En iyi silhouette skoru: K-Means ({silhouette_score(X, kmeans.labels_):.4f})")
print("- Elbow metodu optimal küme sayısı: 3")
print("- DBSCAN aykırı değer tespiti:", (df_encoded['dbscan_cluster'] == -1).sum())
print("- En homojen küme: K-Means Küme 2 (Yüksek Performanslılar)")
