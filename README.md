
## 🎓 Python ile Veri Bilimi Dersi - Students Performance in Exams Projesi

### 📌 Proje Amacı

Bu proje, öğrencilerin sınav skorları ile demografik özellikleri arasındaki ilişkileri analiz etmek, başarı düzeylerine göre segmentasyon (kümeleme) yapmak ve eğitim kurumları için veri temelli stratejiler geliştirmek amacıyla gerçekleştirilmiştir.

---

### 🧩 Kullanılan Veri Seti

Veri seti, 1000 öğrencinin aşağıdaki bilgilerini içermektedir:

* Cinsiyet (`gender`)
* Etnik Grup (`race/ethnicity`)
* Ebeveyn Eğitim Düzeyi (`parental level of education`)
* Öğle Yemeği Durumu (`lunch`)
* Test Hazırlık Kursu Katılımı (`test preparation course`)
* Matematik, Okuma ve Yazma Puanları (`math`, `reading`, `writing score`)
* Ortalama Başarı (`average_score`)
* Başarı Etiketi (`success`: True/False)

---

### 🛠️ Kullanılan Teknolojiler

* **Python** (Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, Scipy)
* **Jupyter Notebook / VS Code**
* **Görselleştirme araçları**: Matplotlib, Seaborn

---

### 📊 Uygulanan Teknikler

#### 1. **Veri Ön İşleme**

* Eksik veriler dolduruldu
* Aykırı değerler Z-Score ile temizlendi
* Veriler StandardScaler ile normalize edildi
* Kategorik veriler One-Hot Encoding ile dönüştürüldü

#### 2. **Kümeleme (Clustering)**

* **Algoritmalar:** K-Means, Hiyerarşik Kümeleme, DBSCAN
* **Küme Sayısı Belirleme:** Elbow yöntemi ile optimum k=3 bulundu
* **Küme Kalitesi:** Silhouette skorları ile değerlendirildi
* **Küme Etiketleme:** Düşük, Orta, Yüksek başarı grupları

---

### 🧠 Sonuçlar

* Öğrenciler başarı düzeylerine göre başarılı şekilde segmentlendi.
* Özellikle **ebeveyn eğitim düzeyi** ve **test hazırlık kursuna katılım**, başarıda etkili değişkenler olarak belirlendi.
* Aykırı değerlerin temizlenmesi, analiz kalitesini artırdı.
* Kümeleme sonucunda eğitim kurumları için hedefli destek planları önerilebilir hale geldi.

---

### 📈 Kümeleme Kalite Metrikleri

| Yöntem     | Silhouette Skoru |
| ---------- | ---------------- |
| K-Means    | 0.2415           |
| Hiyerarşik | 0.1772           |
| DBSCAN     | 0.1720           |

> ⚠️ Not: Gerçek dünya verilerinde bu skorların çok yüksek olması beklenmez. Önemli olan anlamlı gruplar elde etmektir.

---

### 🧩 Yönetim Bilişim Sistemlerine Katkısı

* **Veriye Dayalı Karar Verme**
* **Eğitimde Kişiselleştirme**
* **Performans Takibi ve Erken Müdahale**
* **İş Zekâsı Entegrasyonu**

---

### 📁 Projeyi Çalıştırmak İçin

1. Gerekli Python kütüphanelerini yükleyin (`pip install -r requirements.txt`)
2. `analiz.py` veya `notebook.ipynb` dosyasını açın
3. Tüm hücreleri çalıştırın

---

### ✨ Projenin Gücü

Bu proje, sadece bir teknik uygulama değil; **veri biliminin eğitimde stratejik kararlar için nasıl kullanılabileceğini gösteren güçlü bir örnektir.**

---
![cinsiyet göre matamatik notları](https://github.com/user-attachments/assets/1de6babb-b3d8-4890-9bbb-a43b870eca12)

        ----------------------------------------------------------------------------------------------------
![Test haz göre okuma skoru](https://github.com/user-attachments/assets/50f00b6d-b6f3-420a-808e-95c696b49124)

----------------------------------------------------------------------------------------------------

![Ebeveyn Eğitim ve ortalama Skor](https://github.com/user-attachments/assets/5b5e1b3f-3748-4be0-b8bb-303b628ed5ab)

----------------------------------------------------------------------------------------------------

![enik](https://github.com/user-attachments/assets/3d28f362-cd03-4e4f-b81c-e807fd71b60c)

----------------------------------------------------------------------------------------------------

![koralasyon](https://github.com/user-attachments/assets/98667b52-f10a-4ac6-8e71-e7a90cdc810a)

----------------------------------------------------------------------------------------------------
![Etnik grubu göre başarı oranı](https://github.com/user-attachments/assets/506c87aa-4a48-4f00-90dd-bd25c1e1e8a1)

![Ebeveyn eğitim düzeyin göre başarı oranı](https://github.com/user-attachments/assets/ce4f8251-18cd-4168-96ed-c6d828811962)


----------------------------------------------------------------------------------------------------

![K-Means kümeleme ve Hiyarşik kümeleme DBSCAN Kü](https://github.com/user-attachments/assets/b47f5229-2b06-47c1-a675-a4d75473b937)


----------------------------------------------------------------------------------------------------

![Hiyeraşik kümeleme](https://github.com/user-attachments/assets/4433d410-cbe5-414c-8db9-c5ec9a60330e)

----------------------------------------------------------------------------------------------------

![Elbow Metodu op kü sayısı](https://github.com/user-attachments/assets/c1020579-f7d7-4b65-ba26-989e95546265)

----------------------------------------------------------------------------------------------------


