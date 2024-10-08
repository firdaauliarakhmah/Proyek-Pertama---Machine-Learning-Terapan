# -*- coding: utf-8 -*-
"""Stroke Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Bfw7gE6Uq3mPtQpnrWZMTztPwfLMVZfI

# **Model Prediksi Berbasis Machine Learning untuk Diagnosis Risiko Terjadinya Stroke Berdasarkan Analisis Faktor Hipertensi dari Data Pasien**
---
By : [Firda Aulia Rakhmah](https://www.dicoding.com/users/firda_aulia_rakhmah)

# **Deskripsi Proyek**

**Latar Belakang Proyek**


Stroke merupakan salah satu penyebab utama kematian dan kecacatan di seluruh dunia. Faktor risiko seperti hipertensi (tekanan darah tinggi) telah diketahui sebagai salah satu penyebab utama terjadinya stroke. Dengan adanya data medis yang semakin berkembang, metode berbasis *Machine Learning* (ML) dapat memberikan solusi yang efektif untuk menganalisis risiko pasien terkena stroke, khususnya berdasarkan riwayat hipertensi mereka. Memprediksi risiko stroke sejak dini dapat membantu pengambilan keputusan medis yang lebih tepat waktu, dan pada akhirnya menyelamatkan lebih banyak nyawa.

<br>


**Tujuan proyek**

Proyek ini bertujuan untuk membangun model prediksi berbasis *Machine Learning* yang dapat membantu mendiagnosis risiko terjadinya stroke pada pasien dengan mempertimbangkan faktor hipertensi dan faktor risiko lainnya yang relevan. Model ini akan dikembangkan menggunakan data historis pasien untuk mempelajari pola dan hubungan antara hipertensi serta risiko stroke.

## **Import library**
"""

from google.colab import drive
drive.mount('/content/gdrive')

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/gdrive/MyDrive/2-Kuliah/Dicoding'

# Data manipulation
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Machine learning
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

"""## **Data Understanding**

Berikut ini adalah informasi mengenai dataset yang digunakan sebagai bahan penyelesaian proyek.

<br>


| Jenis | Keterangan |
| ------ | ------ |
| Title | Stroke Prediction Dataset |
| Source | [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data) |
| Visibility | Publik |
| Tags | Health, Health Conditions, Public Health, Healthcare, Binary Classification|
| Usability | 10.00 |

### Membaca dataset
"""

df = pd.read_csv('Stroke Dataset.csv')
df

"""Dari dataframe yang di tampilkan di atas, kita dapat menyimpulkan bahwa pada dataset ini memiliki beberapa atribut (kolom) yang ada didalamnya. Berikut adalah penjelasan mengenai atribut dalam dataset risiko stroke:



- `id` : Nomor unik yang dimiliki oleh setiap subjek dalam dataset.
- `gender` : Menyatakan jenis kelamin subjek (misalnya, laki-laki atau perempuan).
- `age` : Usia subjek dalam dataset.
- `hypertension` : Menunjukkan apakah subjek memiliki riwayat hipertensi. Nilai 0 berarti tidak memiliki hipertensi, sedangkan nilai 1 berarti subjek mengidap hipertensi.
- `heart_disease` : Menyatakan apakah subjek memiliki penyakit jantung. Nilai 0 berarti tidak memiliki penyakit jantung, sedangkan nilai 1 berarti subjek mengidap penyakit jantung.
-` ever_married` : Menyatakan status pernikahan subjek, apakah sudah pernah menikah atau belum.
- `work_type` : Menggambarkan jenis pekerjaan yang dimiliki oleh subjek.
- `Residence_type` : Menunjukkan jenis tempat tinggal subjek, apakah tinggal di daerah perkotaan atau pedesaan.
- `avg_glucose_level` : Menyatakan kadar gula darah rata-rata subjek.
- `bmi` : Indeks Massa Tubuh (BMI) yang menggambarkan kategori berat badan subjek.
- `smoking_status` : Menunjukkan kebiasaan merokok subjek, apakah perokok aktif, mantan perokok, atau tidak merokok
- `stroke` : Variabel target yang menyatakan apakah subjek telah terdiagnosis stroke. Nilai 0 menunjukkan tidak terkena stroke, sedangkan nilai 1 menunjukkan subjek terkena stroke.

### Melihat jumlah baris dan kolom
"""

df.shape

"""### Melihat informasi pada dataset"""

df.info()

"""**Penjelasan :**



- `RangeIndex` : Dataset memiliki 5110 baris, dari indeks 0 hingga 5109.
- `Data Columns` : Terdapat 12 kolom dalam dataset.
- `Non-Null Count` : Menunjukkan jumlah nilai yang tidak kosong dalam setiap kolom. Kolom bmi memiliki 4909 nilai non-null, yang berarti ada 201 baris dengan nilai yang hilang (NaN).
- `Dtype` : Tipe data dari setiap kolom. Misalnya, gender adalah tipe objek (string), sedangkan age adalah tipe float64.
- `Memory Usage` : Dataset menggunakan sekitar 479.3 KB di memori.

### Check total missing value tiap kolom
"""

df.isnull().sum()

"""**Penjelasan :**

Dari data di atas dapat di simpulkan bahwa kolom id, gender, age, hypertension heart_disease, ever_married, work_type, Residence_type, avg_glucose_level smoking_status, stroke memiliki jumlah nilai null (hilang) 0 yang berarti data di dalamnya lengkap. sedangkan, untuk kolom bmi memiliki 201 data yang nilainya hilang.

### Check data duplicate
"""

df.duplicated().sum()

"""### Check data apakah sudah balance ?"""

df.stroke.value_counts()

"""## **Visualisasi Data**

### **Visualisasi Univariate Analysis**

Visualisasi Univariate Analysis adalah proses menganalisis dan menggambarkan distribusi satu variabel pada satu waktu, untuk memahami karakteristik dan perilakunya. Dalam analisis univariate, fokusnya adalah pada satu variabel, baik itu numerik atau kategorikal. Pada tahap ini dilakukan untuk melihat distribusi data setiap variabel
"""

df.head(10)

"""#### **Distribusi jumlah subjek berdasarkan jenis kelamin**




"""

df['gender'].value_counts().plot(kind='bar', color='skyblue')

# Judul dan label sumbu
plt.title('Distribusi Gender')
plt.xlabel('Gender')
plt.ylabel('Jumlah')

plt.show()

"""#### **Distribusi jumlah subjek berdasarkan status pernikahan**

"""

df['ever_married'].value_counts().plot(kind='bar', color='skyblue')

# Judul dan label sumbu
plt.title('Distribusi Status Pernikahan')
plt.xlabel('Status Pernikahan')
plt.ylabel('Jumlah')

plt.show()

"""#### **Distribusi  jumlah subjek berdasarkan jenis pekerjaan**"""

df['work_type'].value_counts().plot(kind='bar', color='skyblue')

# Judul dan label sumbu
plt.title('Distribusi Jenis Pekerjaan')
plt.xlabel('Jenis Pekerjaan')
plt.ylabel('Jumlah')

plt.xticks(rotation=45)
plt.show()

"""#### **Distribusi subjek berdasarkan tempat tinggal, apakah mereka tinggal di area urban atau rural**"""

df['Residence_type'].value_counts().plot(kind='bar', color='lightblue')

# Judul dan label sumbu
plt.title('Distribusi Jenis Tempat Tinggal')
plt.xlabel('Jenis Tempat Tinggal')
plt.ylabel('Jumlah')

plt.xticks(rotation=0)
plt.show()

"""#### **Distribusi dari fitur numerik dalam dataset**"""

# Membuat histogram untuk semua kolom numerik
df.hist(figsize=(16, 12), bins=20, color='skyblue', edgecolor='black')

# Menampilkan grafik
plt.suptitle('Distribusi Fitur Numerik', fontsize=16)
plt.show()

"""#### **Visualisasi data hipertensi yang berkaitan dengan stroke ?**




"""

# Bagaimana visualisasi hipertensi yang berkaitan dengan stroke ?

plt.figure(figsize=(10, 6))
sns.countplot(x='hypertension', hue='stroke', data=df)
plt.title('Distribusi Hipertensi Berdasarkan Status Stroke Pasien')

plt.xlabel('Hipertensi (0 = Tidak, 1 = Ya)')
plt.ylabel('Jumlah Pasien')

plt.legend(title='Stroke', labels=['Tidak Stroke', 'Stroke'])
plt.show()

"""#### **Hubungan usia dengan risiko terjadinya stroke ?**"""

# Hubungan usia dengan risiko terjadinya stroke

plt.figure(figsize=(12, 8))
sns.boxplot(x='stroke', y='age', hue='hypertension', data=df)
plt.title('Distribusi Usia Berdasarkan Stroke dan Hipertensi')

plt.xlabel('Stroke (0 = Tidak, 1 = Ya)')
plt.ylabel('Usia')

plt.legend(title='Hipertensi', labels=['Tidak Hipertensi', 'Hipertensi'])
plt.show()

"""## **Data Preparation**

Data preparation adalah proses mempersiapkan data mentah untuk dianalisis dengan membersihkan, mengubah, dan menyusun data agar sesuai dengan kebutuhan analisis.Tujuannya adalah untuk memastikan data yang digunakan dalam analisis berkualitas tinggi, konsisten, dan sesuai dengan model yang akan diterapkan, sehingga hasil analisis lebih akurat dan dapat diandalkan.
"""

df.head(5)

"""## Drop kolom yang tidak diperlukan"""

df.drop("id",axis=1,inplace=True)
df.head()

"""kolom yang harus di hapus adalah id, karena id tidak memiliki kepentingan untuk dimasukkan ke dalam pembuatan model Machine Learning.

## Menghapus kategori kolom yang tidak diperukan untuk pembuatan model machine learning
"""

categorical = list(df.dtypes[df.dtypes == 'object'].index)
categorical
for col in categorical:
    df[col] = df[col].str.lower().str.replace(" ", "_")

for col in categorical:
    print(col)
    print(df[col].unique())

df.drop(df.loc[df['smoking_status']=='unknown'].index, inplace=True)

df.drop(df.loc[df['gender']=='other'].index, inplace=True)

df.reset_index(drop=True)

"""Kategori yang dihapus adalah unknown pada kolom smoking_status dan other pada kolom gender

## Mengisi nilai yang hilang pada kolom 'bmi' dengan nilai rata-rata

Karena pada kolom 'bmi' sebanyak 201 data kosong maka akan diterapkan teknik pengisian nilai dengan nilai rata-rata (mean)
"""

if df['bmi'].isnull().any():
    df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# Menampilkan jumlah nilai hilang di setiap kolom
display(df.isnull().sum().to_frame().reset_index().rename({'index': 'Variables', 0: 'Missing Values'}, axis=1).style.background_gradient('gnuplot2_r'))

"""## Melakukan Upsample untuk menangani ketidakseimbangan kelas

Karena jumlah data untuk kasus stroke tidak seimbang, saya melakukan upsampling untuk mencapai keseimbangan dalam dataset sebelum mengolahnya dengan metode machine learning
"""

# Upsample untuk menangani ketidakseimbangan kelas

# Memisahkan data berdasarkan kelas
df_1 = df[df.stroke == 0]  # k.mayoritas
df_2 = df[df.stroke == 1]  # k.minoritas

# Melakukan upsampling pada kelas minoritas
df_2_upsampled = resample(df_2,
                          replace=True,
                          n_samples=len(df_1),  # Mengambil sampel sebanyak jumlah kelas mayoritas
                          random_state=123)

# Menggabungkan kembali dataset mayoritas dengan dataset minoritas yang di-oversample
df_upsampled = pd.concat([df_1, df_2_upsampled])

# Menampilkan distribusi kelas setelah upsampling
print(df_upsampled.stroke.value_counts())

df_upsampled.reset_index(drop=True)

# Menghitung jumlah setiap kelas dalam 'stroke'
stroke_label = df_upsampled.stroke.value_counts()

# Ukuran gambar
plt.figure(figsize=(8, 4))

# Membuat grafik batang menggunakan seaborn
sns.barplot(x=stroke_label.index, y=stroke_label.values)

# Menambahkan label sumbu
plt.xlabel('Stroke', fontsize=15)
plt.ylabel('Jumlah', fontsize=15)
plt.title('Distribusi Kelas Stroke', fontsize=16)

plt.show()

"""## Melihat visualisasi distribusi kolom numerik setelah upsample"""

# Kolom numerik
numerical = [col for col in df_upsampled.columns if col not in categorical]

# Loop distribusi tiap kolom numerik
for i in numerical:
    plt.figure(figsize=(6,4))

    # Menggunakan histplot untuk distribusi data
    sns.histplot(df_upsampled[i], kde=True, color='red')

    plt.title(f'Distribution of {i}', fontsize=15)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(fontsize=8)

    # Menampilkan plot
    plt.show()
    print('\n')

"""## Melihat visualisasi distribusi kolom kategorikal setelah upsample"""

def sort_order(column):
    orders = (df_upsampled.groupby(column)['stroke'].mean().sort_values(ascending=False)).index
    return orders

for i in categorical:
    if df_upsampled[i].nunique() < 20:
        f, ax = plt.subplots(figsize=(7, 7))

        sns.barplot(x=df_upsampled[i], y=df_upsampled['stroke'], order=sort_order(i), palette='Pastel1')

        plt.xlabel(f'{i}', fontsize=12)
        plt.ylabel('Stroke', fontsize=12)
        plt.xticks(fontsize=15, rotation=90)

        plt.show()
        print('\n')

"""### **Korelasi matriks setelah upsample**

Untuk menunjukkan hubungan antara dua atau lebih variabel. Dalam konteks analisis data dan machine learning, korelasi matriks digunakan untuk Menunjukkan Hubungan Antara Variabel, memahami struktur data, dan untuk mengidentifikasi outlier.
"""

# Memilih hanya kolom numerik
numeric_columns = df_upsampled.select_dtypes(include=['number'])

# Membuat heatmap untuk matriks korelasi
plt.figure(figsize=(12, 10))

# Matriks korelasi dibulatkan ke 2 desimal
correlation_matrix = numeric_columns.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='bwr', center=0, linewidths=0.5, fmt='.2f')

plt.title('Correlation Matrix', size=15)
plt.show()

# Memilih hanya kolom numerik
numeric_columns = df_upsampled.select_dtypes(include=['number'])

# Menghitung matriks korelasi
correlation_matrix = numeric_columns.corr()

# Mendapatkan korelasi dengan 'stroke' dan mengurutkannya
stroke_correlations = correlation_matrix['stroke'].sort_values(ascending=False)[1:]

# Menampilkan hasil
print(stroke_correlations)

"""## Outlier

Pendeteksian outlier adalah langkah krusial dalam analisis data yang bertujuan untuk meningkatkan kualitas dan konsistensi data, serta memastikan hasil analisis yang akurat. Dengan mengidentifikasi outlier, kita dapat mendeteksi kesalahan dalam pengukuran atau input data, memahami variabilitas yang ada, dan menemukan kejadian ekstrem yang mungkin memiliki implikasi signifikan. Selain itu, menangani outlier juga dapat meningkatkan kinerja model prediktif, sehingga model dapat lebih baik dalam generalisasi dan memprediksi data baru. Secara keseluruhan, pendeteksian outlier memungkinkan analisis data yang lebih mendalam dan relevan, serta membuka peluang untuk mendapatkan wawasan baru yang bermanfaat dalam pengambilan keputusan.
"""

# Mendeteksi Outlier pada data

def outlier(data1):
    # Menghitung kuartil pertama (Q1) dan kuartil ketiga (Q3)
    Q1, Q3 = np.nanpercentile(data1, [25, 75])

    # Menghitung IQR
    IQR = Q3 - Q1

    # Menghitung batas bawah dan batas atas
    lowerRange = Q1 - (1.5 * IQR)
    upperRange = Q3 + (1.5 * IQR)

    return lowerRange, upperRange

df_upsampled.value_counts('stroke')

"""### Melihat distribusi fitur sebelum outlier dihapus"""

plt.figure(figsize=(20,10))

plt.subplot(2,4,1)
sns.distplot(df_upsampled['age'])
plt.xlabel('Age',fontsize = 12)
plt.grid()

plt.subplot(2,4,2)
sns.distplot(df_upsampled['avg_glucose_level'])
plt.xlabel('Avg Glucose Level',fontsize = 12)
plt.grid()

plt.subplot(2,4,3)
sns.distplot(df_upsampled['bmi'])
plt.xlabel('BMI',fontsize = 12)
plt.grid()

plt.show()

def graph(y):
    sns.boxplot(x="stroke", y=y, data=df_upsampled)

plt.figure(figsize=(14,7))

plt.subplot(2,4,1)
graph('age')

plt.subplot(2,4,2)
graph('avg_glucose_level')

plt.subplot(2,4,3)
graph('bmi')

plt.show()

lr,ur=outlier(df_upsampled['avg_glucose_level'][df_upsampled.stroke==0])
df_upsampled.drop(df_upsampled.index[(df_upsampled.avg_glucose_level > ur) & (df_upsampled.stroke == 0)],inplace=True)
df_upsampled.drop(df_upsampled.index[(df_upsampled.avg_glucose_level < lr) & (df_upsampled.stroke == 0)],inplace=True)
df_upsampled.value_counts('stroke')

lr,ur=outlier(df_upsampled['avg_glucose_level'][df_upsampled.stroke==0])
df_upsampled.drop(df_upsampled.index[(df_upsampled.avg_glucose_level > ur) & (df_upsampled.stroke == 0)],inplace=True)
df_upsampled.drop(df_upsampled.index[(df_upsampled.avg_glucose_level < lr) & (df_upsampled.stroke == 0)],inplace=True)
df_upsampled.value_counts('stroke')

lr,ur=outlier(df_upsampled['bmi'][df_upsampled.stroke==1])
df_upsampled.drop(df_upsampled.index[(df_upsampled.bmi > ur) & (df_upsampled.stroke == 1)],inplace=True)
df_upsampled.drop(df_upsampled.index[(df_upsampled.bmi < lr) & (df_upsampled.stroke == 1)],inplace=True)
df_upsampled.value_counts('stroke')

lr,ur=outlier(df_upsampled['bmi'][df_upsampled.stroke==1])
df_upsampled.drop(df_upsampled.index[(df_upsampled.bmi > ur) & (df_upsampled.stroke == 0)],inplace=True)
df_upsampled.drop(df_upsampled.index[(df_upsampled.bmi < lr) & (df_upsampled.stroke == 0)],inplace=True)
df_upsampled.value_counts('stroke')

def graph(y):
    sns.boxplot(x="stroke", y=y, data=df_upsampled)

plt.figure(figsize=(14,7))

plt.subplot(2,4,1)
graph('age')

plt.subplot(2,4,2)
graph('avg_glucose_level')

plt.subplot(2,4,3)
graph('bmi')

plt.show()

lr,ur=outlier(df_upsampled['bmi'][df_upsampled.stroke==1])
df_upsampled.drop(df_upsampled.index[(df_upsampled.bmi > ur) & (df_upsampled.stroke == 1)],inplace=True)
df_upsampled.drop(df_upsampled.index[(df_upsampled.bmi < lr) & (df_upsampled.stroke == 1)],inplace=True)
df_upsampled.value_counts('stroke')

lr,ur=outlier(df_upsampled['avg_glucose_level'][df_upsampled.stroke==0])
df_upsampled.drop(df_upsampled.index[(df_upsampled.avg_glucose_level > ur) & (df_upsampled.stroke == 0)],inplace=True)
df_upsampled.drop(df_upsampled.index[(df_upsampled.avg_glucose_level < lr) & (df_upsampled.stroke == 0)],inplace=True)
df_upsampled.value_counts('stroke')

def graph(y):
    sns.boxplot(x="stroke", y=y, data=df_upsampled)

plt.figure(figsize=(14,7))

plt.subplot(2,4,1)
graph('age')

plt.subplot(2,4,2)
graph('avg_glucose_level')

plt.subplot(2,4,3)
graph('bmi')

plt.show()

lr,ur=outlier(df_upsampled['avg_glucose_level'][df_upsampled.stroke==0])
df_upsampled.drop(df_upsampled.index[(df_upsampled.avg_glucose_level > ur) & (df_upsampled.stroke == 0)],inplace=True)
df_upsampled.drop(df_upsampled.index[(df_upsampled.avg_glucose_level < lr) & (df_upsampled.stroke == 0)],inplace=True)
df_upsampled.value_counts('stroke')

lr,ur=outlier(df_upsampled['bmi'][df_upsampled.stroke==1])
df_upsampled.drop(df_upsampled.index[(df_upsampled.bmi > ur) & (df_upsampled.stroke == 1)],inplace=True)
df_upsampled.drop(df_upsampled.index[(df_upsampled.bmi < lr) & (df_upsampled.stroke == 1)],inplace=True)
df_upsampled.value_counts('stroke')

def graph(y):
    sns.boxplot(x="stroke", y=y, data=df_upsampled)

plt.figure(figsize=(14,7))

plt.subplot(2,4,1)
graph('age')

plt.subplot(2,4,2)
graph('avg_glucose_level')

plt.subplot(2,4,3)
graph('bmi')

"""### Melihat distribusi fitur setelah outlier dihapus"""

plt.figure(figsize=(20, 10))

# Plot distribusi untuk kolom 'age'
plt.subplot(2, 4, 1)
sns.histplot(df_upsampled['age'], kde=True, color='blue')
plt.xlabel('Age', fontsize=12)
plt.grid()

# Plot distribusi untuk kolom 'avg_glucose_level'
plt.subplot(2, 4, 2)
sns.histplot(df_upsampled['avg_glucose_level'], kde=True, color='green')
plt.xlabel('Average Glucose Level', fontsize=12)
plt.grid()

# Plot distribusi untuk kolom 'bmi'
plt.subplot(2, 4, 3)
sns.histplot(df_upsampled['bmi'], kde=True, color='orange')
plt.xlabel('BMI', fontsize=12)
plt.grid()

plt.tight_layout()
plt.show()

df_upsampled.reset_index(drop=True,inplace=True)

"""### **One Hot Encoding**

One hot encoding dapat digunakan untuk mengubah kolom-kolom kategorikal menjadi format numerik biner melalui proses one-hot encoding. Tujuannya adalah agar model machine learning yang hanya bisa bekerja dengan data numerik dapat memahami dan menggunakan informasi dari kolom kategorikal seperti gender, ever_married, work_type, Residence_type, dan smoking_status. Setelah dilakukan one-hot encoding, kategori dari kolom-kolom tersebut akan diubah menjadi beberapa kolom biner, yang kemudian dapat digunakan oleh model untuk analisis atau prediksi yang lebih akurat.
"""

final = pd.get_dummies(df_upsampled, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
final.head()

"""### Melihat korelasi matriks setelah encoding"""

plt.figure(figsize=(12, 10))

# Menghitung matriks korelasi
correlation_matrix = final.corr()

# Membuat heatmap untuk matriks korelasi
sns.heatmap(correlation_matrix, annot=True, cmap='bwr', center=0, linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix', fontsize=15)

# Menampilkan heatmap
plt.show()

"""## **Modeling**

### **Membagi data training & testing kemudian melakukan normalisasi**

Membagi data menjadi training dan testing bertujuan untuk mengevaluasi kinerja model secara objektif, di mana data training digunakan untuk melatih model dan data testing untuk menguji model pada data baru, sehingga menghindari overfitting dan memastikan model dapat melakukan generalisasi dengan baik. Sementara itu, normalisasi dilakukan untuk menyelaraskan skala fitur-fitur dalam dataset agar berada dalam rentang yang sama, sehingga algoritma machine learning dapat bekerja lebih efektif tanpa dipengaruhi oleh perbedaan skala antar fitur, yang pada akhirnya meningkatkan akurasi dan kecepatan model dalam mendeteksi pola.
"""

final_features=final[['age', 'avg_glucose_level', 'bmi']]
target = final['stroke']

# Memisahkan data training dan testing
X_train, X_test, y_train, y_test = train_test_split(final_features,target,test_size = 0.2,random_state =2)

# Normalisasi data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""### **Modeling Menggunakan Random Forest Classifier**

Random Forest Classifier adalah algoritma pembelajaran ensemble yang terdiri dari banyak pohon keputusan (decision trees) yang bekerja secara bersama-sama. Random Forest Classifier dari sklearn.ensemble dengan pengaturan n_estimators=30 dan max_features=3. Kelebihan dari algoritma ini adalah kemampuannya untuk menentukan variabel mana yang signifikan dalam proses klasifikasi, yang membantu dalam pemahaman model. Di sisi lain, kelemahan dari algoritma ini adalah kompleksitas yang tinggi, yang dapat menyebabkan waktu pelatihan yang lebih lama dan penggunaan sumber daya yang lebih besar, terutama saat bekerja dengan dataset yang besar.
"""

# Menggunakan Random Forest Classifier

# Inisialisasi Model
rf = RandomForestClassifier(n_estimators=30, max_features=3, random_state=0).fit(X_train_scaled, y_train)
rf_pred= rf.score(X_test_scaled, y_test)

# Evaluasi Model
rf_train_accuracy =rf.score(X_train_scaled,y_train)
rf_accuracy = rf.score(X_test_scaled,y_test)

# Prediksi Probabilitas
pred_prob_rf = rf.predict_proba(X_test_scaled)


print("Training score: {}".format(rf.score(X_train_scaled, y_train)))
print("Test score: {}".format(rf.score(X_test_scaled, y_test)))

"""### **Modeling Menggunakan K-Neigbors Classifier**

KNeighborsClassifier dari sklearn.neighbors adalah algoritma KNN yang melakukan klasifikasi berdasarkan tetangga terdekat. Dengan n_neighbors=2, algoritma ini akan mempertimbangkan dua tetangga terdekat untuk menentukan kelas dari sampel baru. Pemilihan nilai K sangat mempengaruhi kinerja model; dalam hal ini, dua tetangga terdekat akan digunakan untuk memprediksi kelas berdasarkan mayoritas kelas dari dua tetangga tersebut. Metrik jarak, seperti Euclidean, digunakan untuk mengukur kedekatan antar sampel.
"""

# Menggunakan K-Neighbors Classifier

knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
knn_pred=knn.score(X_test_scaled, y_test)

knn_train_accuracy =knn.score(X_train_scaled,y_train)
knn_accuracy = knn.score(X_test_scaled,y_test)
pred_prob_knn = knn.predict_proba(X_test_scaled)

print("Training score: {}".format(knn.score(X_train_scaled, y_train)))
print("Test score: {}".format(knn.score(X_test_scaled, y_test)))

"""## **Evaluasi**

### Feature Importance

Feature importance digunakan untuk mengetahui seberapa besar kontribusi setiap fitur (variabel independen) dalam membantu model melakukan prediksi. Dalam konteks Random Forest (dan beberapa algoritma lainnya), feature importance mengukur seberapa "penting" suatu fitur berdasarkan seberapa sering dan seberapa efektif fitur tersebut digunakan oleh pohon-pohon keputusan dalam hutan untuk membagi data.
"""

final.corr()['stroke'].sort_values(ascending=False)[1:]

feature_final=final[['age','hypertension','heart_disease','avg_glucose_level','bmi','gender_female','gender_male','ever_married_no','ever_married_yes','work_type_children','work_type_govt_job','work_type_never_worked','work_type_private','work_type_self-employed','Residence_type_rural','Residence_type_urban','smoking_status_formerly_smoked','smoking_status_never_smoked','smoking_status_smokes']]
target=final[['stroke']]

"""Menggunakan RandomForestClassifier dari Scikit-learn"""

rf = RandomForestClassifier()
rf_model=rf.fit(feature_final,target)
feat_importances = pd.Series(rf_model.feature_importances_, index=feature_final.columns)
feat_importances.nlargest(12).plot(kind='barh')

"""Berdasarkan data di atas, dapat disimpulkan bahwa fitur Age, Average Glucose Level, dan BMI memiliki peranan yang sangat penting untuk digunakan dalam membuat model machine learning

### **Matriks Evaluasi - Random Forest Classifier**
"""

# Random Forest Classifier

rf_model=RandomForestClassifier(random_state=0)
rf_model.fit(X_train_scaled,y_train)
y_pred=rf_model.predict(X_test_scaled)
from sklearn import metrics

rf_cm = metrics.confusion_matrix(y_test, y_pred)

# Visualisasi confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(rf_cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'bwr');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix - score:'+str(metrics.accuracy_score(y_test,y_pred))
plt.title(all_sample_title, size = 15);
plt.show()
print(metrics.classification_report(y_test,y_pred))

"""### **Matriks Evaluasi - KNeighborsClassifier**"""

k_range = range(1,11)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.vlines(k_range,0, scores, linestyle="dashed", colors='maroon')
plt.ylim(0.70,0.99)
plt.xticks([i for i in range(1,11)]);

# KNeighborsClassifier

knn_model=KNeighborsClassifier(n_neighbors = 2)
knn_model.fit(X_train_scaled,y_train)
y_pred=knn_model.predict(X_test_scaled)
from sklearn import metrics

knn_cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(knn_cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'bwr');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix - score:'+str(metrics.accuracy_score(y_test,y_pred))
plt.title(all_sample_title, size = 15);
plt.show()
print(metrics.classification_report(y_test,y_pred))

"""### **Perbandingan Akurasi**"""

# Perbandingan Akurasi

Model_Name = ['Random Forest','KNeighbors']
Accuracy = [rf_pred,knn_pred]

plt.bar
plt.title('Perbandingan akurasi model')
plt.xlabel('Akurasi')
plt.ylabel('Model')
sns.barplot(x = Accuracy,y = Model_Name)
plt.show()

"""Menggunakan Forest Classifier :

- `Akurasi` : 99%
- `Precision 0` : 100%
- `Recall 0` : 98%
- `Precision 1` : 98%
- `Recall 1` : 100%
- `F1-Score` : 0.99

<br>

Menggunakan K-Neighbors Classifier :

- `Akurasi` : 98%
- `Precision 0` : 100%
- `Recall 0` : 95%
- `Precision 1` : 96%
- `Recall 1` : 100%
- `F1-Score` : 0.98

<br>



Secara keseluruhan penyelesaian proyek ini sangat berhasil, kedua model yang digunakan dapat menunjukkan kinerjanya dengan sangat baik. Goals dapat di capai melalui visualisasi data yang di gambarkan dan mampu menyelesaikan permasalahan yang ada. Hanya saja untuk pemilihan model dengan kinerja terbaik jatuh pada model dengan Forest Classifier, karena model ini sedikit lebih unggul dalam hal akurasi dan keseimbangan antara precision dan recall di kedua kelas dibandingkan K-Neighbors Classifier. Meskipun kedua model cukup baik dalam mengklasifikasikan data, Forest Classifier memiliki performa yang lebih konsisten dan sedikit lebih baik secara keseluruhan.
"""