# Laporan Proyek Machine Learning - Firda Aulia Rakhmah

## Domain Proyek

Domain yang saya pilih untuk proyek pertama ini adalah Kesehatan, dengan judul proyek **Model Prediksi Berbasis Machine Learning untuk Diagnosis Risiko Terjadinya Stroke Berdasarkan Analisis Faktor Hipertensi dari Data Pasien**.

- Latar Belakang 
![Stroke](https://github.com/user-attachments/assets/dc26336d-6749-4c00-96a5-bee716f37ca5)


Stroke adalah penyakit pembuluh darah otak. Definisi menurut WHO, Stroke adalah suatu keadaan dimana ditemukan tanda-tanda klinis yang berkembang cepat berupa defisit neurologik fokal dan global, yang dapat memberat dan berlangsung lama selama 24 jam atau lebih dan atau dapat menyebabkan kematian, tanpa adanya penyebab lain yang jelas selain vascular. Stroke terjadi apabila pembuluh darah otak mengalami penyumbatan atau pecah. Akibatnya sebagian otak tidak mendapatkan  pasokan darah yang membawa oksigen yang diperlukan sehingga mengalami kematian sel/jaringan [[1](https://p2ptm.kemkes.go.id/infographic-p2ptm/stroke/apa-itu-stroke)]. 

Stroke merupakan salah satu penyebab utama kematian dan kecacatan di seluruh dunia. Data Institute for Health Metrics and Evaluation (IHME) tahun 2019 menunjukkan stroke sebagai penyebab kematian utama di Indonesia (19,42% dari total kematian). Berdasarkan hasil Riskesdas prevalensi stroke di Indonesia meningkat 56% dari 7 per 1000 penduduk pada tahun 2013, menjadi 10,9 per 1000 penduduk pada tahun 2018 [[2](https://ayosehat.kemkes.go.id/kenali-stroke-dan-penyebabnya)]. Presiden Joko Widodo dalam Rapat Kerja Kesehatan Nasional tahun 2024 mengungkapkan Kematian akibat penyakit tidak menular, paling banyak adalah (disebabkan) stroke, yang Mencapai 330.000-an kematian akibat stroke [[3](https://nasional.kompas.com/read/2024/04/24/11450061/singgung-persoalan-kesehatan-jokowi-kematian-akibat-stroke-capai-330000)] .

Faktor risiko seperti hipertensi (tekanan darah tinggi) telah diketahui sebagai salah satu penyebab utama terjadinya stroke [[4](https://www.halodoc.com/artikel/apa-saja-penyebab-stroke-ini-jawabannya?srsltid=AfmBOopLcmWbwnXA32MOvI5OQjgDdjuYil6inEbWKESmFQlkDuyveEpE)]. Dengan adanya data medis yang semakin berkembang, metode berbasis Machine Learning (ML) dapat memberikan solusi yang efektif untuk menganalisis risiko pasien terkena stroke, khususnya berdasarkan riwayat hipertensi mereka. Memprediksi risiko stroke sejak dini dapat membantu pengambilan keputusan medis yang lebih tepat waktu, dan pada akhirnya menyelamatkan lebih banyak nyawa.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang di atas, berikut ini merupakan permasalahan yang dapat diselesaikan dari proyek ini : 
- Model seperti apa yang memiliki akurasi paling baik ?
- Bagaimana cara membuat model untuk memprediksi penyakit stroke dengan menggunakan machine learning ? 
- Bagaimana faktor hipertensi mampu membantu mendiagnosis risiko terjadinya penyakit stroke ?

### Goals
Tujuan dari proyek ini adalah : 
- Mengetahui variabel yang berpengaruh terhadap prediksi diagnosis penyakit stroke.
- Membangun model prediksi berbasis Machine Learning yang dapat membantu mendiagnosis risiko terjadinya stroke pada pasien dengan mempertimbangkan faktor hipertensi dan faktor risiko lainnya yang relevan.
- Mengetahui perbandingan beberapa algoritma model Machine Learning sehingga ditemukan akurasi yang paling baik.

### Solution statements
Untuk mencapai tujuan tersebut, dalam proyek ini saya membuat 2 model Machine Learning yang berbeda untuk dibandingkan akurasi terbaik, diantaranya adalah menggunakan:
- Random Forest adalah algoritma machine learning yang kuat yang dapat digunakan untuk berbagai tugas termasuk regresi dan klasifikasi. Ini adalah metode ensemble, yang berarti bahwa model random forest terdiri dari banyak decision tree kecil, yang disebut estimator, yang masing-masing menghasilkan prediksi mereka sendiri. Random forest menggabungkan prediksi estimator untuk menghasilkan prediksi yang lebih akurat [[5](https://deepai.org/machine-learning-glossary-and-terms/random-forest)]. 
- K-Nearest Neighbor (KNN) adalah algoritma sederhana yang mengklasifikasikan data atau kasus baru berdasarkan ukuran kesamaan. Hal ini sebagian besar digunakan untuk mengklasifikasikan titik data berdasarkan tetangga terdekatnya sebagai acuan [[6](https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e?gi=13449fe30a85)]. 

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah data hasil prediksi stroke. Data ini dapat diunduh melalui Kaggle. Pada dataset ini terdapat 5110 baris dan 12 kolom, diantaranya:
1. `id` : Nomor unik yang dimiliki oleh setiap subjek dalam dataset.
2. `gender` : Menyatakan jenis kelamin subjek (misalnya, laki-laki atau perempuan).
3. `age` : Usia subjek dalam dataset.
4. `hypertension` : Menunjukkan apakah subjek memiliki riwayat hipertensi. Nilai 0 berarti tidak memiliki hipertensi, sedangkan nilai 1 berarti subjek mengidap hipertensi.
5. `heart_disease` : Menyatakan apakah subjek memiliki penyakit jantung. Nilai 0 berarti tidak memiliki penyakit jantung, sedangkan nilai 1 berarti subjek mengidap penyakit jantung.
6. ` ever_married` : Menyatakan status pernikahan subjek, apakah sudah pernah menikah atau belum.
7. `work_type` : Menggambarkan jenis pekerjaan yang dimiliki oleh subjek.
8. `Residence_type` : Menunjukkan jenis tempat tinggal subjek, apakah tinggal di daerah perkotaan atau pedesaan.
9. `avg_glucose_level` : Menyatakan kadar gula darah rata-rata subjek.
10. `bmi` : Indeks Massa Tubuh (BMI) yang menggambarkan kategori berat badan subjek.
11. `smoking_status` : Menunjukkan kebiasaan merokok subjek, apakah perokok aktif, mantan perokok, atau tidak merokok
12. `stroke` : Variabel target yang menyatakan apakah subjek telah terdiagnosis stroke. Nilai 0 menunjukkan tidak terkena stroke, sedangkan nilai 1 menunjukkan subjek terkena stroke.

<br>

Informasi Dataset:

Jenis | Keterangan
--- | ---
Title | Stroke Prediction Dataset
Source | [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
Maintainer | [Fedesoriano](https://www.kaggle.com/fedesoriano)
License | Data files © Original Authors
Visibility | Public
Tags | Health, Health Conditions, Public Health, Healthcare, Binary Classification
Usability | 10.0

<br> 

Tahapan dalam penyelesaian proyek ini sebelum data diolah pada pre-processing adalah : 
- Meload dataset (read data .csv)
- Melihat jumlah baris dan kolom pada data .csv
- Melihat informasi pada dataset
- Mengecek total missing value tiap kolom
- Mengecek data yang terduplikat
- Mengecek apakah data sudah balance ?


<br>

Saya menggunakan tahapan Visualisasi Univariate Analysis sebagai bagian dari tahap visualisasi data untuk memahami data sebelum melakukan pre-processing. Berikut ini tahapan visualisasi yang saya lakukan : 
- Distribusi jumlah subjek berdasarkan jenis kelamin
  ![Uploading Distribusi Gender.png…]()

- Distribusi jumlah subjek berdasarkan status pernikahan
  ![Uploading Distribusi Status Pernikahan.png…]()

- Distribusi jumlah subjek berdasarkan jenis pekerjaan
  ![Uploading Distribusi Jenis Pekerjaan.png…]()

- Distribusi subjek berdasarkan tempat tinggal, apakah mereka tinggal di area urban atau rural ?
  ![Uploading Distribusi Jenis Tempat Tingal.png…]()

- Distribusi dari fitur numerik dalam dataset
  ![Uploading Distribusi Fitur Numerik.png…]()

- Visualisasi data hipertensi yang berkaitan dengan stroke
  ![Uploading Distribusi Hipertemsi Berdasarkan Status Stroke Pasien.png…]()

- Hubungan usia dengan risiko terjadinya stroke
  ![Uploading Distribusi Usia Berdasarkan Stroke dan Hipertensi.png…]()


## Data Preparation
Berikut adalah teknik yang dilakukan dalam proses data *preparation*:
- **Menghapus kolom yang tidak diperlukan**. Kolom atau variabel yang dihapus adalah id, karena tidak memiliki kepentingan untuk dimasukkan ke dalam model.
- **Penanganan data yang hilang atau *missing values***. Dalam dataset ini, ada sebanyak 201 data kosong pada kolom bmi. Maka diterapkan teknik melakukan imputasi atau nilai pengganti. Nilai pengganti yang digunakan adalah nilai rata-rata *(mean)*.
- **Melakukan *upsample* agar data seimbang**. Dalam dataset ini, ditemukan bahwa data belum seimbang, maka dilakukan upsample agar data menjadi seimbang dan menghasilkan prediksi yang bagus.
- **Mendeteksi *outliers***. Dalam proyek ini saya menggunakan InterQuartile Range untuk mendeteksi outliers.
- **Melakukan *one hot encoding***. Ini dilakukan pada data kategorikal agar datanya berubah menjadi data numerikal.
- **Membagi dataset, dan melakukan scaling dengan *MinMaxScaler***. Teknik ini dilakukan untuk membuat numerical data pada dataset memiliki rentang nilai (scale) yang sama. 

## Modeling
Setelah menyelesaikan data preparation, langkah berikutnya adalah membangun model machine learning. Dalam proyek ini, saya akan membuat dua model, yaitu Random Forest dan K-Nearest Neighbor (KNN).

1. **Random Forest** Untuk implementasi algoritma ini, menggunakan metode *Random Forest Classifier* dari sklearn.ensemble, dengan n_estimators=30 dan max_features=3. Model ini menghasilkan : 
    - `Akurasi` : 99%
    - `Precision 0` : 100%
    - `Recall 0` : 98%
    - `Precision 1` : 98%
    - `Recall 1` : 100%
    - `F1-Score` : 0.99
      
Kelebihan dari random forest adalah ...


2. **K-Nearest Neighbor (KNN)** Untuk mengimplementasikan algoritma ini menggunakan metode *KNeighborsClassifier* dari sklearn.neighbors dengan argumen n_neighbors=2. Model ini menghasilkan : 
    - `Akurasi` : 98%
    - `Precision 0` : 100%
    - `Recall 0` : 95%
    - `Precision 1` : 96%
    - `Recall 1` : 100%
    - `F1-Score` : 0.98

Kelebihan dari K-Nearest Neighbor (KNN) adalah ...

## Evaluation
Secara keseluruhan penyelesaian proyek ini sangat berhasil, kedua model yang digunakan dapat menunjukkan kinerjanya dengan sangat baik. Goals dapat di capai melalui visualisasi data yang di gambarkan dan mampu menyelesaikan permasalahan yang ada. Perbandingan akurasi dari kedua model pun sangat mudah untuk dimengerti, seperti visualisasi data berikut : 



Berdasarkan visualisasi data di atas, untuk pemilihan model dengan kinerja terbaik jatuh pada model dengan Forest Classifier, karena model ini sedikit lebih unggul dalam hal akurasi dan keseimbangan antara precision dan recall di kedua kelas dibandingkan K-Neighbors Classifier. Meskipun kedua model cukup baik dalam mengklasifikasikan data, Forest Classifier memiliki performa yang lebih konsisten dan sedikit lebih baik secara keseluruhan. 


## Referensi
[[1](https://p2ptm.kemkes.go.id/infographic-p2ptm/stroke/apa-itu-stroke)] P2PTM Kemenkes RI. -*Apa itu Stroke ?*. Kemenkes. https://p2ptm.kemkes.go.id/infographic-p2ptm/stroke/apa-itu-stroke

[[2](https://ayosehat.kemkes.go.id/kenali-stroke-dan-penyebabnya)] Direktorat Promosi Kesehatan dan Pemberdayaan Masyarakat. -*Kenali Stroke dan Penyebabnya*. Ayo Sehat. https://ayosehat.kemkes.go.id/kenali-stroke-dan-penyebabnya

[[3](https://nasional.kompas.com/read/2024/04/24/11450061/singgung-persoalan-kesehatan-jokowi-kematian-akibat-stroke-capai-330000)] Dian Erika Nugraheny, Ihsanuddin. -*Singgung Persoalan Kesehatan, Jokowi: Kematian akibat Stroke Capai 330.000*. Kompas. https://nasional.kompas.com/read/2024/04/24/11450061/singgung-persoalan-kesehatan-jokowi-kematian-akibat-stroke-capai-330000

[[4](https://www.halodoc.com/artikel/apa-saja-penyebab-stroke-ini-jawabannya?srsltid=AfmBOopLcmWbwnXA32MOvI5OQjgDdjuYil6inEbWKESmFQlkDuyveEpE
)] dr. Verury Verona Handayani. -*Apa Saja Penyebab Stroke? Ini Jawabannya*. Halodoc. https://www.halodoc.com/artikel/apa-saja-penyebab-stroke-ini-jawabannya?srsltid=AfmBOopLcmWbwnXA32MOvI5OQjgDdjuYil6inEbWKESmFQlkDuyveEpE

[[5](https://deepai.org/machine-learning-glossary-and-terms/random-forest)] Wood, T. -*What is a Random Forest?*. DeepAI. https://deepai.org/machine-learning-glossary-and-terms/random-forest

[[6](https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e?gi=13449fe30a85)] Subramanian, D. (2019). *A Simple Introduction to K-Nearest Neighbors Algorithm*. Towards Data Science. https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e?gi=13449fe30a85
