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
6. `ever_married` : Menyatakan status pernikahan subjek, apakah sudah pernah menikah atau belum.
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

Dalam tahapan visualisasi data, Saya menggunakan tahapan *Visualisasi Univariate Analysis* sebagai bagian dari tahap visualisasi untuk memahami data sebelum melakukan *pre-processing*. Berikut ini tahapan visualisasi yang saya lakukan : 
- Distribusi jumlah subjek berdasarkan jenis kelamin
  
  <img width="341" alt="Distribusi Gender" src="https://github.com/user-attachments/assets/de156827-d607-4caf-b489-6b9ca2455779">

  Berdasarkan visualisasi bar chart distribusi gender, terlihat bahwa jumlah individu dengan gender Female mencapai sekitar 3000 orang, jauh lebih tinggi dibandingkan dengan Male yang berjumlah sekitar 2000 orang, sementara kategori Other sangat sedikit hingga hampir tidak terlihat. Ketidakseimbangan signifikan ini dapat mempengaruhi analisis dan model machine learning, yang mungkin cenderung bias terhadap prediksi gender dominan.

- Distribusi jumlah subjek berdasarkan status pernikahan
  
  <img width="331" alt="Distribusi Status Pernikahan" src="https://github.com/user-attachments/assets/831ea298-9d4e-4059-b004-76a3086e058d">

  Berdasarkan visualisasi bar chart distribusi status pernikahan, sangat terlihat bahwa distribusi orang yang memiliki status pernikahan (yes) lebih banyak yaitu sebesar kurang lebih 3400 di banding dengan yang belum memiliki status pernikahan (no) sebesar kurang lebih 1700. 

- Distribusi jumlah subjek berdasarkan jenis pekerjaan
  
  <img width="335" alt="Distribusi Jenis Pekerjaan" src="https://github.com/user-attachments/assets/1dc5f634-7ae9-4d6d-9d5c-6dd634c554d3">
  
  Dari grafik visualisasi di atas, terlihat bahwa mayoritas individu bekerja di sektor swasta ("Private") dengan jumlah yang signifikan lebih tinggi dibandingkan jenis pekerjaan lainnya, mencapai hampir 3000 orang. Pekerjaan wiraswasta ("Self-employed"), anak-anak ("children"), dan pegawai pemerintah ("Govt_job") memiliki jumlah yang hampir seimbang, masing-masing sekitar 500 hingga 600 orang. Sementara itu, jumlah individu yang tidak pernah bekerja ("Never_worked") sangat kecil, hampir tidak terlihat dalam grafik. Hal ini menunjukkan bahwa pekerjaan di sektor swasta mendominasi distribusi pekerjaan.

- Distribusi subjek berdasarkan tempat tinggal, apakah mereka tinggal di area urban atau rural ?
  
  <img width="332" alt="Distribusi Jenis Tempat Tingal" src="https://github.com/user-attachments/assets/99b65b85-54a9-45be-bd47-c306d56d7a11">
  
  Berdasarkan grafik di atas, distribusi penduduk antara wilayah perkotaan ("Urban") dan pedesaan ("Rural") tampak hampir seimbang. Jumlah penduduk di wilayah urban sedikit lebih tinggi, yaitu lebih dari 2500 orang, dibandingkan dengan penduduk di wilayah rural yang berada di angka yang hampir sama. Grafik ini menunjukkan bahwa tidak ada perbedaan yang signifikan antara jumlah penduduk yang tinggal di kedua jenis tempat tinggal ini.

- Distribusi dari fitur numerik dalam dataset
  
  <img width="458" alt="Distribusi Fitur Numerik" src="https://github.com/user-attachments/assets/e5dbcda9-b921-42e7-8029-4488502bcc9c">

  Histogram menunjukkan distribusi beberapa fitur numerik dalam dataset. Sebagian besar individu tidak memiliki hipertensi atau penyakit jantung, seperti yang ditunjukkan oleh kemiringan yang signifikan terhadap 0 pada plot tersebut. Distribusi usia cukup seimbang, dengan frekuensi yang lebih tinggi pada kelompok usia yang lebih tua, terutama antara 50 hingga 80 tahun. Kadar glukosa rata-rata memiliki kemiringan ke kanan, dengan sebagian besar individu memiliki kadar antara 70 dan 120, sementara beberapa menunjukkan kadar yang lebih tinggi. Distribusi BMI menunjukkan konsentrasi antara 20 dan 40, yang mengindikasikan sebagian besar individu memiliki nilai BMI sedang. Terakhir, distribusi stroke sangat miring, menunjukkan bahwa sebagian besar individu belum pernah mengalami stroke.

Translated with DeepL.com (free version)
  
- Visualisasi data hipertensi yang berkaitan dengan stroke
  
  <img width="451" alt="Distribusi Hipertemsi Berdasarkan Status Stroke Pasien" src="https://github.com/user-attachments/assets/0ab76b72-ac9a-4017-aad6-8413dd45c85f">

  Dari grafik visualisasi di atas menunjukkan distribusi hipertensi pada pasien berdasarkan status stroke. Sebagian besar pasien yang tidak memiliki hipertensi (hipertensi = 0) juga tidak mengalami stroke, dengan jumlah lebih dari 4000 orang. Sementara itu, pasien yang memiliki hipertensi (hipertensi = 1) memiliki proporsi yang lebih kecil, namun tetap terdapat kasus stroke pada kedua kelompok, baik yang memiliki hipertensi maupun yang tidak. Meskipun hipertensi berkaitan dengan stroke, grafik ini menunjukkan bahwa sebagian besar pasien tanpa hipertensi juga tidak mengalami stroke.
  
- Hubungan usia dengan risiko terjadinya stroke
  
  <img width="449" alt="Distribusi Usia Berdasarkan Stroke dan Hipertensi" src="https://github.com/user-attachments/assets/77a15449-88ff-4bc2-a922-ae372ce89f23">

  Visualisasi data di atas membandingkan distribusi usia berdasarkan kejadian stroke (0 = Tidak, 1 = Ya) dan status hipertensi (biru = tanpa hipertensi, oranye = dengan hipertensi). Hal ini menunjukkan bahwa individu dengan hipertensi cenderung lebih tua pada kategori stroke dan non-stroke. Khususnya, usia rata-rata individu yang mengalami stroke lebih tinggi dibandingkan dengan mereka yang tidak mengalami stroke, terlepas dari status hipertensinya. Selain itu, ada beberapa pencilan pada kelompok usia yang lebih muda untuk orang yang mengalami stroke tetapi tidak memiliki hipertensi, menunjukkan bahwa meskipun usia merupakan faktor, individu yang lebih muda juga dapat mengalami stroke.

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
1. Random Forest, dalam mengimplementasikan algoritma ini, saya menggunakan method *RandomForestClassifier* dari sklearn.ensemble dengan argumen n_estimators=30 dan max_features=3. dan dihasilkan akurasi test score sebesar 0,97 dan confusion matrix score sebesar 0,98.

   Cara kerja random forest :
   - Persiapan data (Scaling) : Data input (`X_train_scaled`, `X_test_scaled`) distandarisasi atau dinormalisasi. Ini membantu memastikan bahwa semua fitur berada dalam skala yang sama, yang dapat meningkatkan kinerja model.
   - Inisialisasi model : Sebuah instance dari `RandomForestClassifier` dibuat dengan parameter yang ditentukan, seperti jumlah pohon (`n_estimators`) dan jumlah fitur yang digunakan dalam setiap pemisahan (`max_features`).
   - Saat model dilatih (`fit`), untuk setiap pohon keputusan yang akan dibangun, algoritma mengambil sampel acak dari data pelatihan dengan pengembalian (`bootstrap sampling`). Ini berarti bahwa beberapa contoh mungkin terpilih lebih dari sekali, sementara yang lain mungkin tidak terpilih sama sekali.
   - Pemisahan Fitur: Hanya fitur acak (berdasarkan `max_feature`s) yang dipertimbangkan saat membangun pohon. Ini meningkatkan variasi antar pohon.
   - Menggunakan metode `score` untuk menghitung akurasi model pada data pelatihan dan pengujian lalu Menghitung proporsi prediksi yang benar dibandingkan dengan total data dan Ini akan memberikan gambaran tentang seberapa baik model telah belajar dari data.
   - Menggunakan `predict_proba`, model dapat memberikan probabilitas untuk setiap kelas untuk setiap contoh. Ini membantu dalam memahami kepercayaan model terhadap prediksinya.
  
   Kelebihan :
   - Memiliki akurasi yang baik pada banyak dataset.
   - Kombinasi dari banyak pohon membantu mengurangi risiko overfitting.
   - Dapat menentukan fitur mana yang paling berkontribusi terhadap prediksi.
     
   Kekurangan :
   - Sulit untuk diinterpretasikan, tidak seperti pohon keputusan tunggal.
   - Membangun banyak pohon dapat memakan waktu dan sumber daya komputasi.
   
3. K-Nearest Neighbor (KNN), dalam mengimplementasikan algoritma ini, saya menggunakan method *KNeighborsClassifier* dari sklearn.neighbors dengan argumen n_neighbors=2. dan dihasilkan akurasi test score sebesar 0,94 dan confusion matrix score sebesar 0,97.

   Cara Kerja KNN :

   Kelebihan :

   Kekurangan :
   

## Evaluation
Dari 2 cara pemodelan yang diguanakan yaitu random forest dan knn, evaluasi yang dapat di ambil dari kedua model tersebut adalah : 
1. **Random Forest**  Model ini bagus dan menghasilkan : 
    - `Akurasi` : 99%
    - `Precision 0` : 100%
    - `Recall 0` : 98%
    - `Precision 1` : 98%
    - `Recall 1` : 100%
    - `F1-Score` : 0.99

<br>

   <img width="360" alt="Random Forest" src="https://github.com/user-attachments/assets/bc4d00a8-7c5a-4ac6-9b61-0f5b6f87d99e">


2. **K-Nearest Neighbor (KNN)**  Model ini bagus dan menghasilkan : 
    - `Akurasi` : 98%
    - `Precision 0` : 100%
    - `Recall 0` : 95%
    - `Precision 1` : 96%
    - `Recall 1` : 100%
    - `F1-Score` : 0.98

<br>

   <img width="359" alt="KNN" src="https://github.com/user-attachments/assets/ef184169-f6d0-4cb1-8979-71d8dea73a1e">
   

<br>


**Perbandingan akurasi dari kedua model Machine Learning :**

<img width="393" alt="Perbandingan Akurasi" src="https://github.com/user-attachments/assets/da79e748-0aeb-41c3-995c-f0dfc6c79418">

1. **Model dengan akurasi terbaik :** Berdasarkan evaluasi performa diatas, Forest Classifier terbukti memiliki akurasi terbaik sebesar 99%. Model ini unggul dalam hal akurasi dan keseimbangan antara precision dan recall, terutama dalam membedakan antara pasien yang berisiko terkena stroke dan yang tidak, dibandingkan dengan K-Neighbors Classifier.
2. **Pembuatan Model untuk Memprediksi Stroke:** Dalam proyek ini, langkah-langkah pembuatan model melibatkan pemrosesan data medis pasien, termasuk faktor-faktor seperti usia, hipertensi, dan kondisi kesehatan lainnya. Dataset digunakan untuk melatih beberapa algoritma machine learning, di antaranya K-Neighbors Classifier dan Forest Classifier. Model Forest Classifier akhirnya dipilih sebagai model terbaik karena memiliki kemampuan generalisasi yang lebih baik dan performa yang konsisten dalam memprediksi risiko stroke.
3. **Peran Hipertensi dalam Mendiagnosis Risiko Stroke:** Hipertensi merupakan faktor risiko utama untuk stroke karena tekanan darah tinggi dapat merusak pembuluh darah di otak. Dalam analisis ini, model machine learning mampu menggunakan data hipertensi untuk membantu memperkirakan kemungkinan pasien terkena stroke. Faktor hipertensi berperan penting karena berkontribusi dalam meningkatkan akurasi prediksi risiko stroke, dengan pasien yang memiliki hipertensi memiliki risiko lebih tinggi dibandingkan yang tidak.

Secara keseluruhan, proyek ini berhasil memecahkan masalah dengan menemukan model prediksi yang andal untuk risiko stroke, serta mengidentifikasi hipertensi sebagai faktor kunci dalam diagnosis.

## Referensi
[1]. (https://p2ptm.kemkes.go.id/infographic-p2ptm/stroke/apa-itu-stroke) P2PTM Kemenkes RI. -*Apa itu Stroke ?*. Kemenkes. https://p2ptm.kemkes.go.id/infographic-p2ptm/stroke/apa-itu-stroke

[2]. (https://ayosehat.kemkes.go.id/kenali-stroke-dan-penyebabnya) Direktorat Promosi Kesehatan dan Pemberdayaan Masyarakat. -*Kenali Stroke dan Penyebabnya*. Ayo Sehat. https://ayosehat.kemkes.go.id/kenali-stroke-dan-penyebabnya

[3]. (https://nasional.kompas.com/read/2024/04/24/11450061/singgung-persoalan-kesehatan-jokowi-kematian-akibat-stroke-capai-330000) Dian Erika Nugraheny, Ihsanuddin. -*Singgung Persoalan Kesehatan, Jokowi: Kematian akibat Stroke Capai 330.000*. Kompas. https://nasional.kompas.com/read/2024/04/24/11450061/singgung-persoalan-kesehatan-jokowi-kematian-akibat-stroke-capai-330000

[4]. (https://www.halodoc.com/artikel/apa-saja-penyebab-stroke-ini-jawabannya?srsltid=AfmBOopLcmWbwnXA32MOvI5OQjgDdjuYil6inEbWKESmFQlkDuyveEpE
) dr. Verury Verona Handayani. -*Apa Saja Penyebab Stroke? Ini Jawabannya*. Halodoc. https://www.halodoc.com/artikel/apa-saja-penyebab-stroke-ini-jawabannya?srsltid=AfmBOopLcmWbwnXA32MOvI5OQjgDdjuYil6inEbWKESmFQlkDuyveEpE

[5]. (https://deepai.org/machine-learning-glossary-and-terms/random-forest) Wood, T. -*What is a Random Forest?*. DeepAI. https://deepai.org/machine-learning-glossary-and-terms/random-forest

[6]. (https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e?gi=13449fe30a85) Subramanian, D. (2019). *A Simple Introduction to K-Nearest Neighbors Algorithm*. Towards Data Science. https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e?gi=13449fe30a85
