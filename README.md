# Proyek Pertama Seoul Bike Rental Demand Prediction

#### Disusun oleh : Michael David Baskoro Krisnanto

Ini adalah proyek pertama predictive analytics untuk memenuhi submission dicoding. Proyek ini membangun model machine learning yang dapat memprediksi bike rental demand.

## Domain Proyek

### Latar Belakang

Bike sharing adalah layanan penyewaan sepeda yang biasanya disediakan oleh perusahaan swasta atau pemerintah untuk digunakan oleh masyarakat umum. Layanan ini biasanya dapat diakses melalui aplikasi smartphone atau kios penyewaan yang tersebar di seluruh kota. Bike sharing menyediakan alternatif transportasi yang ramah lingkungan dan ekonomis, serta dapat meningkatkan mobilitas di daerah-daerah perkotaan yang padat.

Bike sharing telah menjadi semakin populer di seluruh dunia karena berbagai alasan. Pertama, bike sharing dapat memberikan solusi alternatif untuk transportasi yang lebih berkelanjutan, seperti mengurangi emisi karbon dan mengurangi kemacetan lalu lintas. Kedua, bike sharing dapat membantu masyarakat yang tidak memiliki akses ke kendaraan pribadi atau transportasi umum, serta dapat meningkatkan kualitas hidup dan kesehatan melalui olahraga dan aktivitas fisik. Ketiga, bike sharing dapat menjadi alternatif transportasi yang lebih ekonomis, terutama untuk perjalanan pendek di dalam kota.

Sebagai respons terhadap meningkatnya popularitas bike sharing, banyak kota di seluruh dunia telah memperkenalkan program bike sharing yang disubsidi oleh pemerintah atau disediakan oleh perusahaan swasta. Namun, untuk menjamin keberhasilan layanan bike sharing, penting untuk memahami kebiasaan dan preferensi pengguna bike sharing di kota tertentu, serta faktor-faktor yang memengaruhi permintaan bike sharing. Oleh karena itu, dataset seperti Seoul Bike Sharing Demand sangat berharga dalam membantu pengambil keputusan untuk meningkatkan layanan bike sharing dan mobilitas yang berkelanjutan di daerah perkotaan.

Dalam mencapai hal tersebut, maka dilakukan penelitian untuk memprediksi demand sewa sepeda menggunakan model machine learning.  Prediksi ini nantinya dijadikan acuan bagi perusahaan dalam meningkatkan layanan bike sharing yang berkelanjutan.

## Business Understanding

Pemerintah dan perusahaan penyedia bike sharing memerlukan informasi yang akurat tentang kebiasaan dan preferensi pengguna bike sharing di Seoul, serta faktor-faktor yang memengaruhi permintaan bike sharing. Informasi ini diperlukan untuk meningkatkan kualitas layanan bike sharing serta meningkatkan mobilitas yang berkelanjutan di daerah perkotaan.

### Problem Statement

1. Menganalisis data untuk memahami kebiasaan dan preferensi pengguna bike sharing di Seoul, serta fitur - fitur yang memengaruhi permintaan bike sharing.
2. Mengembangkan model prediktif untuk memprediksi permintaan bike sharing berdasarkan variabel-variabel tertentu, seperti waktu, kondisi cuaca, musim, dan hari libur.

### Goals

1. Mengetahui fitur yang paling berpengaruh pada demand bike sharing.
2. Membuat model machine learning yang dapat memprediksi permintaan bike sharing seakurat mungkin berdasarkan karakteristik tertentu.

### Solution Statement

1. Menganalisis data dengan melakukan univariate analysis dan multivariate analysis. Memahami data juga dapat dilakukan dengan visualisasi. Memahami data dapat membantu untuk mengetahui kolerasi antar fitur dan mendeteksi outlier.
2. Membangun model regresi yang dapat memprediksi bilangan kontinu. ALgoritma yang dipakai dalam proyek ini adalah K-Nearest Neighbour, Random Forest, dan GradientBoost.

## Data Understanding
Data bike sharing di kota Seoul, Korea Selatan. Dataset ini mencakup informasi tentang penggunaan sepeda umum di Seoul dari tahun 2017 hingga 2018, seperti jumlah sepeda yang dipinjam, waktu pengambilan dan pengembalian sepeda, cuaca, dan data lainnya yang akan dijelaskan lebih lanjut.

Dataset yang digunakan dalam proyek ini merupakan datapat diunduh di [UC Irvine Machine Learning Repository : Seoul Bike Sharing Demand
Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv).

Berikut informasi pada dataset :

+ Dataset memiliki format CSV (Comma-Seperated Values).
+ Dataset memiliki 8760 sample dengan 14 fitur.
+ Dataset memiliki 4 fitur bertipe int64, 6 fitur bertipe float64 dan 4 fitur bertipe object.
+ Tidak ada missing value dalam dataset.

### Variable - variable pada dataset

+ date: Tanggal sewa sepeda.
+ rented_bike_count: Jumlah total sepeda yang dirental.
+ hour: Jam dalam sehari.
+ temperature: Suhu cuaca dalam derajat celcius.
+ humidity: Kelembapan dalam %.
+ wind_speed: Kecepatan angin dalam m/s.
+ visibility: Jarak pandang atmosfer dalam jarak 10m.
+ dew_point_temperature: Titik embun suhu dalam derajat celcius.
+ solar_radiation: Menunjukkan cahaya dan energi yang berasal dari matahari dalam MJ/m2 .
+ rainfall: Curah hujan dalam milimeter.
+ snowfall: Curah salju dalam milimeter.
+ seasons: Musim Gugur, Musim Semi, Musim Panas, Musim Dingin.
+ holiday: Apakah hari libur
+ functioning_day: Apakah hari tersebut bukan akhir pekan atau libur

### Univariate Analysis

Univariate Analysis adalah proses menganalisis setiap fitur secara terpisah.

#### Analisis sebaran pada setiap fitur numerik

![numerical_histograms](https://user-images.githubusercontent.com/20243946/222890135-28e02ed4-581a-443c-92cb-58515e0d7ac8.png)
Berikut analisis dari grafik di atas :

+  Mayoritas rented_bike_count terpusat pada rentang 0 - 500 dengan penurunan tajam dalam frekuensi untuk nilai yang lebih tinggi
+ Temperature memiliki persebaran yang cukup merata.
+ Visibility memiliki left-skewed distribution.
+ Solar Radiation, Rainfall, dan Snowfall memiliki right-skewed distribution.
+ Perlu dilakukan normalisasi distribusi

#### Analisis jumlah nilai unique pada setiap fitur kategorik

![categorical_features_unique_values](https://user-images.githubusercontent.com/20243946/222890299-a27bbd87-c867-463c-93f6-571265ccad18.png)

+ Setiap nilai jam 0 sampai 23 lengkap berdasarkan jumlah sampel nya yaitu 365 berarti setiap jam pada 1 hari tidak ada yang hilang .
+ Persebaran jumlah sampel setiap musim merata sedangkan jumlah sampel pada holiday dan functioning_day kurang merata.


### Multivariate Analysis

Multivariate Analysis menunjukkan hubungan antara dua atau lebih fitur dalam data.

#### Analisis fitur numerik
 ![numeric_matrix_correlations](https://user-images.githubusercontent.com/20243946/222915428-aa54842a-0afe-446c-a6b3-9b84756a30a4.png)


+ korelasi matrix menunjukkan korelasi positif humidity dan dew_point_temperature memiliki nilai 0.59 yang cukup besar jika dibandingkan dengan lainnya.
+ fitur dew_point_temperature akan dibuang karena terkorelasi sangat besar dengan fitur temparture yang menyebabkan multicolinearity
+ temperature dan dew_point_temperature memiliki korelasi positif yang cukup signifikan terhadap rented_bike_count.

#### Analisis fitur kategorik

Analisis ini dilakukan untuk melihat kolerasi antara fitur kategorik dengan fitur target (rented_bike_count).

+ Fitur Hour
  ![hour categorical_features_multivariate](https://user-images.githubusercontent.com/20243946/222890498-e5cd955f-f37b-4c97-9655-afac21bac83f.png)

  Jam 8 pagi dan jam 17 sore merupakan jam dimana jumlah rented_bike_count memuncak jika dibandingkan dengan jam - jam sebelumnya.

+ Fitur Seasons
  ![seasons categorical_features_multivariate](https://user-images.githubusercontent.com/20243946/222890535-9b25317d-017f-4ac2-ac81-18f65826fb8c.png)
  Musim Panas merupakan musim dimana paling banyak terjadi peminjaman bike diikuti musim gugur lalu musim semi.

+ Fitur Holiday
  ![holiday categorical_features_multivariate](https://user-images.githubusercontent.com/20243946/222890541-bf516db2-a6e5-4d3c-bf54-109527ee3c4a.png)
  Peminjaman bike pada hari kerja lebih banyak jika dibandingkan dengan hari libur. 

+ Fitur Functioning Day
  ![functioning_day categorical_features_multivariate](https://user-images.githubusercontent.com/20243946/222890526-e227b1cf-3c7d-41ad-a6ee-3c435fe458d3.png)
  Peminjaman sepada hanya terjadi pada functional_day

![day-month](https://user-images.githubusercontent.com/20243946/222920065-6c827652-b5f0-41fd-b8b1-1f44306804be.png)
fiture date dipecah menjadi fitur year, month and day pada code cell dibawah, dimana:
+ Terjadi kenaikan pada peminjaman mulai bulan 2 sampai bulan 7 pada puncaknya dan mengalami penurunan mulai bulan 10 sampai 12
+ Jika melihat fitur daily,  peminjaman pada hari kerja lebih signifikan




## Data preparation

+ One Hot Encoding
  One hot encoding adalah teknik mengubah data kategorik menjadi data numerik dimana setiap kategori menjadi kolom baru dengan nilai 0 atau 1. Fitur yang akan diubah menjadi numerik pada proyek ini adalah Hours, Month, Day, Year, Seasons, Holiday.
  
+ Train Test Split
  Train test split aja proses membagi data menjadi 80% data latih  dan 20% data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 7,561 dibagi menjadi 6048 untuk data latih dan 1513 untuk data uji.
  
+ Normalization
  Teknik normalisasi yang digunakan pada proyek ini adalah QuartileTransform dan Standarisasi mean dan std dengan sklearn.preprocessing.QuantileTransformer dan sklearn.preprocessing.StandardScaler.

## Modeling

+ Algoritma
  Penelitian ini melakukan pemodelan dengan 3 algoritma, yaitu K-Nearest Neighbour, Random Forest, dan GradientBoostingRegressor
  + K-Nearest Neighbour
    K-Nearest Neighbour bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat. Proyek ini menggunakan [sklearn.neighbors.KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_neighbors` = Jumlah k tetangga tedekat.

  + Random Forest
    Algoritma random forest adalah teknik dalam machine learning dengan metode ensemble. Teknik ini beroperasi dengan membangun banyak decision tree pada waktu pelatihan. Proyek ini menggunakan [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_estimators` = Jumlah maksimum decision tree.
    + `max_depth` = Kedalaman maksimum setiap tree.
    + `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.

  + GradientBoostingRegressor
    Gradient Boosting Regressor adalah algoritma machine learning yang dapat digunakan untuk memprediksi nilai numerik pada data dengan cara memperhitungkan kesalahan (error) dari model sebelumnya dan menambahkan model baru pada setiap iterasi dengan tujuan untuk mengurangi kesalahan (error) secara signifikan. Gradient Boosting Regressor menggunakan pohon keputusan (decision tree) sebagai model dasar (base model) pada setiap iterasinya. Model dasar ini dapat diatur jumlahnya dan kedalamannya sesuai dengan kebutuhan. Setelah selesai, model yang dihasilkan dapat digunakan untuk memprediksi nilai numerik pada data yang belum pernah dilihat sebelumnya.
    
    + `n_estimators` = Jumlah maksimum estimator di mana boosting dihentikan.
    + `learning_rate` = Learning rate memperkuat kontribusi setiap regressor.
    + `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.


## Evaluation

Metrik evaluasi yang digunakan pada proyek ini adalah akurasi dan mean squared error (MSE). Akurasi menentukan tingkat kemiripan antara hasil prediksi dengan nilai yang sebenarnya (y_test). Mean squared error (MSE) mengukur error dalam model statistik dengan cara menghitung rata-rata error dari kuadrat hasil aktual dikurang hasil prediksi. Berikut formula MSE :
![formula_mse](https://user-images.githubusercontent.com/20243946/222915059-33b7dfc1-f5f8-437f-90ef-202c554b8fe7.jpeg)

Berikut hasil evaluasi pada proyek ini :

+ Akurasi
  | model    | accuracy |
  |----------|----------|
  | knn      | 0.791616534943286 |
  | rf       | 0.8940512865363659 |
  | boosting | 0.8880770626943283 |
  

+ Mean Squared Error (MSE)

  ![MSE](https://user-images.githubusercontent.com/20243946/222914980-860432aa-5c60-469c-a3d6-1898a0862060.png)  

Dari hasil evaluasi dapat dilihat bahwa model dengan algoritma RandomForsst memiliki akurasi lebih tinggi tinggi dan tingkat error lebih kecil dibandingkan algoritma lainnya dalam proyek ini.
