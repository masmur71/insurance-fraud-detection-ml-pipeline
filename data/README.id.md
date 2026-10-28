# Dataset Fraud Asuransi Kendaraan (Sintetis)

**Bahasa:** [English](README.md) | Indonesia

Folder ini berisi **notebook generator dan dokumentasi** untuk dataset sintetis fraud asuransi kendaraan Indonesia. Dataset hasil generate tidak dipublikasikan bersama repositori: jalankan generator untuk membuatnya secara lokal dengan seed `20261028`.

> **Penting:** seluruh data hasil generate bersifat **sintetis**. Model hanya bisa memulihkan aturan yang ditanam oleh generator, bukan perilaku fraud di dunia nyata. Setiap metrik yang dikutip harus diberi label **"synthetic data"**.

## Daftar isi

- [Struktur repositori](#struktur-repositori)
- [Artefak yang dihasilkan](#artefak-yang-dihasilkan)
- [Konfigurasi](#konfigurasi)
- [Meregenerasi dataset](#meregenerasi-dataset)
- [Skema tabel](#skema-tabel)
- [Mekanisme fraud yang ditanam](#mekanisme-fraud-yang-ditanam)
- [Jebakan leakage](#jebakan-leakage)
- [Memuat data](#memuat-data)
- [Notebook EDA](#notebook-eda)
- [Batasan dan etika](#batasan-dan-etika)
- [Checklist sebelum modeling](#checklist-sebelum-modeling)
- [Reproducibility](#reproducibility)
- [Pemeliharaan dokumentasi](#pemeliharaan-dokumentasi)

## Struktur repositori

Hanya notebook dan dua file README yang dipublikasikan:

```text
data/
├── README.md                 dokumentasi bahasa Inggris (default)
├── README.id.md              dokumentasi bahasa Indonesia
└── notebooks/
    ├── vehicle_insurance_fraud_synth.ipynb   generator + EDA inline
    └── eda_vehicle_insurance_fraud.ipynb     EDA read-only
```

File Parquet dan JSON hasil generate **bukan bagian dari repositori yang dipublikasikan**. File tersebut ditulis ke `data/` saat generator dijalankan dan sebaiknya tetap lokal.

## Artefak yang dihasilkan

Menjalankan generator akan membuat file berikut di samping README ini:

| File | Fungsi |
|---|---|
| `customers.parquet` | satu baris per customer |
| `policies.parquet` | satu baris per polis |
| `claims.parquet` | tabel level klaim; duplikat `claim_id` sengaja disuntikkan |
| `workshops.parquet` | tabel referensi workshop |
| `claims_truth.parquet` | label laten, flag deteksi, probabilitas fraud, dan ring era; hanya untuk analisis |
| `claims_features.parquet` | tabel analisis setelah deduplikasi, berisi atribut polis/customer dan fitur turunan |
| `generator_config.json` | parameter generator yang terkunci dan metadata generasi |
| `manifest.json` | seed, versi library, row count, ukuran file, dan metadata fitur |

## Konfigurasi

| Pengaturan | Nilai |
|---|---|
| Seed | `20261028` |
| Periode klaim terkonfigurasi | 2021-01-01 s/d 2025-12-31 |
| Kebijakan split | train 2021-2023, validation 2024 H1, test 2024 H2-2025 |
| Label noise | 15% klaim fraud diberi label 0 |
| Mekanisme fraud | model logistik dengan noise, missingness MNAR, data kotor, dan drift 2025 |

Semua nilai di atas berasal dari konfigurasi generator yang terkunci. Row count dan fraud rate aktual dicatat di `manifest.json` setelah proses generate.

## Meregenerasi dataset

1. Buka [`notebooks/vehicle_insurance_fraud_synth.ipynb`](notebooks/vehicle_insurance_fraud_synth.ipynb).
2. Jalankan **Run all**. Notebook menulis ke folder `data/` yang sama (di Colab: `/content/data`).
3. Notebook membuat tabel mentah, truth, `claims_features.parquet`, dan `generator_config.json`, lalu memperbarui `manifest.json`.

Jalankan generator sebelum membuka notebook EDA. EDA membutuhkan file Parquet hasil generate.

Generator dan EDA inline masih berada di notebook yang sama. Untuk eksplorasi sehari-hari, gunakan [`eda_vehicle_insurance_fraud.ipynb`](notebooks/eda_vehicle_insurance_fraud.ipynb) yang **read-only**: tidak menulis atau mengubah file apa pun.

> Menjalankan ulang generator akan **menimpa** file hasil generate. Lakukan hanya jika memang ingin meregenerasi dataset.

## Skema tabel

### `customers.parquet` — 1 baris = 1 customer

| Kolom | Keterangan |
|---|---|
| `customer_id` | primary key, format `C000001` |
| `age` | usia 18-75 |
| `occupation_group` | employee, self_employed, entrepreneur, student, professional, retired, civil_servant, other |
| `province` | label provinsi sintetis |
| `customer_since` | tanggal mulai menjadi customer |
| `customer_tenure_months` | tenure saat snapshot |
| `n_policies_total` | 1 atau 2 polis |

### `policies.parquet` — 1 baris = 1 polis

| Kolom | Keterangan |
|---|---|
| `policy_id` | primary key, format `P0000001` |
| `customer_id` | foreign key ke `customers` |
| `policy_start_date`, `policy_end_date` | periode polis 1 tahun |
| `coverage_type` | `Comprehensive` atau `TLO` |
| `vehicle_type` | motorcycle, private_car, commercial_vehicle, truck |
| `vehicle_age_years` | umur kendaraan |
| `vehicle_value` | nilai kendaraan (IDR) |
| `sum_insured` | nilai pertanggungan (IDR) |
| `annual_premium` | premi tahunan (IDR) |
| `sales_channel` | agent, online, broker |
| `payment_frequency` | annual, semiannual, quarterly, monthly |

### `claims.parquet` — 1 baris = 1 klaim

| Kolom | Keterangan |
|---|---|
| `claim_id` | ID klaim; duplikat sengaja disuntikkan, sehingga tidak unik |
| `policy_id`, `customer_id` | foreign key |
| `incident_date`, `report_date` | tanggal kejadian dan pelaporan |
| `incident_type` | collision, theft, flood, vandalism, fire |
| `incident_hour` | jam 0-23; ada nilai hilang (MNAR) |
| `incident_province` | provinsi kejadian |
| `claim_amount` | nominal klaim (IDR); ada outlier ekstrem |
| `police_report_filed` | 0/1; ada nilai hilang (MNAR) |
| `witness_count` | jumlah saksi; ada nilai hilang (MNAR) |
| `workshop_id` | workshop perbaikan |
| `is_authorized_workshop` | 0/1 |
| `prior_claims_12m` | jumlah klaim 12 bulan sebelumnya |
| `prior_claims_total` | total klaim sebelumnya |
| `is_fraud` | **label teramati** yang dipakai untuk modeling |
| `claim_status` | approved, rejected, under_review — **leakage** |
| `payout_amount` | nominal payout — **leakage** |
| `siu_investigation_flag` | flag investigasi SIU — **leakage** |
| `days_to_settlement` | durasi penyelesaian — **leakage** |

### `workshops.parquet` — 1 baris = 1 workshop

| Kolom | Keterangan |
|---|---|
| `workshop_id` | primary key, format `W0001` |
| `workshop_name`, `province`, `city` | atribut lokasi |
| `is_authorized_workshop` | 0/1 |
| `opened_date` | tanggal buka; sejumlah workshop dibuka pada 2025 untuk cerita drift |

### `claims_truth.parquet` — analisis saja, jangan dipakai sebagai fitur

| Kolom | Keterangan |
|---|---|
| `claim_id` | primary key |
| `true_fraud` | label laten sebelum label noise |
| `detected` | apakah fraud terdeteksi |
| `fraud_probability` | probabilitas fraud dari model logit generator |
| `ring_era` | `none`, `old`, atau `new` (ring workshop) |

### `claims_features.parquet` — tabel analisis

Gabungan `claims` (setelah deduplikasi), atribut polis, dan atribut customer. Berisi kolom mentah, versi kategori yang sudah dinormalisasi (`*_clean`), fitur turunan, `split`, dan **kolom leakage**.

Fitur turunan utama:

| Kolom | Rumus / arti |
|---|---|
| `report_delay_days` | `report_date - incident_date` |
| `days_since_policy_start` | `incident_date - policy_start_date` |
| `claim_to_sum_insured_ratio` | `claim_amount / sum_insured` |
| `claim_to_premium_ratio` | `claim_amount / annual_premium` |
| `workshop_claim_count_90d` | jumlah klaim workshop yang sama dalam 90 hari |
| `customer_tenure_months` | tenure customer saat polis mulai |
| `split` | train / validation / test berbasis tanggal kejadian |

## Mekanisme fraud yang ditanam

Fraud dihasilkan oleh model logit dengan noise, lalu diberi label noise 15%. Nilai berikut adalah pengali odds: generator menambahkan logaritma naturalnya ke logit. Nilai ini bukan koefisien logit.

| Pemicu | Koefisien |
|---|---:|
| Klaim dalam 60 hari pertama polis | 3,0 |
| Report delay > 14 hari | 2,0 |
| Theft tanpa laporan polisi | 4,0 |
| Rasio klaim / sum insured > 0,8 | 2,5 |
| Prior claims 12 bulan >= 3 | 2,0 |
| Kejadian jam 00-04 | 1,5 |
| Workshop non-authorized | 1,5 |
| Overvaluation | 2,0 |
| Online + customer baru | 1,8 |
| Interaksi klaim awal x nominal besar | 1,5 |

Efek tambahan:

- **Ring workshop:** ring lama dan ring baru yang diperkenalkan pada 2025. Ring lama kuat sebelum 2025 dan melemah pada 2025; ring baru hanya aktif pada 2025.
- **Drift 2025:** efek kanal online dikali 1,55 dan komposisi ring berubah.
- **Missingness MNAR:** `witness_count`, `police_report_filed`, dan `incident_hour` lebih sering hilang pada klaim fraud.
- **Data kotor:** variasi kapitalisasi kategori (3%), report delay negatif (0,3%), outlier nominal klaim (0,5%), dan duplikat (0,2%).

## Jebakan leakage

Empat kolom ini hanya diketahui setelah klaim diproses:

| Kolom | Alasan |
|---|---|
| `claim_status` | Hasil keputusan; fraud yang terdeteksi hampir selalu rejected/under_review |
| `payout_amount` | Nol untuk fraud yang ditolak; payout adalah konsekuensi keputusan |
| `siu_investigation_flag` | Hasil triage investigasi, bukan input saat klaim dilaporkan |
| `days_to_settlement` | Baru diketahui setelah proses selesai |

Jika dimasukkan, model akan tampak hampir sempurna tetapi tidak berguna untuk klaim baru. `claims_features.parquet` sengaja mempertahankan kolom ini agar demonstrasi bisa direproduksi; **drop sebelum modeling**.

## Memuat data

Jalankan generator terlebih dahulu, lalu muat file hasil generate:

```python
from pathlib import Path
import pandas as pd

DATA = Path("data")

claims = pd.read_parquet(DATA / "claims.parquet")
features = pd.read_parquet(DATA / "claims_features.parquet")

LEAKAGE = ["claim_status", "payout_amount", "siu_investigation_flag", "days_to_settlement"]
TARGET = "is_fraud"

model_frame = (
    features
    .drop_duplicates("claim_id")
    .drop(columns=LEAKAGE)
)

train = model_frame[model_frame["split"] == "train"]
valid = model_frame[model_frame["split"] == "validation"]
test = model_frame[model_frame["split"] == "test"]

X_train = train.drop(columns=[TARGET, "claim_id", "split"])
y_train = train[TARGET]
```

`claims_truth.parquet` hanya untuk diagnosis label noise dan error analysis. Jangan pernah di-join ke fitur model.

## Notebook EDA

[`notebooks/eda_vehicle_insurance_fraud.ipynb`](notebooks/eda_vehicle_insurance_fraud.ipynb) berisi 15 bagian:

1. Setup dan resolusi folder data
2. Inventaris file dan skema
3. Integritas relasional
4. Target dan label noise
5. Missingness MNAR
6. Data kotor yang disengaja
7. Sinyal univariat (lift + Wilson CI)
8. Kategori dan workshop
9. Jebakan leakage
10. Drift waktu 2025
11. Fairness check (`age`, `province`)
12. Sanity check split
13. Korelasi fitur non-leakage
14. Ringkasan dan checklist
15. Batasan

Notebook ini **tidak menyimpan output** saat dibuat. Setelah dataset di-generate, jalankan **Run all** untuk menghasilkan tabel dan grafik secara lokal. Tidak ada file yang ditulis.

## Batasan dan etika

- Data sintetis; bukan perilaku fraud nyata.
- Frekuensi klaim dan exposure tidak dimodelkan (~1,5 klaim per polis hanya asumsi).
- Premi bukan tarif aktual; nominal IDR hanya terlihat masuk akal.
- Tidak ada narrative klaim, sehingga analisis teks tidak relevan.
- `age` dan `province` tidak menghasilkan fraud. Jika dipakai, jelaskan risiko profiling secara eksplisit.
- Deteksi fraud diasumsikan konstan 85%; investigasi nyata bersifat selektif dan bias.

## Checklist sebelum modeling

1. Drop empat kolom leakage dan semua kolom truth.
2. Deduplikasi `claim_id`.
3. Normalisasi kategori: `strip()` + `lower()`.
4. Tangani report delay negatif dan outlier nominal klaim.
5. Tambahkan indikator missing untuk kolom MNAR.
6. Split berdasarkan waktu; jangan acak.
7. Pilih threshold dari cost matrix, bukan F1 semata.
8. Laporkan PR-AUC, ROC-AUC, calibration, dan confusion matrix per split.
9. Putuskan pemakaian `age` dan `province`; dokumentasikan alasannya.
10. Cantumkan **"synthetic data"** dan seed `20261028` pada setiap metrik.

## Reproducibility

- `generator_config.json` menyimpan seluruh parameter, calibrated intercept, row count, dan fraud rate.
- `manifest.json` menyimpan seed, versi Python/NumPy/pandas, ukuran setiap file Parquet hasil generate, serta metadata `claims_features.parquet` (`feature_rows`, `feature_columns`, `features_exclude_truth`).
- Generator menulis manifest awal setelah tabel mentah, lalu memperbaruinya setelah `claims_features.parquet` dibuat.
- Dengan seed yang sama, output dapat direproduksi; hanya timestamp `generated_at_utc` yang berubah.

## Pemeliharaan dokumentasi

`README.id.md` dan [`README.md`](README.md) dipelihara sebagai pasangan. Saat skema, struktur repositori, atau generator berubah, perbarui kedua file tersebut bersamaan.
