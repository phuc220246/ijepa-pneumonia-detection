# 🫁 I-JEPA Pneumonia Detection

> **Self-Supervised Pneumonia Detection from Chest X-Rays via I-JEPA**  
> Kiến trúc lai I-JEPA + Random Masking · ViT-Small/16 · NIH → RSNA · AUC = 0.8297

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20T4-yellow)](https://kaggle.com)

---

## 📋 Mục lục

| # | Nội dung |
|---|---|
| 1 | [Tóm tắt đề tài](#-tóm-tắt-đề-tài) |
| 2 | [Kết quả chính](#-kết-quả-chính) |
| 3 | [Cấu trúc thư mục](#-cấu-trúc-thư-mục) |
| 4 | [Cài đặt môi trường](#-cài-đặt-môi-trường) |
| 5 | [Dữ liệu](#-dữ-liệu) |
| 6 | [Hướng dẫn chạy](#-hướng-dẫn-chạy) |
| 7 | [Notebooks](#-notebooks) |
| 8 | [Tài liệu nghiên cứu](#-tài-liệu-nghiên-cứu) |
| 9 | [Trích dẫn](#-trích-dẫn) |

---

## 🔬 Tóm tắt đề tài

Nghiên cứu này đề xuất ứng dụng kiến trúc **I-JEPA (Image-based Joint-Embedding Predictive Architecture)** với chiến lược **random patch masking** để phát hiện viêm phổi từ ảnh X-quang ngực trong điều kiện nhãn dữ liệu hạn chế.

### Vấn đề
- Thiếu bác sĩ X-quang chuyên khoa tại tuyến cơ sở Việt Nam
- Gán nhãn ảnh y tế tốn kém: 5–15 phút/ảnh × hàng nghìn ảnh
- Mô hình học có giám sát cần nhiều nhãn mới hoạt động tốt

### Giải pháp
- **Phase 1 — Pretraining (không nhãn):** I-JEPA học cấu trúc giải phẫu lồng ngực từ 50.000 ảnh NIH không cần nhãn
- **Phase 2 — Fine-tuning (ít nhãn):** Chỉ cần ~200 ảnh có nhãn để đạt AUC > 0.80

### Đóng góp khoa học
| Đóng góp | Chi tiết |
|---|---|
| Kiến trúc mới | I-JEPA + Random Masking cho ảnh X-quang (thay block masking gốc) |
| Label efficiency | Phân tích hệ thống 6 mức nhãn: 1% → 100% |
| Ổn định vượt trội | Variance thấp hơn ResNet50 **143 lần** tại 1% nhãn |
| CLAHE ablation | Phát hiện domain shift nhân tạo khi dùng CLAHE với SSL |

---

## 📊 Kết quả chính

### So sánh tổng hợp (100% nhãn, threshold tối ưu F1)

| Mô hình | AUC | F1 | Recall | Specificity | Precision |
|---|---|---|---|---|---|
| ResNet50 ImageNet | **0.8862** | **0.6508** | 0.6807 | **0.8804** | **0.6233** |
| ViT-Small ImageNet | 0.8797 | 0.6379 | 0.6142 | 0.9094 | 0.6635 |
| I-JEPA Linear Probe | 0.7737 | 0.5344 | 0.7062 | 0.7275 | 0.4298 |
| I-JEPA Partial FT v1 | 0.8003 | 0.5630 | 0.6907 | 0.7781 | 0.4752 |
| I-JEPA Full FT v1 | 0.8297 | 0.5921 | **0.7738** | 0.7333 | 0.5188 |

### Label Efficiency — I-JEPA vs ResNet50 (1% nhãn = 187 ảnh)

| Mô hình | AUC (mean) | Std | Ghi chú |
|---|---|---|---|
| I-JEPA Full FT v2 | **0.8041** | **±0.0002** | Ổn định |
| ResNet50 ImageNet | 0.7821 | ±0.0286 | Dao động lớn |
| ViT-Small ImageNet | 0.7906 | ±0.0250 | |

> **Phát hiện chính:** I-JEPA ổn định hơn ResNet50 **143 lần** tại 1% nhãn

### Threshold Tuning — I-JEPA Full FT v1

| Chiến lược | Threshold | Recall | Specificity | F1 |
|---|---|---|---|---|
| Mặc định | 0.50 | 0.7738 | 0.7333 | 0.5752 |
| Best F1 | 0.60 | 0.6896 | 0.8139 | **0.5921** |
| Recall ≥ 0.85 *(Đề xuất tuyến cơ sở)* | 0.39 | **0.8559** | 0.6404 | 0.5536 |
| Recall ≥ 0.90 | 0.28 | 0.9024 | 0.5372 | 0.5167 |

---

## 📁 Cấu trúc thư mục

```
ijepa-pneumonia-detection/
│
├── 📄 README.md                         ← Bạn đang đọc file này
├── 📄 requirements.txt                  ← Thư viện cần cài
├── 📄 LICENSE
│
├── 📂 notebooks/                        ← Kaggle notebooks (.ipynb)
│   ├── NB01_data_preparation.ipynb      ← Chuẩn bị dữ liệu + split
│   ├── NB02_baselines.ipynb             ← Train ResNet50 + ViT baseline
│   ├── NB03_ijepa_pretrain.ipynb        ← I-JEPA pretraining trên NIH
│   ├── NB04_ijepa_finetune.ipynb        ← Fine-tune 3 chiến lược
│   ├── NB05_label_efficiency.ipynb      ← Thực nghiệm 1%→100% nhãn
│   ├── NB06_clahe_ablation.ipynb        ← So sánh có/không CLAHE
│   └── NB07_xai_visualization.ipynb     ← Grad-CAM + Attention Rollout
│
├── 📂 src/                              ← Source code Python thuần
│   ├── 📂 models/
│   │   ├── ijepa.py                     ← Kiến trúc I-JEPA chính
│   │   ├── vit.py                       ← ViT-Small/16 backbone
│   │   ├── resnet_baseline.py           ← ResNet50 baseline
│   │   └── vit_baseline.py              ← ViT ImageNet baseline
│   │
│   ├── 📂 datasets/
│   │   ├── nih_dataset.py               ← NIH DataLoader (không nhãn)
│   │   └── rsna_dataset.py              ← RSNA DataLoader (binary label)
│   │
│   ├── 📂 training/
│   │   ├── pretrain.py                  ← Pretraining loop I-JEPA
│   │   ├── finetune.py                  ← Fine-tuning 3 chiến lược
│   │   └── losses.py                    ← MSE latent + BCEWithLogitsLoss
│   │
│   ├── 📂 evaluation/
│   │   ├── metrics.py                   ← AUC, F1, Recall, Specificity
│   │   ├── threshold_analysis.py        ← 6 chiến lược threshold tuning
│   │   └── label_efficiency.py          ← Phân tích label efficiency
│   │
│   └── 📂 explainability/
│       ├── gradcam.py                   ← Grad-CAM cho ResNet50
│       ├── attention_rollout.py         ← Attention Rollout cho ViT/I-JEPA
│       └── pointing_game.py             ← Pointing Game + Overlap Ratio
│
├── 📂 data/                             ← Metadata CSV (không chứa ảnh)
│   ├── nih_pretrain_50k.csv             ← Danh sách 50k ảnh NIH
│   ├── rsna_train.csv                   ← RSNA train (18,678 ảnh)
│   ├── rsna_val.csv                     ← RSNA validation (4,003 ảnh)
│   └── rsna_test.csv                    ← RSNA test (4,003 ảnh) — FIXED split
│
├── 📂 results/                          ← Kết quả thực nghiệm
│   ├── main_results.csv                 ← Bảng tổng hợp tất cả mô hình
│   ├── label_efficiency.csv             ← AUC theo % nhãn × seed
│   ├── clahe_ablation.csv               ← AUC có/không CLAHE
│   ├── threshold_analysis.csv           ← 6 chiến lược threshold
│   └── 📂 heatmaps/                     ← Ảnh Grad-CAM + Attention Map
│
├── 📂 checkpoints/                      ← Model weights (xem hướng dẫn tải)
│   ├── 📂 pretrain/
│   │   └── encoder_epoch50.pth          ← Encoder sau 50 epochs (NIH 50k)
│   └── 📂 finetune/
│       ├── linear_probe_best.pth
│       ├── partial_ft_v1_best.pth
│       ├── partial_ft_v2_best.pth
│       ├── full_ft_v1_best.pth          ← Best model (AUC = 0.8297)
│       └── full_ft_v2_best.pth
│
└── 📂 docs/                             ← Tài liệu nghiên cứu
    ├── paper_NCKH.pdf                   ← Bài báo NCKH (bản chính)
    ├── BAO_CAO_da_cap_nhat.docx         ← Báo cáo đề tài (bản cập nhật)
    └── figures/                         ← Hình vẽ trong báo cáo
```

---

## ⚙️ Cài đặt môi trường

### Yêu cầu hệ thống
- Python 3.10+
- GPU: NVIDIA T4 (16GB VRAM) trở lên
- RAM: 16GB+
- Ổ cứng: ~50GB (dữ liệu)

### Cài đặt thư viện

```bash
git clone https://github.com/YOUR_USERNAME/ijepa-pneumonia-detection.git
cd ijepa-pneumonia-detection
pip install -r requirements.txt
```

### Nội dung `requirements.txt`

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
pydicom>=2.4.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
albumentations>=1.3.0
```

---

## 💾 Dữ liệu

> ⚠️ **Ảnh gốc không được đưa lên GitHub** do kích thước lớn (>50GB). Chỉ có metadata CSV.

### Tải dữ liệu

| Dataset | Nguồn | Vai trò |
|---|---|---|
| NIH ChestX-ray14 (224×224) | [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) | Pretraining (không nhãn) |
| RSNA Pneumonia Detection | [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) | Fine-tuning + Evaluation |

### Cấu trúc sau khi tải

```
/kaggle/input/
├── nih-chest-xrays/          ← NIH (50k ảnh đã được lấy mẫu)
└── rsna-pneumonia-detection-challenge/
    ├── stage_2_train_images/ ← DICOM files
    └── stage_2_train_labels.csv
```

### Phân bố nhãn RSNA (sau khi split)

| Split | Tổng | Pneumonia | Bình thường | Tỷ lệ dương tính |
|---|---|---|---|---|
| Train | 18,678 | 4,207 (22.5%) | 14,471 (77.5%) | 22.5% |
| Validation | 4,003 | 902 (22.5%) | 3,101 (77.5%) | 22.5% |
| Test | 4,003 | 902 (22.5%) | 3,101 (77.5%) | 22.5% |

---

## 🚀 Hướng dẫn chạy

### Cách 1 — Chạy bằng Notebooks (Kaggle) ✅ Khuyến nghị

Mở từng notebook theo thứ tự trên Kaggle:

```
NB01 → NB02 → NB03 → NB04 → NB05 → NB06 → NB07
```

Xem chi tiết tại phần [Notebooks](#-notebooks) bên dưới.

---

### Cách 2 — Chạy bằng script Python

**Bước 1: Chuẩn bị dữ liệu**
```bash
python src/datasets/prepare_rsna.py \
    --input_dir /path/to/rsna/dicom \
    --output_dir /path/to/rsna/png \
    --csv_path /path/to/stage_2_train_labels.csv
```

**Bước 2: Pretrain I-JEPA trên NIH**
```bash
python src/training/pretrain.py \
    --data_csv data/nih_pretrain_50k.csv \
    --image_dir /path/to/nih/images \
    --epochs 50 \
    --batch_size 32 \
    --mask_ratio 0.75 \
    --output_dir checkpoints/pretrain/
```

**Bước 3: Fine-tune trên RSNA**
```bash
# Linear Probing
python src/training/finetune.py \
    --strategy linear_probe \
    --encoder_ckpt checkpoints/pretrain/encoder_epoch50.pth \
    --train_csv data/rsna_train.csv \
    --val_csv data/rsna_val.csv

# Full Fine-tuning (best)
python src/training/finetune.py \
    --strategy full_ft_v1 \
    --encoder_ckpt checkpoints/pretrain/encoder_epoch50.pth \
    --train_csv data/rsna_train.csv \
    --val_csv data/rsna_val.csv
```

**Bước 4: Đánh giá**
```bash
python src/evaluation/metrics.py \
    --model_ckpt checkpoints/finetune/full_ft_v1_best.pth \
    --test_csv data/rsna_test.csv \
    --threshold 0.39
```

**Bước 5: Sinh heatmap Grad-CAM**
```bash
python src/explainability/gradcam.py \
    --model_ckpt checkpoints/finetune/full_ft_v1_best.pth \
    --image_path /path/to/xray.png \
    --output_dir results/heatmaps/
```

---

## 📓 Notebooks

Tất cả thực nghiệm được thực hiện trên **Kaggle GPU T4** dưới dạng notebook `.ipynb`.

| Notebook | Mô tả | Thời gian chạy | Output chính |
|---|---|---|---|
| **NB01** `data_preparation` | Đọc DICOM → PNG, tạo split train/val/test cố định (stratified, seed=42) | ~30 phút | `rsna_train/val/test.csv` |
| **NB02** `baselines` | Train ResNet50 + ViT-Small ImageNet, đánh giá trên test | ~3 giờ | `main_results.csv` (baseline rows) |
| **NB03** `ijepa_pretrain` | Pretraining I-JEPA trên 50k ảnh NIH không nhãn, 50 epochs | ~4 giờ | `encoder_epoch50.pth` |
| **NB04** `ijepa_finetune` | Fine-tune 5 cấu hình (Linear/Partial/Full × v1/v2) trên RSNA | ~6 giờ | `*_best.pth`, kết quả AUC |
| **NB05** `label_efficiency` | Lặp lại train tại 6 mức nhãn (1%→100%) với multiple seeds | ~8 giờ | `label_efficiency.csv` |
| **NB06** `clahe_ablation` | So sánh pipeline có/không CLAHE trên tất cả mô hình | ~4 giờ | `clahe_ablation.csv` |
| **NB07** `xai_visualization` | Grad-CAM (ResNet50) + Attention Rollout (ViT/I-JEPA), Pointing Game | ~2 giờ | `heatmaps/`, pointing game scores |

### Thứ tự phụ thuộc

```
NB01 (data)
  └── NB02 (baselines)
  └── NB03 (pretrain)
        └── NB04 (finetune)
              ├── NB05 (label efficiency)
              ├── NB06 (CLAHE ablation)
              └── NB07 (XAI)
```

> 💡 **Tip:** NB02 và NB03 có thể chạy song song vì không phụ thuộc nhau.

---

## 🏗️ Kiến trúc I-JEPA

```
Ảnh X-quang (224×224)
        │
        ▼
  Chia thành 196 patches (16×16)
        │
        ├──── Student Encoder (ViT-Small/16) ──── 196 patch embeddings (dim=384)
        │                                               │
        │                                               ▼
        │                                    Predictor (Narrow Transformer)
        │                                    Dự đoán 147 target patches (75%)
        │                                               │
        └──── Target Encoder (EMA update) ─── Ground truth embeddings
                                                        │
                                            MSE Loss trong latent space
                                            (không tái tạo pixel)
```

### Thông số kỹ thuật

| Thành phần | Chi tiết |
|---|---|
| Student/Target Encoder | ViT-Small/16 — 12 blocks, 6 heads, embed_dim=384 |
| Predictor | Narrow Transformer — 4 blocks, 4 heads, embed_dim=256 |
| Mask ratio | 0.75 (che 147/196 patches) |
| EMA momentum | 0.996 → 0.999996 (tăng tuyến tính 50 epochs) |
| Loss function | MSE trong latent space |
| Pretrain data | NIH ChestX-ray14 — 50,000 ảnh không nhãn |
| Pretrain epochs | 50 epochs (~4 giờ trên Kaggle T4) |

---

## 📌 Điểm khác biệt với I-JEPA gốc

| Thành phần | I-JEPA gốc (Meta AI) | Phiên bản này |
|---|---|---|
| Masking strategy | Block masking (vùng hình chữ nhật liền nhau) | **Random patch masking** |
| Student Encoder input | Chỉ xử lý context patches | Xử lý toàn bộ 196 patches |
| Pretraining scale | 300–800 epochs, ImageNet | **50 epochs, NIH 50k** |
| Hardware | Cluster GPU lớn | **Kaggle T4 (16GB)** |

> **Lý do điều chỉnh:** Random masking đơn giản hơn, ổn định hơn trên Kaggle T4, và được hỗ trợ bởi A-JEPA, Audio-JEPA, RadJEPA cho domain mới.

---

## 📈 Pretrain Loss Curve

```
Epoch  1: Loss = 0.1185
Epoch 10: Loss ≈ 0.0200  (giảm nhanh)
Epoch 30: Loss ≈ 0.0080  (giảm chậm hơn)
Epoch 50: Loss = 0.0041  (giảm 96.5%, chưa bão hòa)
```

---

## 📄 Tài liệu nghiên cứu

| Tài liệu | Mô tả | Link |
|---|---|---|
| Bài báo NCKH | Bài báo khoa học đầy đủ (tiếng Anh) | [`docs/paper_NCKH.pdf`](docs/paper_NCKH.pdf) |
| Báo cáo đề tài | Báo cáo 6 chương (tiếng Việt) | [`docs/BAO_CAO_da_cap_nhat.docx`](docs/BAO_CAO_da_cap_nhat.docx) |

### Tài liệu tham khảo chính

| # | Tác giả | Năm | Tiêu đề |
|---|---|---|---|
| [1] | Assran et al. | 2023 | Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA) |
| [2] | He et al. | 2022 | Masked Autoencoders Are Scalable Vision Learners (MAE) |
| [3] | Dosovitskiy et al. | 2021 | An Image is Worth 16x16 Words (ViT) |
| [4] | Wang et al. | 2017 | ChestX-ray8: Hospital-scale Chest X-ray Database |
| [5] | Alis et al. | 2026 | RadJEPA: Adapting I-JEPA for Radiology |
| [6] | Azizi et al. | 2021 | Big Self-Supervised Models Advance Medical Image Classification |

---

## ⚠️ Giới hạn nghiên cứu

1. **Compute budget:** Chỉ 50 epochs pretrain (loss chưa bão hòa) — AUC là giới hạn dưới
2. **Random masking:** Không implement block masking gốc của I-JEPA
3. **Dữ liệu Mỹ only:** Chưa validate trên ảnh X-quang bệnh nhân Việt Nam
4. **XAI không đồng nhất:** Grad-CAM (ResNet50) vs Attention Rollout (ViT/I-JEPA) — không so sánh trực tiếp được
5. **Single seed ≥10%:** Crossover point là ước lượng, chưa xác nhận thống kê

---

## 📜 Trích dẫn

Nếu bạn sử dụng code hoặc kết quả trong nghiên cứu của mình:

```bibtex
@article{nguyen2026ijepa_pneumonia,
  title     = {Self-Supervised Pneumonia Detection from Chest X-Rays via I-JEPA:
               A Latent-Space Predictive Approach with Label-Efficient Learning},
  author    = {Nguyen, Tong Phuc},
  year      = {2026},
  note      = {Undergraduate Research, Nam Can Tho University}
}
```

---

## 👤 Tác giả

**Nguyễn Tống Phúc**  
Sinh viên NCKH — Trường Đại học Nam Cần Thơ  
GVHD: ThS. Trần Văn Thiện

---

## 📝 License

MIT License — xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

*⚕️ Lưu ý: Kết quả từ hệ thống này chỉ mang tính tham khảo khoa học. Không dùng thay thế chẩn đoán của bác sĩ.*
