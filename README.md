# 🫁 I-JEPA Pneumonia Detection

> **Self-Supervised Pneumonia Detection from Chest X-Rays via I-JEPA**  
> Kiến trúc lai I-JEPA + Random Masking · ViT-Small/16 · NIH → RSNA · AUC = 0.8297

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20T4-yellow)](https://kaggle.com)

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
| NIH ChestX-ray14 (224×224) | [Kaggle](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized) | Pretraining (không nhãn) |
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
