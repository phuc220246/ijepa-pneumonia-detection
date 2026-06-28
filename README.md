# 🫁 I-JEPA Pneumonia Detection

> **Self-Supervised Pneumonia Detection from Chest X-Rays via I-JEPA**  
> Kiến trúc lai I-JEPA + Random Masking · ViT-Small/16 · NIH → RSNA · AUC = 0.8297

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20T4-yellow)](https://kaggle.com)

---

## 🔬 Tóm tắt đề tài

Nghiên cứu này **đánh giá** việc ứng dụng kiến trúc **I-JEPA (Image-based Joint-Embedding Predictive Architecture)** với chiến lược **random patch masking** cho bài toán phát hiện viêm phổi từ ảnh X-quang ngực, đặc biệt là câu hỏi liệu tiền-huấn-luyện tự giám sát *in-domain* có cải thiện hiệu quả sử dụng nhãn so với khởi tạo ImageNet hay không.

### Vấn đề (động cơ)
- Thiếu bác sĩ X-quang chuyên khoa tại tuyến cơ sở Việt Nam
- Gán nhãn ảnh y tế tốn kém: 5–15 phút/ảnh × hàng nghìn ảnh
- Mô hình học có giám sát cần nhiều nhãn mới hoạt động tốt

### Phương pháp
- **Phase 1 — Pretraining (không nhãn):** I-JEPA học biểu diễn cấu trúc giải phẫu lồng ngực từ 50.000 ảnh NIH không nhãn (50 epochs, latent-space prediction)
- **Phase 2 — Fine-tuning & đánh giá:** Tinh chỉnh trên RSNA; Full fine-tuning đạt **AUC = 0.8297** với **Recall = 0.7738** tại ngưỡng mặc định 0.50 — ưu thế rõ về recall/hành vi ngưỡng so với baseline cùng kiến trúc

### Đóng góp khoa học
| Đóng góp | Chi tiết |
|---|---|
| Kiến trúc lai | I-JEPA + Random Masking cho ảnh X-quang (thay block masking gốc) |
| Hành vi ngưỡng (RQ1) | I-JEPA Full FT v1 hoạt động tốt quanh ngưỡng mặc định 0.50 với Recall = 0.7738, cao hơn baseline ViT cùng kiến trúc |
| Phân tích label efficiency (RQ2) | Đánh giá hệ thống 6 mức nhãn (1% → 100%) với thiết kế chống rò rỉ nhãn — **kết quả trung thực, gồm cả kết quả âm tính** |
| CLAHE ablation | Phát hiện CLAHE làm giảm AUC nhất quán khi dùng cùng tiền-huấn-luyện SSL (ΔAUC = −0.047) |
| Pipeline tái lập | Toàn bộ quy trình EDA → SSL pretrain → fine-tune → XAI chạy được trên một Kaggle T4 |

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

> **Đóng góp chính của I-JEPA không phải AUC tuyệt đối** (thấp hơn baseline do ngân sách pretrain hạn chế) **mà là hành vi ngưỡng**: I-JEPA Full FT v1 đạt Recall = 0.7738 ngay tại ngưỡng mặc định 0.50 — phù hợp bối cảnh tầm soát nơi bỏ sót ca bệnh tốn kém hơn báo động giả.

### Label Efficiency — 6 mức nhãn (Test AUC trên RSNA)

> Cả ba mô hình đều xuất phát từ tiền-huấn-luyện *chưa từng thấy nhãn RSNA*: I-JEPA nạp **encoder SSL (NIH) + đầu phân loại khởi tạo ngẫu nhiên**; baseline nạp trọng số ImageNet. Thiết kế này đảm bảo so sánh hợp lệ, không rò rỉ nhãn.

| %Nhãn (n ảnh) | I-JEPA SSL | ResNet50 | ViT-Small/16 |
|---|---|---|---|
| 1% (187) | 0.527 ± 0.037 | 0.782 ± 0.016 | 0.789 ± 0.014 |
| 5% (934) | 0.598 ± 0.024 | 0.800 ± 0.026 | 0.825 ± 0.004 |
| 10% (1,868) | 0.781 | 0.816 | 0.834 |
| 25% (4,670) | 0.795 | 0.834 | 0.850 |
| 50% (9,339) | 0.796 | 0.838 | 0.861 |
| 100% (18,678) | 0.808 | 0.856 | 0.867 |

> **Phát hiện (RQ2 — kết quả âm tính):** Trong điều kiện tài nguyên của đề tài (pretrain 50 epoch trên 50k ảnh, random masking), I-JEPA in-domain **chưa vượt** baseline ImageNet ở *bất kỳ* mức nhãn nào. Ở 1% (encoder frozen) chỉ đạt ≈ 0.53 do đặc trưng SSL **chưa hội tụ**; khi fine-tune (≥10%) vẫn thấp hơn baseline ~0.05–0.06 AUC. Khoảng cách cố định ngay ở 100% gợi ý nguyên nhân là *chất lượng pretrain* chứ không phải số lượng nhãn — định hướng cần pretrain quy mô/thời lượng lớn hơn.

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

> 🔒 **Kiểm soát rò rỉ dữ liệu:** Ảnh RSNA được tuyển từ các ca *không* nằm trong bản phát hành công khai NIH ChestX-ray14; đã kiểm tra không có chồng lấp bệnh nhân giữa tập pretrain (50k NIH) và tập fine-tune/test (RSNA).

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

> **Lý do điều chỉnh:** Random masking đơn giản hơn, ổn định hơn trên Kaggle T4, và được hỗ trợ bởi A-JEPA, Audio-JEPA, RadJEPA cho domain mới. *Lưu ý:* đề tài chưa thực hiện ablation trực tiếp so block-vs-random masking trên CXR — đây là hướng phát triển.

---

## 📈 Pretrain Loss Curve

```
Epoch  1: Loss = 0.1185
Epoch 10: Loss ≈ 0.0200  (giảm nhanh)
Epoch 30: Loss ≈ 0.0080  (giảm chậm hơn)
Epoch 50: Loss = 0.0041  (giảm 96.5%, chưa bão hòa)
```

> Loss giảm ổn định, không mode collapse — nhưng **chưa bão hòa** ở 50 epoch. Đây là một nguyên nhân chính khiến biểu diễn SSL chưa đủ mạnh để cạnh tranh với ImageNet transfer (xem phần Label Efficiency).

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

1. **Ngân sách compute:** Chỉ 50 epochs pretrain trên 50k ảnh (loss chưa bão hòa) — nhiều khả năng là nguyên nhân chính khiến I-JEPA in-domain chưa vượt baseline ImageNet
2. **Random masking:** Chưa implement và ablate block masking gốc của I-JEPA trên CXR
3. **Dữ liệu Mỹ only:** Chưa validate trên ảnh X-quang bệnh nhân Việt Nam (VinDr-CXR là hướng ưu tiên)
4. **XAI không đồng nhất:** Grad-CAM (ResNet50) vs Attention Rollout (ViT/I-JEPA) — chưa so sánh trực tiếp được
5. **Single seed ≥10%:** Kết quả label-efficiency ở vùng 10–100% là một lần chạy (seed = 42), chưa đánh giá đa-seed
6. **Chưa kiểm định thống kê:** Khoảng cách AUC giữa các mô hình chưa được xác nhận bằng DeLong test / bootstrap CI

---

## 📜 Trích dẫn

Nếu bạn sử dụng code hoặc kết quả trong nghiên cứu của mình:

```bibtex
@article{nguyen2026ijepa_pneumonia,
  title     = {Self-Supervised Pneumonia Detection from Chest X-Rays via I-JEPA:
               A Latent-Space Predictive Approach},
  author    = {Nguyen, Tong Phuc},
  year      = {2026},
  note      = {Undergraduate Research, Nam Can Tho University}
}
```
## 🌐 Đường link truy cập hệ thống

Hệ thống **PneumoScan AI** hiện được triển khai trực tuyến trên máy chủ CloudFly. Người dùng có thể truy cập và trải nghiệm các chức năng của hệ thống như tải ảnh X-quang ngực, lựa chọn mô hình AI, thực hiện dự đoán và xem kết quả phân loại thông qua địa chỉ dưới đây.

👉 **Địa chỉ Web công khai:** [https://your-domain.com](http://103.82.21.245/)

👉 Địa chỉ Web công khai:
---

## 👤 Tác giả

**Nguyễn Tòng Phúc**  
Sinh viên NCKH — Trường Đại học Nam Cần Thơ  
GVHD: NCS. Trần Văn Thiện

---

## 📝 License

MIT License — xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

*⚕️ Lưu ý: Kết quả từ hệ thống này chỉ mang tính tham khảo khoa học. Không dùng thay thế chẩn đoán của bác sĩ.*
