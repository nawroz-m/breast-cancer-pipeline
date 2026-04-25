# 🖼️ Breast Cancer Detector

A simple Streamlit-based AI application for **object detection and segmentation**.  
Users upload an image, receive a predicted bounding box, and optionally refine it using a segmentation model.

---

## 🚀 Features

- Upload image (JPG, PNG, WEBP)
- Predict bounding box using a deep learning model
- Review and confirm prediction
- Generate segmentation contour after confirmation
- Interactive model switching (ready for multi-model setup)

---

## 🧠 Pipeline

1. **Input Image**
2. **Bounding Box Prediction**
3. **User Confirmation**
4. **Segmentation / Contour Prediction**

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Streamlit
- OpenCV
- Albumentations

---

## 📦 Models

- Bounding Box Model (CNN-based encoder-decoder)
- Segmentation Model (U-Net style architecture)

> Note: Currently single active model per task; multi-model support is extendable.

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py