import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import streamlit as st
from PIL import Image
from torch import nn
import torch
import numpy as np
import cv2
from albumentations.pytorch import ToTensorV2
import hashlib
import base64
from io import BytesIO
from ultralytics import YOLO
import torchvision
from utils_classes import UNet, DoubleConv, Cnn_network, TumorClassifier, Segment_Cnn_network
from utils_pred import predict_bbox, predict_contour, predict_classes, predict_contour_cnn_resnet
from utils_helper import crop_and_resize, render_image_fixed, placeholder_content

# -------------------- CONFIG --------------------
st.set_page_config(page_title="BSU detector", page_icon="bsu-icon.png", layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



BBOX_MODELS = ["YOLOv8 (best)", "CNN (Unet)"]
SEGMENT_MODELS = ["Unet (best)", "CNN (Resnet)"]


# -------------------- LOAD MODEL --------------------

# bounding box models
bbox_model1 = YOLO("models/best_tumor_model.pt")

bbox_model2 = Cnn_network().to(device) 
bbox_model2.load_state_dict(torch.load('models/bbox_m6.pth', map_location=device))
bbox_model2.eval()

# Segmentation model
unet_segment_model = UNet()
unet_segment_model = torch.load(
    "models/best_unet.pt",
    map_location=device,
    weights_only=False  
)
unet_segment_model.eval()

# -------------------- LOAD Segmentation MODEL --------------------
cnn_res_segment_model = Segment_Cnn_network(num_class=1).to(device)

state_dict = torch.load('models/mask6_with_dice_bce_loss(best1).pt', map_location=device)
cnn_res_segment_model.load_state_dict(state_dict)
cnn_res_segment_model.eval()


# Classification model 
classification_model = TumorClassifier().to(device) 
checkpoint = torch.load('models/classifier_lyn.pth', map_location=device)
classification_model.load_state_dict(checkpoint["model_state_dict"])
classification_model.eval()

def load_bbox_model():
    if st.session_state.bbox_model == BBOX_MODELS[0]:
        return bbox_model1
    elif st.session_state.bbox_model == BBOX_MODELS[1]:
        return bbox_model2

def load_seg_model():
    if st.session_state.seg_model == SEGMENT_MODELS[0]:
        return unet_segment_model
    elif st.session_state.seg_model == SEGMENT_MODELS[1]:
        return cnn_res_segment_model 

def load_classification_model():
    return classification_model

# -------------------- SESSION --------------------
if "selected_model" not in st.session_state:
    st.session_state.selected_model = BBOX_MODELS[0]

if "stage" not in st.session_state:
    st.session_state.stage = "bbox"

def update_from_left():
    st.session_state.selected_model = st.session_state.left_select

def update_from_right():
    st.session_state.selected_model = st.session_state.right_select
    
# -------------------- UI --------------------
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "classification_result" not in st.session_state:
    st.session_state.classification_result = None

if "pred_box" not in st.session_state:
    st.session_state.pred_box = None

if "bbox_model" not in st.session_state:
    st.session_state.bbox_model = BBOX_MODELS[0]

if "seg_model" not in st.session_state:
    st.session_state.seg_model = SEGMENT_MODELS[0]
def sync_segmentation_with_bbox():
    if st.session_state.bbox_model == BBOX_MODELS[0]:   # YOLOv8 (best)
        st.session_state.seg_model = SEGMENT_MODELS[0]  # Unet (best)
    elif st.session_state.bbox_model == BBOX_MODELS[1]: # CNN (Unet)
        st.session_state.seg_model = SEGMENT_MODELS[1]  # CNN (Resnet)
sync_segmentation_with_bbox()

if "bbox_preview" not in st.session_state:
    st.session_state.bbox_preview = None

if "mask_preview" not in st.session_state:
    st.session_state.mask_preview = None

def reset_session():
    keys = [
        "current_image",
        "classification_result",
        "pred_box",
        "bbox_preview",
        "mask_preview",
    ]
    for k in keys:
        st.session_state[k] = None

    st.session_state.stage = "bbox"
    st.session_state.last_uploaded = None

def on_bbox_change():
    st.session_state.bbox_model = st.session_state.bbox_select
    sync_segmentation_with_bbox()

    # clear old results 
    st.session_state.current_image = None
    st.session_state.pred_box = None
    st.session_state.bbox_preview = None
    st.session_state.mask_preview = None
    st.session_state.classification_result = None

    st.session_state.stage = "bbox"

def go_back_stage():
    if st.session_state.stage == "contour":
        st.session_state.stage = "bbox"
        st.session_state.mask_preview = None  # remove forward result

    elif st.session_state.stage == "classify":
        st.session_state.stage = "contour"
        st.session_state.classification_result = None  # remove forward result

    st.rerun()

left_col, right_col = st.columns([1, 1], gap="large")

# LEFT
with left_col:
    # st.markdown("### Upload")
    st.markdown("### Upload Medical Image")
    st.caption(""" 
        AI-assisted diagnosis tool — for research and clinical support only \n\n 
        📌 Upload a breast scan image (MRI / Ultrasound / X-ray).
    """)
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
     #  Update the session
    if "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = None
    # -------- CLEAR STATE WHEN FILE REMOVED --------
    if uploaded_file is not None:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest() 

        if st.session_state.last_uploaded != file_hash:
            reset_session()   #  reset everything cleanly
            st.session_state.last_uploaded = file_hash
            st.rerun()        #  force fresh UI
    # ---------------- PREVIEW PANEL ----------------

    def pil_from_array(img):
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
        return img

    def image_download_button(img, filename):
        buf = BytesIO()
        pil_img = pil_from_array(img)
        pil_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="⬇️ Download",
            data=byte_im,
            file_name=filename,
            mime="image/png",
            use_container_width=True
        )

    # -------- BBOX PREVIEW --------
    if (st.session_state.bbox_preview is not None and  st.session_state.stage in ["contour", "classify"]):
        with st.expander("📦 Cancer Box Preview", st.session_state.stage != "bbox"):
            st.image(st.session_state.bbox_preview, use_container_width=True, clamp=True)
            image_download_button(st.session_state.bbox_preview, "bbox_result.png")

    # -------- MASK PREVIEW --------
    if (st.session_state.mask_preview is not None and  st.session_state.stage == "classify"):
        with st.expander("🧬 Masked Preview", st.session_state.stage != "bbox"):
            st.image(st.session_state.mask_preview, use_container_width=True, clamp=True)
            image_download_button(st.session_state.mask_preview, "mask_result.png")

    if uploaded_file is None and st.session_state.last_uploaded is not None:
        reset_session()
        st.rerun()
with right_col:
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB") 

        # PLACEHOLDER (key part)
        image_placeholder = st.empty() 

        # ---------------- STAGE 1: BBOX ----------------
        if st.session_state.stage == "bbox":
            with st.spinner(f"Running {st.session_state.selected_model}..."):
                current_model = load_bbox_model()
                pred_img, pred_box = predict_bbox(image, current_model, st.session_state.bbox_model)

                #  update only AFTER ready
                st.session_state.current_image = pred_img
                st.session_state.pred_box = pred_box
                st.session_state.bbox_preview = pred_img.copy()

            render_image_fixed(st.session_state.current_image)

        # ---------------- STAGE 2: CONTOUR ----------------
        elif st.session_state.stage == "contour":
            with st.spinner("Running contour model..."):
                current_model = load_seg_model()
                if st.session_state.seg_model == SEGMENT_MODELS[1]:
                    contour_img = predict_contour_cnn_resnet(image, current_model)
                else:
                    contour_img = predict_contour(image, current_model, st.session_state.pred_box, device)

                st.session_state.current_image = contour_img
                st.session_state.mask_preview = contour_img.copy()

            render_image_fixed(st.session_state.current_image)

      
        # BUTTONS (Forward + Back) 
        col_btn1, col_btn2 = st.columns([1, 1]) 
         
        with col_btn1:
            if st.session_state.stage != "classify":
                if st.button("⬅️ Back", disabled=st.session_state.stage == "bbox", use_container_width=True):
                    go_back_stage()

        # CONFIRM BUTTON 
        with col_btn2:
            if st.session_state.stage != "classify":
                if st.button("✅ Confirm Results", use_container_width=True, type="primary"):
                    if st.session_state.stage == "bbox":
                        st.session_state.stage = "contour"

                    elif st.session_state.stage == "contour":
                        st.session_state.stage = "classify"

                    st.rerun()

        # BUTTON CHANGES STAGE
        if st.session_state.stage != "classify": 
            if  st.session_state.stage == "bbox":
                st.selectbox(
                    "Bounding Box Model",
                    options=BBOX_MODELS,
                    key="bbox_select",
                    index=BBOX_MODELS.index(st.session_state.bbox_model),
                    on_change=on_bbox_change
                )
            if st.session_state.stage == "contour":
                st.selectbox(
                    "Mask Model",
                    options=SEGMENT_MODELS,
                    key="segment_select",
                    index=SEGMENT_MODELS.index(st.session_state.seg_model),
                    disabled=True
                )

          # ---------------- STAGE 3: CLASSIFICATION ----------------
        elif st.session_state.stage == "classify":

            with st.spinner("Running classification model..."):
                current_model = load_classification_model()                
            render_image_fixed(st.session_state.current_image)

            if st.session_state.classification_result is None:
                st.session_state.classification_result = predict_classes(image, current_model, st.session_state.pred_box, device)
 
            result = st.session_state.classification_result 

            col1, col2 = st.columns([1, 1])             
            with col1:
                if st.button("⬅️ Back", disabled=st.session_state.stage == "bbox", use_container_width=True):
                    go_back_stage()

            with col2:
                st.button("✅ Confirm Results", disabled=True, use_container_width=True, type="primary")

            with col1:
                st.markdown("#### 🧠 Final Analysis")
                if result["label"] == "Malignant":
                    st.error(f"Result: {result['label']}")
                else:
                    st.success(f"Result: {result['label']}")

                st.metric(
                    label="Confidence",
                    value=f"{result['confidence']*100:.2f}%"
                )

            with col2:
                st.write("#### Probability Breakdown")
                for cls, prob in result["all_probs"].items():
                    st.progress(float(prob))
                    st.caption(f"{cls}: {prob*100:.1f}%")

    else:
       placeholder_content()
