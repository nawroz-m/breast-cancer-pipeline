import streamlit as st
from PIL import Image
from torch import nn
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import hashlib

# -------------------- CONFIG --------------------
st.set_page_config(page_title="BSU detector", page_icon="🖼️", layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_size = 300
g_mean = (0.485, 0.456, 0.406)
g_std = (0.229, 0.224, 0.225)

MODELS = ["YOLOv8 (Fast)", "ResNet50 (Balanced)", "EfficientNet (Accurate)"]

# -------------------- MODEL --------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c)
        )

        if in_c != out_c or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class Cnn_network(nn.Module):
    def __init__(self, num_class):
        super().__init__()

        self.block1 = nn.Sequential(
            ResidualBlock(3, 32, stride=2),
            ResidualBlock(32, 32)
        )
        self.block2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64)
        )
        self.block3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        self.block4 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )

        self.dropout = nn.Dropout2d(0.3)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.up4 = nn.ConvTranspose2d(32, 32, 2, 2)

        self.conv1 = ResidualBlock(256, 128)
        self.conv2 = ResidualBlock(128, 64)
        self.conv3 = ResidualBlock(64, 32)
        self.conv4 = ResidualBlock(32, 32)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.final = nn.Linear(32, 16)
        self.box_head = nn.Linear(16, num_class)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.dropout(self.block4(x3))

        x = self.up1(x4)
        x = nn.functional.interpolate(x, size=x3.shape[2:])
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = nn.functional.interpolate(x, size=x2.shape[2:])
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = nn.functional.interpolate(x, size=x1.shape[2:])
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = self.conv4(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.final(x)
        return torch.sigmoid(self.box_head(x))

class Segment_ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c)
        )

        # match dimensions for skip connection
        if in_c != out_c or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.skip(x)      # original path

        out = self.conv(x)           # transformed path

        out = out + residual
        out = self.relu(out)  #  RESIDUAL ADDED

        return out

# Builing the network
class Segment_Cnn_network(nn.Module):
  def __init__(self, num_class):
    super(Segment_Cnn_network, self).__init__()

    # Encoder
    self.block1 = nn.Sequential(
        Segment_ResidualBlock(3, 32, stride=2),
        Segment_ResidualBlock(32, 32, stride=1)
    )

    self.block2 = nn.Sequential(
        Segment_ResidualBlock(32, 64, stride=2),
        Segment_ResidualBlock(64, 64, stride=1)
    )

    self.block3 = nn.Sequential(
        Segment_ResidualBlock(64, 128, stride=2),
        Segment_ResidualBlock(128, 128, stride=1)
    )

    self.block4 = nn.Sequential(
        Segment_ResidualBlock(128, 256, stride=2),
        Segment_ResidualBlock(256, 256, stride=1)
    )

    self.dropout = nn.Dropout2d(0.3)

    # Decoder
    self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
    self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

    self.conv1 = Segment_ResidualBlock(256, 128, stride=1)
    self.conv2 = Segment_ResidualBlock(128, 64, stride=1)
    self.conv3 = Segment_ResidualBlock(64, 32, stride=1)
    self.conv4 = Segment_ResidualBlock(32, 32, stride=1)


    self.final = nn.Conv2d(32, 1, kernel_size=1)


  def forward(self, x):

    # Encoder
    x1 = self.block1(x)
    x2 = self.block2(x1)
    x3 = self.block3(x2)
    x4 = self.block4(x3) # <-- Bottleneck
    x4 = self.dropout(x4) # <-- reduce noise at bottleneck

    # Decoder
    x = self.up1(x4)
    x = nn.functional.interpolate(x, size=x3.shape[2:])
    x = torch.cat([x, x3], dim=1)
    x = self.conv1(x)

    x = self.up2(x)
    x = nn.functional.interpolate(x, size=x2.shape[2:])
    x = torch.cat([x, x2], dim=1)
    x = self.conv2(x)

    x = self.up3(x)
    x = nn.functional.interpolate(x, size=x1.shape[2:])
    x = torch.cat([x, x1], dim=1)
    x = self.conv3(x)

    x = self.up4(x)
    x = self.conv4(x)

    x = self.final(x)
    return x

# -------------------- LOAD MODEL --------------------
model = Cnn_network(num_class=4).to(device)

state_dict = torch.load('models/bbox_m2.pt', map_location=device)
model.load_state_dict(state_dict)
model.eval()


# -------------------- LOAD Segmentation MODEL --------------------
segment_model = Segment_Cnn_network(num_class=1).to(device)

state_dict = torch.load('models/mask6_with_dice_bce_loss(best1).pt', map_location=device)
segment_model.load_state_dict(state_dict)
segment_model.eval()

def load_model(name):
    if st.session_state.stage == "bbox":

        return model  # only one model for now
    if st.session_state.stage == "contour":
        return segment_model


# -------------------- TRANSFORM --------------------
validation_transform_pipeline = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=g_mean, std=g_std),
    ToTensorV2(),
])


# -------------------- PREDICTION --------------------
def predict_bbox(image_pil, model):
    img = np.array(image_pil)

    transformed = validation_transform_pipeline(image=img)
    img_tensor = transformed["image"].to(device)

    with torch.no_grad():
        pred = model(img_tensor.unsqueeze(0))

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    h_img, w_img, _ = img_np.shape

    x, y, w, h = pred[0]

    x = int(x.item() * w_img)
    y = int(y.item() * h_img)
    w = int(w.item() * w_img)
    h = int(h.item() * h_img)

    x, y = max(0, x), max(0, y)
    w, h = max(1, w), max(1, h)

    x = min(x, w_img - 1)
    y = min(y, h_img - 1)
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    # denormalize
    img_np = img_np * np.array(g_std) + np.array(g_mean)
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_np

def predict_contour(image_pil, model):
    model.eval()

    # PIL → numpy
    img = np.array(image_pil)

    # apply SAME transform as training
    transformed = validation_transform_pipeline(image=img)
    img_tensor = transformed["image"].to(device)

    with torch.no_grad():
        pred = model(img_tensor.unsqueeze(0))
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

    # prepare image for drawing
    img_vis = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_vis = img_vis * np.array(g_std) + np.array(g_mean)
    img_vis = (img_vis * 255).clip(0, 255).astype(np.uint8).copy()

    # mask → binary
    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    # find contours
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours (green)
    cv2.drawContours(img_vis, contours, -1, (0, 255, 0), 2)

    return img_vis
# -------------------- SESSION --------------------
if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODELS[0]

if "stage" not in st.session_state:
    st.session_state.stage = "bbox"

def update_from_left():
    st.session_state.selected_model = st.session_state.left_select


def update_from_right():
    st.session_state.selected_model = st.session_state.right_select

    
# -------------------- UI --------------------
if "current_image" not in st.session_state:
    st.session_state.current_image = None

left_col, right_col = st.columns([1, 1], gap="large")

# LEFT
with left_col:
    st.markdown("### Upload")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
     #  Update the session
    if "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = None
    if uploaded_file is not None:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        # if st.session_state.last_uploaded != uploaded_file.name:
        #     st.session_state.stage = "bbox"
        #     st.session_state.last_uploaded = uploaded_file.name

        if st.session_state.last_uploaded != file_hash:
            st.session_state.stage = "bbox"
            st.session_state.last_uploaded = file_hash
            st.session_state.current_image = None  #  reset image
    st.selectbox(
        "Model Selection",
        options=MODELS,
        key="left_select",
        index=MODELS.index(st.session_state.selected_model),
        on_change=update_from_left
    )

    st.info(f"Current System Model: **{st.session_state.selected_model}**")

with right_col:
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB") 

        # PLACEHOLDER (key part)
        image_placeholder = st.empty() 

        # ---------------- SHOW LAST IMAGE FIRST ----------------
        if st.session_state.current_image is not None:
            image_placeholder.image(st.session_state.current_image, width='stretch')

        # ---------------- STAGE 1: BBOX ----------------
        if st.session_state.stage == "bbox":
            with st.spinner(f"Running {st.session_state.selected_model}..."):
                current_model = load_model(st.session_state.selected_model)
                pred_img = predict_bbox(image, current_model)

                #  update only AFTER ready
                st.session_state.current_image = pred_img

            image_placeholder.image(st.session_state.current_image, width='stretch')

        # ---------------- STAGE 2: CONTOUR ----------------
        elif st.session_state.stage == "contour":
            with st.spinner("Running contour model..."):
                current_model = load_model(st.session_state.selected_model)
                contour_img = predict_contour(image, current_model)
                 # update only AFTER ready
                st.session_state.current_image = contour_img

            image_placeholder.image(st.session_state.current_image, width='stretch')


        # BUTTON CHANGES STAGE
        if st.button("✅ Confirm Results", use_container_width=True, type="primary"):
            st.session_state.stage = "contour"
            st.rerun()
        # Model selection option
        # st.write("### Or if it looks wrong choose a diffirent model")
        st.selectbox(
            "If it looks wrong choose a diffirent model",
            options=MODELS,
            key="right_select",
            index=MODELS.index(st.session_state.selected_model),
            on_change=update_from_right,
        )
    else:
        st.markdown("### Preview & Verification")
        placeholder = st.empty()

        placeholder.markdown(
            """
            <div style="
                height: 450px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                border: 2px dashed #aaa;
                border-radius: 10px;
                color: #888;
                font-size: 18px;
                text-align: center;
            ">
                <h2>Upload your image to start with</h2>
                <p>Detecting cancer → <strong> Confirm </strong> → Detecting mask →  <strong> Confirm </strong> → Classify cancer</p>
            </div>
            """,
            unsafe_allow_html=True
        )
