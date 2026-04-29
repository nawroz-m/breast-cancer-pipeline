import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import cv2
# ------------- Helpers methods ----------------
def crop_and_resize(img, box, size=256):
    x1, y1, x2, y2 = box

    crop_img = img[y1:y2, x1:x2]

    crop_img = cv2.resize(crop_img, (size, size))

    return crop_img

def download_link(img_path, text="download sample image"):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    return f'''
    <a href="data:file/png;base64,{b64}" download="sample.png"
       style="color:#1f77b4; text-decoration:underline; cursor:pointer;">
       {text}
    </a>
    '''

def render_image_fixed(img_array):
    # convert numpy → PNG → base64
    buffer = BytesIO()
    Image.fromarray(img_array).save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    st.markdown(
        f"""
       <div style="
            height: 420px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            border: 2px dashed #4a90e2;
            border-radius: 12px;
            padding: 25px;
            background-color: #f5f7fa;
            color: #333333;
        ">
            <img src="data:image/png;base64,{img_str}"
                 style=" 
                    width: 100%;
                    height: 100%;
                    max-width: 100;
                    max-height: 100;
                    object-fit: contain;
                 ">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")


def placeholder_content():
    st.markdown("### 🧠 AI Diagnosis Workflow")

    st.markdown(
        """
        <div style="
            height: 420px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            border: 2px dashed #4a90e2;
            border-radius: 12px;
            padding: 25px;
            background-color: #f5f7fa;
            color: #333333;
        ">

        <h3 style="text-align:center; margin-bottom:20px; color:#222;">
            Upload a medical image to begin analysis
        </h3>

        <div style="font-size:16px; line-height:1.8; color:#444;">

        <p>🔹 <b>Step 1:</b> Upload image (MRI / Ultrasound / X-ray)</p>
        <p>🔹 <b>Step 2:</b> Detect tumor region → <b>Confirm</b></p>
        <p>🔹 <b>Step 3:</b> Segment tumor mask → <b>Confirm</b></p>
        <p>🔹 <b>Step 4:</b> Classify result (Benign / Malignant)</p>

        </div>

        <div style="
            margin-top:20px;
            text-align:center;
            font-size:13px;
            color:#777;
        ">
            AI-assisted diagnostic pipeline for research & clinical support
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

def upload_desc():
    st.markdown(f"""
    <div style="
            margin-bottom:10px;
            font-size:13px;
            color:#777;
        ">    
        AI-assisted diagnosis tool — for research and clinical support only  
        <div> 📌 Upload a breast scan image (MRI / Ultrasound / X-ray) {download_link("assets/malignant (167).png")}</div>

    </div>
    """, unsafe_allow_html=True)