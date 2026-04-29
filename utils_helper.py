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


def render_image_fixed(img_array):
    # convert numpy → PNG → base64
    buffer = BytesIO()
    Image.fromarray(img_array).save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    st.markdown(
        f"""
        <div style="
            width: 100%;
            height: 420px;
            border: 1px dashed #ccc;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #111;
            overflow: hidden;
            padding: 2px;
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
    st.markdown("### Preview & Verification")
    placeholder = st.empty()

    placeholder.markdown(
            """
            <div style="
                height: 420px;
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