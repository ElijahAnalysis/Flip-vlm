import streamlit as st
import requests
import base64
from PIL import Image
import io

st.set_page_config(layout="wide")

def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_background(image_path):
    base64_img = get_base64_image(image_path)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Background image
set_background(r"C:\Users\User\Desktop\flip_vlm\dvJEbeccgoI.jpg")

# App title
st.title("🔎 Flip.kz Dual Encoder Search")

# API base
base_url = "http://127.0.0.1:1234"

# --- Search Tabs ---
tab1, tab2 = st.tabs(["🔤 Text Search", "🖼️ Image Search"])

# --- TEXT SEARCH ---
with tab1:
    query = st.text_input("Enter your text query:", value="Bucket")

    if query:
        response = requests.post(f"{base_url}/text_search", params={"text_query": query})

        if response.status_code == 200:
            images = response.json()["images"]
            cols = st.columns(5)
            for i, img_info in enumerate(images):
                with cols[i % 5]:
                    st.image(base_url + img_info["image_url"], use_column_width=True)
        else:
            st.error("❌ Failed to retrieve images from FastAPI.")

# --- IMAGE SEARCH ---
with tab2:
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        response = requests.post(
            f"{base_url}/image_search",
            files={"image": uploaded_image}
        )

        if response.status_code == 200:
            images = response.json()["images"]
            cols = st.columns(5)
            for i, img_info in enumerate(images):
                with cols[i % 5]:
                    st.image(base_url + img_info["image_url"], use_column_width=True)
        else:
            st.error("❌ Failed to retrieve similar images from FastAPI.")
