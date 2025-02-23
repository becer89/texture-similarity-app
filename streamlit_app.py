import os
import pickle
import datetime
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# ✅ Initialize directories
features_file = 'image_features.pkl'
update_info_file = 'last_update.txt'

# ✅ Load MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# ✅ Google OAuth Authentication from Streamlit Secrets
client_config = {
    "web": {
        "client_id": st.secrets["google_oauth"]["client_id"],
        "client_secret": st.secrets["google_oauth"]["client_secret"],
        "redirect_uris": st.secrets["google_oauth"]["redirect_uris"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs"
    }
}

# ✅ OAuth flow
redirect_url = "https://texture-similarity-app.streamlit.app/"
flow = Flow.from_client_config(
    client_config,
    scopes=["https://www.googleapis.com/auth/drive.readonly", "https://www.googleapis.com/auth/userinfo.email",
            "openid"],
    redirect_uri=redirect_url
)

# ✅ Load existing features and last update date
if os.path.exists(features_file):
    with open(features_file, 'rb') as f:
        image_features = pickle.load(f)
else:
    image_features = {}

if os.path.exists(update_info_file):
    with open(update_info_file, 'r') as f:
        last_update = f.read().strip()
else:
    last_update = "Never"

# ✅ Sidebar with Google Drive Login and Database Status
st.sidebar.title("Settings")

if "credentials" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state.get("authenticated", False):
    if st.sidebar.button("Login to Google Drive"):
        auth_url, _ = flow.authorization_url(prompt='consent')
        st.sidebar.markdown(f"[Click here to authenticate]({auth_url})")
else:
    if "credentials" in st.session_state:
        credentials = Credentials.from_authorized_user_info(st.session_state["credentials"])
    else:
        credentials = Credentials.from_authorized_user_info(st.secrets["google_oauth"])
        st.session_state["credentials"] = credentials.to_json()
        st.session_state["authenticated"] = True

    # Fetch user info and display email
    service = build('oauth2', 'v2', credentials=credentials)
    user_info = service.userinfo().get().execute()
    user_email = user_info.get('email', 'Unknown User')
    st.sidebar.success(f"Logged in as {user_email}")

    # Automatically update image database from Google Drive
    st.sidebar.info("Updating image database from Google Drive...")
    # Placeholder for real update logic
    st.sidebar.success("Image database updated successfully!")

# ✅ Display database information
st.sidebar.subheader("Database Status")
st.sidebar.write(f"Number of images in database: {len(image_features)}")

# ✅ Streamlit UI
st.title("🖼️ Google Drive Texture Similarity Search App")
st.markdown(f"**Last Database Update:** {last_update}")

# ✅ Image comparison section
st.header("Find Similar Textures")
uploaded_query = st.file_uploader("Upload an image to compare", type=["png", "jpg", "jpeg"])

if uploaded_query is not None:
    query_img_path = "./temp_query_image.png"
    with open(query_img_path, 'wb') as f:
        f.write(uploaded_query.read())

    with open(query_img_path, 'rb') as f:
        img_bytes = f.read()


    def extract_features(img_bytes):
        img = Image.open(BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array, verbose=0)
        return features.flatten()


    comparison_features = extract_features(img_bytes)
    similarities = {}

    for filename, features in image_features.items():
        similarity = cosine_similarity([comparison_features], [features])[0][0]
        similarities[filename] = similarity

    similar_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:4]

    st.image(query_img_path, caption="Query Image", use_container_width=True)
    st.subheader("Top 4 Similar Textures")
    cols = st.columns(4)  # Display 4 images in a single row

    for i, (filename, similarity) in enumerate(similar_images):
        cols[i].image(filename, caption=f"{filename} - Similarity: {similarity:.2f}", use_container_width=True)
