import os
import pickle
import datetime
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from urllib.parse import urlparse, parse_qs

# ✅ Initialize directories
features_file = 'image_features.pkl'
update_info_file = 'last_update.txt'

# ✅ Load ResNet50 model using PyTorch
model = models.resnet50(pretrained=True)
model.eval()

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

# ✅ OAuth flow with proper redirect URI
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

# ✅ Check for authorization code in the URL
query_params = st.query_params
if "code" in query_params and not st.session_state.get("authenticated", False):
    authorization_code = query_params["code"][0]
    try:
        flow.fetch_token(code=authorization_code)
    credentials = flow.credentials
    st.session_state["credentials"] = credentials.to_json()
    st.session_state["authenticated"] = True
    st.success("✅ Successfully authenticated!")
except Exception as e:
st.error(f"Authentication failed: {str(e)}. Please try logging in again.")
if st.sidebar.button("Retry Login"):
    auth_url, _ = flow.authorization_url(prompt='consent')
    st.sidebar.markdown(f'<a href="{auth_url}" target="_blank">Click here to re-authenticate</a>',
                        unsafe_allow_html=True)
credentials = flow.credentials
st.session_state["credentials"] = credentials.to_json()
st.session_state["authenticated"] = True
st.success("✅ Successfully authenticated!")

if not st.session_state.get("authenticated", False):
    if st.sidebar.button("Login to Google Drive"):
        auth_url, _ = flow.authorization_url(prompt='consent')
        st.sidebar.markdown(f'<a href="{auth_url}" target="_blank">Click here to authenticate</a>',
                            unsafe_allow_html=True)
else:
    credentials = Credentials.from_authorized_user_info(eval(st.session_state["credentials"]))

    # Fetch user info and display email
    service = build('oauth2', 'v2', credentials=credentials)
    user_info = service.userinfo().get().execute()
    user_email = user_info.get('email', 'Unknown User')
    st.sidebar.success(f"Logged in as {user_email}")

# ✅ Check if user is authenticated before allowing access to functionality
if st.session_state.get("authenticated", False):
    st.sidebar.subheader("Database Status")
    st.sidebar.write(f"Number of images in database: {len(image_features)}")


    # ✅ Function to extract features using PyTorch ResNet50
    def extract_features_pytorch(img_bytes):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            features = model(img_tensor)
        return features.flatten().numpy()


    # ✅ Streamlit UI
    st.title("🖼️ Google Drive Texture Similarity Search App - PyTorch Edition")
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

        comparison_features = extract_features_pytorch(img_bytes)
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
else:
    st.sidebar.warning("Please log in to access Google Drive functionality.")
    st.write("🔒 Authentication required to access functionality.")
