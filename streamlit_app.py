import os
import pickle
import datetime
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import requests
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ✅ Google Drive Folder ID
FOLDER_ID = '1vRb-LrIrEtcxDsV_QllDeCnp9YqZDQ-D'

# ✅ Initialize directories
features_file = 'image_features.pkl'
update_info_file = 'last_update.txt'

# ✅ Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


# ✅ Google Drive API Authentication
def authenticate_google():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    service = build('drive', 'v3', credentials=credentials)
    return service


# ✅ Function to extract features from an image in memory
def extract_features_from_bytes(img_bytes):
    img = Image.open(BytesIO(img_bytes)).convert('RGB').resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features.flatten()


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

# ✅ Streamlit UI
st.title("🖼️ Google Drive Texture Similarity Search App")
st.markdown(f"**Last Database Update:** {last_update}")

# ✅ Authenticate Google Drive
service = authenticate_google()


# ✅ Retrieve file links from Google Drive API
def get_google_drive_file_links(folder_id):
    file_links = {}
    results = service.files().list(q=f"'{folder_id}' in parents and mimeType contains 'image/' and trashed = false",
                                   fields="files(id, name)").execute()
    items = results.get('files', [])
    for item in items:
        file_links[item['name']] = f"https://drive.google.com/uc?export=download&id={item['id']}"
    return file_links


# ✅ Download images on-the-fly from Google Drive
def process_images_from_drive():
    file_links = get_google_drive_file_links(FOLDER_ID)
    new_files = []
    for img_name, img_url in file_links.items():
        if img_name not in image_features:
            try:
                response = requests.get(img_url)
                if response.status_code == 200:
                    features = extract_features_from_bytes(response.content)
                    image_features[img_name] = features
                    new_files.append(img_name)
            except Exception as e:
                st.sidebar.error(f"Error processing {img_name}: {str(e)}")
    if new_files:
        with open(features_file, 'wb') as f:
            pickle.dump(image_features, f)
        global last_update
        last_update = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(update_info_file, 'w') as f:
            f.write(last_update)
        st.sidebar.success(f"Database updated successfully with {len(new_files)} new images.")
    else:
        st.sidebar.info("No new images found to update.")


# ✅ Compare number of files in database vs Google Drive folder
st.sidebar.header("Update Google Drive Database")
file_links = get_google_drive_file_links(FOLDER_ID)
st.sidebar.markdown(f"**Files in Google Drive folder:** {len(file_links)}")
st.sidebar.markdown(f"**Files in local database:** {len(image_features)}")

if st.sidebar.button("Update Database"):
    process_images_from_drive()

# ✅ Image comparison
st.header("Find Similar Textures")
uploaded_query = st.file_uploader("Upload an image to compare", type=["png", "jpg", "jpeg"])

if uploaded_query is not None:
    query_img_path = "./temp_query_image.png"
    with open(query_img_path, 'wb') as f:
        f.write(uploaded_query.read())

    with open(query_img_path, 'rb') as f:
        img_bytes = f.read()

    comparison_features = extract_features_from_bytes(img_bytes)
    similarities = {}

    for filename, features in image_features.items():
        similarity = cosine_similarity([comparison_features], [features])[0][0]
        similarities[filename] = similarity

    # Sort by similarity score (highest first)
    similar_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:4]

    st.image(query_img_path, caption="Query Image", use_container_width=True)
    st.subheader("Top 4 Similar Textures")
    cols = st.columns(4)  # Display 4 images in a single row

    for i, (filename, similarity) in enumerate(similar_images):
        img_url = file_links.get(filename, None)
        if img_url:
            response = requests.get(img_url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                cols[i].image(img, caption=f"{filename} - Similarity: {similarity:.2f}", use_container_width=True)
