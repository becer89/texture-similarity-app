import os
import pickle
import datetime
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# ✅ Google Drive Folder ID
FOLDER_ID = '1vRb-LrIrEtcxDsV_QllDeCnp9YqZDQ-D'

# ✅ Initialize directories
features_file = 'image_features.pkl'
update_info_file = 'last_update.txt'


# ✅ Authenticate Google Drive
def authenticate_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Opens a browser for Google login
    drive = GoogleDrive(gauth)
    return drive


# ✅ Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


# ✅ Function to extract features from an image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
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
st.sidebar.header("Google Drive Authentication")
if st.sidebar.button("Login to Google Drive"):
    drive = authenticate_drive()
    st.sidebar.success("Successfully authenticated with Google Drive!")
else:
    st.sidebar.warning("Please authenticate to access Google Drive.")
    st.stop()


# ✅ Download images from Google Drive folder
def download_images_from_drive(folder_id, drive):
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    downloaded_files = []
    for file in file_list:
        img_name = file['title']
        img_path = os.path.join('temp_images', img_name)
        if img_name not in image_features:
            file.GetContentFile(img_path)
            features = extract_features(img_path)
            image_features[img_name] = features
            downloaded_files.append(img_name)
    if downloaded_files:
        with open(features_file, 'wb') as f:
            pickle.dump(image_features, f)
        global last_update
        last_update = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(update_info_file, 'w') as f:
            f.write(last_update)
        st.sidebar.success(f"Database updated successfully with {len(downloaded_files)} new images.")
    else:
        st.sidebar.info("No new images found to update.")


# ✅ Update database from Google Drive
st.sidebar.header("Update Google Drive Database")
if st.sidebar.button("Update Database"):
    os.makedirs('temp_images', exist_ok=True)
    download_images_from_drive(FOLDER_ID, drive)

# ✅ Image comparison
st.header("Find Similar Textures")
uploaded_query = st.file_uploader("Upload an image to compare", type=["png", "jpg", "jpeg"])

if uploaded_query is not None:
    query_img_path = "./temp_query_image.png"
    with open(query_img_path, 'wb') as f:
        f.write(uploaded_query.read())

    comparison_features = extract_features(query_img_path)
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
        img_path = os.path.join('temp_images', filename)
        with Image.open(img_path) as img:
            cols[i].image(img.copy(), caption=f"{filename} - Similarity: {similarity:.2f}", use_container_width=True)
