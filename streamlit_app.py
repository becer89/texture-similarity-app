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

# ✅ Initialize directories
folder_path = './images_database/'
os.makedirs(folder_path, exist_ok=True)
features_file = './image_features.pkl'
update_info_file = './last_update.txt'

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
st.title("🖼️ Local Texture Similarity Search App (Private)")
st.markdown(f"**Last Database Update:** {last_update}")

# ✅ Update database from local folder
st.sidebar.header("Update Local Database")
if st.sidebar.button("Update Database"):
    updated = False
    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')) and img_name not in image_features:
            img_path = os.path.join(folder_path, img_name)
            features = extract_features(img_path)
            image_features[img_name] = features
            updated = True

    if updated:
        with open(features_file, 'wb') as f:
            pickle.dump(image_features, f)
        last_update = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(update_info_file, 'w') as f:
            f.write(last_update)
        st.sidebar.success(f"Database updated successfully. Last update: {last_update}")
    else:
        st.sidebar.info("No new images found to update.")

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
    similar_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]

    st.image(query_img_path, caption="Query Image", use_column_width=True)
    st.subheader("Top 5 Similar Textures")
    for filename, similarity in similar_images:
        img_path = os.path.join(folder_path, filename)
        with Image.open(img_path) as img:
            st.image(img.copy(), caption=f"{filename} - Similarity: {similarity:.2f}", use_column_width=True)
