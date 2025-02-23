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
folder_path = 'C:/images_database'
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

# ✅ Update database from local folder with enhanced debug info
st.sidebar.header("Update Local Database")
num_files_in_db = len(image_features)
st.sidebar.markdown(f"**Number of images in database:** {num_files_in_db}")

if st.sidebar.button("Update Database"):
    updated = False
    current_files = set(f.lower() for f in image_features.keys())  # Case-insensitive matching
    new_files = []

    # List all files in the directory and log them
    all_files = [f for f in os.listdir(folder_path) if
                 os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    st.sidebar.write(f"Found {len(all_files)} images in folder.")

    for img_name in all_files:
        if img_name.lower() not in current_files:
            img_path = os.path.join(folder_path, img_name)
            try:
                features = extract_features(img_path)
                image_features[img_name] = features
                new_files.append(img_name)
                updated = True
            except Exception as e:
                st.sidebar.error(f"Error processing {img_name}: {str(e)}")

    # Display debug information
    if new_files:
        st.sidebar.write(f"New files added: {', '.join(new_files)}")
    else:
        st.sidebar.info("No new images found to update.")

    # Save updated database
    if updated:
        with open(features_file, 'wb') as f:
            pickle.dump(image_features, f)
        last_update = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(update_info_file, 'w') as f:
            f.write(last_update)
        st.sidebar.success(f"Database updated successfully. Last update: {last_update}")

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
        img_path = os.path.join(folder_path, filename)
        with Image.open(img_path) as img:
            cols[i].image(img.copy(), caption=f"{filename} - Similarity: {similarity:.2f}", use_container_width=True)