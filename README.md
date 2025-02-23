# 🖼️ Texture Similarity Search App

This app allows users to upload 3D texture images and search for visually similar images using a ResNet50 feature extraction model.

## 🚀 Features
- Upload textures and update the database.
- Find the 5 most similar textures from the database.
- Automatically tracks the last update date.

## 🔧 Requirements
- streamlit
- tensorflow
- pillow
- scikit-learn
- numpy

## ✅ How to Run Locally
1. Clone the repository:
git clone https://github.com/your-username/texture-similarity-app.git cd texture-similarity-app
2. Install dependencies:
pip install -r requirements.txt
3. Run the application:
streamlit run streamlit_app.py