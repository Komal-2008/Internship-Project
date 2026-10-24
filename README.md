# ♻️ Smart Waste Classification System

AI-powered web app that classifies waste images into 10 categories using a
CNN (Convolutional Neural Network) trained with TensorFlow/Keras, served
through a Streamlit frontend.

## 🗂️ Waste Categories
Cardboard, E-Waste, Food Organics, Glass, Metal, Miscellaneous Trash,
Paper, Plastic, Textile Trash, Vegetation

## 🚀 Live Demo
_(add your Streamlit Cloud link here after deployment)_

## 📁 Project Structure
```
smart-waste-classifier/
├── app.py                          # Streamlit frontend + prediction logic
├── smart_waste_classifier.keras    # Trained CNN model (tracked via Git LFS)
├── class_names.pkl                 # Pickled list of class names
├── requirements.txt                # Python dependencies
├── .gitignore
├── .gitattributes                  # ⚠️ DO NOT create manually — this is auto-generated
│                                    #    when you run: git lfs track "*.keras"
├── .streamlit/
│   └── config.toml                 # App theme
└── training/
    └── train_model.py              # Script used to train the CNN (not needed to run the app)
```

## ⚠️ Important Note on Git LFS
The trained model (`smart_waste_classifier.keras`) is **147MB**, which
exceeds GitHub's 100MB limit for normal files. This repo uses **Git LFS**
to handle it. Before pushing, run:
```bash
git lfs install
git lfs track "*.keras"
```
This automatically creates a `.gitattributes` file — **do not create it
manually**, and do not edit it by hand.

## 🛠️ Tech Stack
- Python
- Streamlit
- TensorFlow / Keras
- NumPy / Pillow

## ▶️ Run Locally
```bash
git clone https://github.com/Komal-2008/smart-waste-classifier.git
cd smart-waste-classifier
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Note
This app is built for educational purposes and should not be used as a
substitute for professional waste-management advice.
=======
# Internship-Project
