
import streamlit as st
import numpy as np
import cv2
import joblib

from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# Configuration de la page
st.set_page_config(
    page_title="Classification d'images",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Classification d'images")
st.write("Importez une image pour déterminer si elle représente un bateau ou une voiture.")


# Charger le modèle déjà entraîné
@st.cache_resource
def charger_modele():
    donnees = joblib.load("modele_classification.pkl")
    return donnees["model"], donnees["class_names"], donnees["model_name"]


# Extraction des 67 caractéristiques, comme dans le notebook
def extract_features(img):
    img8 = (img * 255).astype(np.uint8)

    # 1. Statistiques RGB : 6 caractéristiques
    rgb = np.concatenate([
        img.mean(axis=(0, 1)),
        img.std(axis=(0, 1))
    ])

    # 2. Histogramme HSV : 24 caractéristiques
    hsv = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV)

    hist = np.concatenate([
        cv2.calcHist([hsv], [0], None, [8], [0, 180]).ravel(),
        cv2.calcHist([hsv], [1], None, [8], [0, 256]).ravel(),
        cv2.calcHist([hsv], [2], None, [8], [0, 256]).ravel()
    ])

    hist = hist / (hist.sum() + 1e-8)

    # Image en niveaux de gris
    gray = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)

    # 3. GLCM : 20 caractéristiques
    glcm = graycomatrix(
        gray // 8,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True
    )

    glcm_f = np.concatenate([
        graycoprops(glcm, prop).ravel()
        for prop in [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation"
        ]
    ])

    # 4. LBP : 10 caractéristiques
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    lbp_h, _ = np.histogram(
        lbp,
        bins=np.arange(0, 11),
        density=True
    )

    # 5. Moments de Hu : 7 caractéristiques
    hu = cv2.HuMoments(cv2.moments(gray)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return np.concatenate([rgb, hist, glcm_f, lbp_h, hu])


# Charger le modèle
try:
    modele, class_names, nom_modele = charger_modele()
except Exception as e:
    st.error(f"Impossible de charger le modèle : {e}")
    st.stop()


st.caption(f"Modèle utilisé : {nom_modele}")

# Importer une image
fichier = st.file_uploader(
    "Choisissez une image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if fichier is not None:
    try:
        image = Image.open(fichier).convert("RGB")
        st.image(image, caption="Image importée", use_container_width=True)

        if st.button("Classer l'image", type="primary"):
            # Reproduire le prétraitement du notebook
            img = np.array(image)
            img = cv2.resize(img, (32, 32))
            img = img.astype("float32") / 255.0

            # Extraire les caractéristiques
            caracteristiques = extract_features(img).reshape(1, -1)

            # Vérifier le nombre de caractéristiques
            if caracteristiques.shape[1] != 67:
                st.error(
                    f"Nombre de caractéristiques incorrect : "
                    f"{caracteristiques.shape[1]} au lieu de 67."
                )
                st.stop()

            # Prédiction
            prediction = modele.predict(caracteristiques)[0]
            st.success(f"Classe prédite : **{class_names[int(prediction)]}**")

            # Probabilités fournies par le KNN
            if hasattr(modele, "predict_proba"):
                probabilites = modele.predict_proba(caracteristiques)[0]
                classes_modele = modele.named_steps["knn"].classes_

                st.subheader("Résultats par classe")

                for classe, proba in zip(classes_modele, probabilites):
                    nom_classe = class_names[int(classe)]
                    st.write(f"{nom_classe} : {proba * 100:.1f} %")
                    st.progress(float(proba))

                st.caption(
                    "Ces pourcentages représentent les proportions de voisins "
                    "dans le KNN, et non une certitude garantie."
                )

    except Exception as e:
        st.error(f"Une erreur est survenue pendant la prédiction : {e}")
