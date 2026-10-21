
import streamlit as st
import numpy as np
import cv2
import joblib
import sys

from pathlib import Path
from PIL import Image
from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)

# ============================================================
# 1. CONFIGURATION DE LA PAGE
# ============================================================

st.set_page_config(
    page_title="Classification d'images",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Classification d'images")
st.write(
    "Importez une image pour déterminer si elle représente "
    "un bateau ou une voiture."
)

st.caption(f"Version de Python : {sys.version.split()[0]}")

# ============================================================
# 2. CHARGEMENT DU MODELE
# ============================================================

MODEL_PATH = Path(__file__).parent / "modele_classification.pkl"


@st.cache_resource
def charger_modele(chemin):
    donnees = joblib.load(chemin)

    if not isinstance(donnees, dict):
        raise ValueError(
            "Le fichier du modèle ne contient pas le dictionnaire attendu."
        )

    model = donnees.get("model")
    class_names = donnees.get("class_names")
    model_name = donnees.get("model_name", "Modèle inconnu")

    if model is None or class_names is None:
        raise ValueError(
            "Le fichier doit contenir les clés 'model' et 'class_names'."
        )

    return model, class_names, model_name


try:
    if not MODEL_PATH.exists():
        st.error(f"Fichier modèle introuvable : {MODEL_PATH.name}")
        st.info(
            "Vérifiez le nom du fichier sur GitHub. "
            "Il doit correspondre exactement à celui indiqué dans le code."
        )
        st.stop()

    modele, class_names, nom_modele = charger_modele(str(MODEL_PATH))

    st.success("Modèle chargé avec succès.")
    st.caption(f"Modèle utilisé : {nom_modele}")

except Exception as e:
    st.error("Impossible de charger le modèle.")
    st.exception(e)
    st.stop()

# ============================================================
# 3. EXTRACTION DES 67 CARACTERISTIQUES
# ============================================================

def extract_features(img):
    """
    img : image RGB de taille 32x32, valeurs entre 0 et 1.
    Retourne 67 caractéristiques.
    """

    img8 = np.clip(img * 255, 0, 255).astype(np.uint8)

    # 1. Statistiques RGB : 6 caractéristiques
    rgb = np.concatenate([
        img.mean(axis=(0, 1)),
        img.std(axis=(0, 1))
    ])

    # 2. Histogramme HSV : 24 caractéristiques
    hsv = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV)

    hist = np.concatenate([
        cv2.calcHist(
            [hsv], [0], None, [8], [0, 180]
        ).ravel(),

        cv2.calcHist(
            [hsv], [1], None, [8], [0, 256]
        ).ravel(),

        cv2.calcHist(
            [hsv], [2], None, [8], [0, 256]
        ).ravel()
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
    lbp = local_binary_pattern(
        gray, P=8, R=1, method="uniform"
    )

    lbp_h, _ = np.histogram(
        lbp,
        bins=np.arange(0, 12),
        density=True
    )

    # Conserver 10 caractéristiques comme dans le modèle attendu
    lbp_h = lbp_h[:10]

    # 5. Moments de Hu : 7 caractéristiques
    hu = cv2.HuMoments(cv2.moments(gray)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    features = np.concatenate([
        rgb, hist, glcm_f, lbp_h, hu
    ])

    return features


# ============================================================
# 4. IMPORTATION DE L'IMAGE
# ============================================================

fichier = st.file_uploader(
    "Choisissez une image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if fichier is not None:

    st.write("Étape 1 : image importée.")

    try:
        image_pil = Image.open(fichier).convert("RGB")

        st.image(
            image_pil,
            caption="Image importée",
            use_container_width=True
        )

        st.write("Étape 2 : image chargée.")

        if st.button("Classer l'image", type="primary"):

            st.write("Étape 3 : préparation de l'image.")

            img = np.array(image_pil)
            img = cv2.resize(img, (32, 32))
            img = img.astype(np.float32) / 255.0

            st.write("Étape 4 : extraction des caractéristiques.")

            caracteristiques = extract_features(img).reshape(1, -1)

            st.write(
                "Nombre de caractéristiques :",
                caracteristiques.shape[1]
            )

            if caracteristiques.shape[1] != 67:
                st.error(
                    "Le nombre de caractéristiques est incorrect. "
                    "Le modèle attend 67 caractéristiques."
                )
                st.stop()

            st.write("Étape 5 : calcul de la prédiction.")

            prediction = modele.predict(caracteristiques)[0]

            st.write("Valeur prédite :", str(prediction))
            st.write("Classes du fichier :", str(class_names))

            # Traduire la valeur prédite en nom de classe
            if isinstance(class_names, dict):
                nom_classe = class_names.get(
                    prediction,
                    class_names.get(str(prediction), str(prediction))
                )
            else:
                try:
                    nom_classe = class_names[int(prediction)]
                except (ValueError, TypeError, IndexError):
                    nom_classe = str(prediction)

            st.success(f"Classe prédite : **{nom_classe}**")

            # Probabilités de prédiction, si disponibles
            if hasattr(modele, "predict_proba"):

                probabilites = modele.predict_proba(
                    caracteristiques
                )[0]

                st.subheader("Résultats par classe")

                if hasattr(modele, "classes_"):
                    classes_modele = modele.classes_
                else:
                    classes_modele = range(len(probabilites))

                for classe, proba in zip(
                    classes_modele, probabilites
                ):
                    if isinstance(class_names, dict):
                        nom = class_names.get(
                            classe,
                            class_names.get(str(classe), str(classe))
                        )
                    else:
                        try:
                            nom = class_names[int(classe)]
                        except (ValueError, TypeError, IndexError):
                            nom = str(classe)

                    st.write(f"{nom} : {proba * 100:.1f} %")
                    st.progress(float(proba))

                st.caption(
                    "Les probabilités dépendent du modèle. "
                    "Pour KNN, elles représentent généralement "
                    "la proportion de voisins de chaque classe."
                )

            st.write("Étape 6 : classification terminée.")

    except Exception as e:
        st.error("Une erreur est survenue pendant la classification.")
        st.exception(e)
