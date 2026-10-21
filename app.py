
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

st.caption(f"Version de Python utilisée : {sys.version.split()[0]}")

# ============================================================
# 2. CHEMIN ET CHARGEMENT DU MODELE
# ============================================================

MODEL_PATH = Path(__file__).parent / "modele_classification.pkl"


@st.cache_resource
def charger_modele(chemin):
    donnees = joblib.load(chemin)

    if not isinstance(donnees, dict):
        raise ValueError(
            "Le fichier PKL doit contenir un dictionnaire."
        )

    if "model" not in donnees or "class_names" not in donnees:
        raise ValueError(
            "Le fichier PKL doit contenir les clés "
            "'model' et 'class_names'."
        )

    modele = donnees["model"]
    class_names = donnees["class_names"]
    nom_modele = donnees.get("model_name", "Modèle inconnu")

    return modele, class_names, nom_modele


if not MODEL_PATH.exists():
    st.error(
        f"Fichier du modèle introuvable : {MODEL_PATH.name}"
    )
    st.info(
        "Vérifiez que le fichier PKL est présent dans le dépôt "
        "GitHub et que son nom correspond exactement."
    )
    st.stop()

try:
    modele, class_names, nom_modele = charger_modele(
        str(MODEL_PATH)
    )

    st.success("Modèle chargé correctement.")
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

    IMPORTANT :
    cette fonction doit être identique à celle utilisée
    pendant l'entraînement du modèle.
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
        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ],
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
        gray,
        P=8,
        R=1,
        method="uniform"
    )

    lbp_h, _ = np.histogram(
        lbp,
        bins=np.arange(0, 11),
        density=True
    )

    # 5. Moments de Hu : 7 caractéristiques
    hu = cv2.HuMoments(
        cv2.moments(gray)
    ).ravel()

    hu = -np.sign(hu) * np.log10(
        np.abs(hu) + 1e-12
    )

    features = np.concatenate([
        rgb,
        hist,
        glcm_f,
        lbp_h,
        hu
    ])

    return features


# ============================================================
# 4. FONCTION POUR OBTENIR LE NOM D'UNE CLASSE
# ============================================================

def obtenir_nom_classe(classe):
    if isinstance(class_names, dict):
        if classe in class_names:
            return str(class_names[classe])

        if str(classe) in class_names:
            return str(class_names[str(classe)])

        return str(classe)

    try:
        return str(class_names[int(classe)])
    except (ValueError, TypeError, IndexError):
        return str(classe)


# ============================================================
# 5. IMPORTATION ET AFFICHAGE DE L'IMAGE
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

        st.write("Étape 2 : image chargée correctement.")

    except Exception as e:
        st.error("Impossible de lire cette image.")
        st.exception(e)
        st.stop()

    # ========================================================
    # 6. CLASSIFICATION
    # ========================================================

    if st.button("Classer l'image", type="primary"):

        try:
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
                    f"Le modèle attend 67 caractéristiques, "
                    f"mais {caracteristiques.shape[1]} ont été calculées."
                )
                st.stop()

            st.write("Étape 5 : prédiction.")

            prediction = modele.predict(caracteristiques)[0]

            nom_prediction = obtenir_nom_classe(prediction)

            st.success(
                f"Classe prédite : **{nom_prediction}**"
            )

            # =================================================
            # 7. DIAGNOSTIC DU MODELE
            # =================================================

            with st.expander("Diagnostic du modèle", expanded=True):

                st.write("Nom du modèle :", nom_modele)
                st.write(
                    "Type du modèle :",
                    type(modele).__name__
                )
                st.write(
                    "Prédiction brute :",
                    str(prediction)
                )
                st.write(
                    "Classes affichées :",
                    str(class_names)
                )

                if hasattr(modele, "classes_"):
                    st.write(
                        "Classes internes :",
                        modele.classes_.tolist()
                    )

                if hasattr(modele, "n_neighbors"):
                    st.write(
                        "Nombre de voisins KNN :",
                        modele.n_neighbors
                    )

                if hasattr(modele, "named_steps"):
                    st.write(
                        "Étapes du pipeline :",
                        list(modele.named_steps.keys())
                    )

                # =============================================
                # 8. PROBABILITES
                # =============================================

                if hasattr(modele, "predict_proba"):

                    probabilites = modele.predict_proba(
                        caracteristiques
                    )[0]

                    if hasattr(modele, "classes_"):
                        classes_modele = modele.classes_
                    else:
                        classes_modele = range(
                            len(probabilites)
                        )

                    st.subheader("Résultats par classe")

                    for classe, proba in zip(
                        classes_modele,
                        probabilites
                    ):
                        nom = obtenir_nom_classe(classe)

                        st.write(
                            f"{nom} : {proba * 100:.2f} %"
                        )

                        st.progress(float(proba))

                    st.caption(
                        "Les probabilités représentent les scores "
                        "fournis par le modèle. Pour KNN, elles "
                        "correspondent généralement aux proportions "
                        "de voisins de chaque classe."
                    )

                else:
                    st.info(
                        "Ce modèle ne fournit pas de probabilités "
                        "avec predict_proba()."
                    )

            st.write("Étape 6 : classification terminée.")

        except Exception as e:
            st.error(
                "Une erreur est survenue pendant la classification."
            )
            st.exception(e)
