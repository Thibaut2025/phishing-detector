import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
import joblib
import os

# Fonction pour extraire les caractéristiques de l'URL
def extract_features(url):
    # Longueur de l'URL
    url_length = len(url)
    
    # Nombre de sous-domaines
    subdomains = len(url.split('.')) - 2 if 'www.' in url else len(url.split('.')) - 1
    
    # Présence de caractères suspects
    suspicious_chars = len(re.findall(r'[-@%]', url))
    
    # Tokenisation de l'URL pour TF-IDF
    tokens = re.split(r'[\W_]', url.lower())
    tokens = [token for token in tokens if token]
    
    return url_length, subdomains, suspicious_chars, tokens

# Charger le modèle et le vectoriseur
@st.cache_resource
def load_model_and_vectorizer():
    try:
        clf = joblib.load('clf.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return clf, vectorizer
    except FileNotFoundError:
        st.error("OK Les fichiers 'clf.pkl' ou 'tfidf_vectorizer.pkl' sont introuvables. Assurez-vous qu'ils sont dans le même répertoire que ce script.")
        return None, None

# Fonction principale de l'application
def main():
    st.title("Détecteur de Phishing URL")
    st.write("Entrez une URL pour vérifier si elle est légitime ou un lien de phishing.")

    # Champ de saisie pour l'URL
    url_input = st.text_input("Entrez l'URL", placeholder="https://example.com")
    
    if st.button("Vérifier"):
        if url_input:
            # Extraire les caractéristiques
            url_length, subdomains, suspicious_chars, tokens = extract_features(url_input)
            
            # Charger le modèle et le vectoriseur
            clf, vectorizer = load_model_and_vectorizer()
            
            if clf is not None and vectorizer is not None:
                # Transformer les tokens avec TF-IDF
                url_text = ' '.join(tokens)
                tfidf_features = vectorizer.transform([url_text])
                
                # Combiner les caractéristiques
                numerical_features = np.array([[url_length, subdomains, suspicious_chars]])
                combined_features = np.hstack([tfidf_features, numerical_features])
                
                # Faire la prédiction
                prediction = clf.predict(combined_features)[0]
                probability = clf.predict_proba(combined_features)[0]
                
                # Afficher les résultats
                if prediction == 0:
                    st.success("L'URL semble **légitime**.")
                    st.write(f"Probabilité de légitimité : {probability[0]*100:.2f}%")
                else:
                    st.error("L'URL semble être un **lien de phishing**.")
                    st.write(f"Probabilité de phishing : {probability[1]*100:.2f}%")
        else:
            st.warning("Veuillez entrer une URL valide.")

if __name__ == "__main__":
    main()