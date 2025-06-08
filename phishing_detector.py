
import streamlit as st
import pandas as pd
import joblib
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlparse
import re
from bs4 import BeautifulSoup
import requests
import os
import traceback

# Titre
st.title("Détecteur d'URLs de Phishing")
st.write("Entrez une URL pour vérifier si elle est légitime ou malveillante.")

# Charger le modèle et le scaler
model_path = 'C:\\Users\\Thibaut\\Documents\\MLProject\\url_pishing_classification\\xgboost_model.pkl'
scaler_path = 'C:\\Users\\Thibaut\\Documents\\MLProject\\url_pishing_classification\\minmax_scaler.pkl'

try:
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        # st.error("Erreur : Fichiers modèle ou scaler non trouvés.")
        model_path = 'xgboost_model.pkl'
        scaler_path = 'minmax_scaler.pkl'
        st.stop()   
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.success("Modèle et scaler chargés avec succès !")
except FileNotFoundError:
    st.error("Erreur : Fichiers modèle ou scaler non trouvés.")
    st.stop()

# Fonction extract_features
def extract_features(url):
    features = {}

    # Analyser l'URL
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ''
    path = parsed_url.path or ''
    query = parsed_url.query or ''

    # Fonction de secours pour récupérer le HTML statique
    def get_html_fallback(url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, timeout=10, headers=headers)
            return response.text
        except Exception as e:
            print(f"Erreur dans get_html_fallback pour {url}: {str(e)}")
            return ""


    # Configurer Selenium pour Colab
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--remote-debugging-port=9222')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-setuid-sandbox')

    driver = None
    
    try:
        # Initialiser ChromeDriver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.set_page_load_timeout(60)
        driver.get(url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "form")))
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
    
        if not html_content.strip():  # Vérifier si le contenu HTML est vide
            print(f"Contenu HTML vide récupéré pour {url} via Selenium, passage au fallback.")
            html_content = get_html_fallback(url)
            soup = BeautifulSoup(html_content, 'html.parser') if html_content else BeautifulSoup("", 'html.parser')
    except Exception as e:
        print(f"Erreur Selenium pour {url}: {e}")
        # Utiliser la solution de secours avec requests
        html_content = get_html_fallback(url)
        soup = BeautifulSoup(html_content, 'html.parser') if html_content else BeautifulSoup("", 'html.parser')
    finally:
        if driver is not None:
            driver.quit()

    try:
        # Extraire les caractéristiques
        features['NumDots'] = url.count('.')
        features['SubdomainLevel'] = len(hostname.split('.')) - 2 if hostname else 0
        features['PathLevel'] = len([p for p in path.split('/') if p])
        features['UrlLength'] = len(url)
        features['NumDash'] = url.count('-')
        features['NumDashInHostname'] = hostname.count('-')
        features['AtSymbol'] = 1 if '@' in url else 0
        features['TildeSymbol'] = 1 if '~' in url else 0
        features['NumUnderscore'] = url.count('_')
        features['NumPercent'] = url.count('%')
        features['NumQueryComponents'] = len(query.split('&')) if query else 0
        features['NumAmpersand'] = url.count('&')
        features['NumHash'] = 1 if '#' in url else 0
        features['NumNumericChars'] = sum(c.isdigit() for c in url)
        features['NoHttps'] = 0 if url.startswith('https') else 1
        features['RandomString'] = 1 if re.search(r'[a-zA-Z0-9]{20,}', url) else 0
        features['IpAddress'] = 1 if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', hostname) and all(0 <= int(x) <= 255 for x in hostname.split('.')) else 0 # Vérification IP
        # features['IpAddress'] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', hostname) else 0
        features['DomainInSubdomains'] = 1 if any(brand in hostname for brand in ['paypal', 'amazon', 'bank']) else 0
        features['DomainInPaths'] = 1 if any(brand in path for brand in ['paypal', 'amazon', 'bank']) else 0
        features['HttpsInHostname'] = 1 if 'https' in hostname.lower() else 0
        features['HostnameLength'] = len(hostname)
        features['PathLength'] = len(path)
        features['QueryLength'] = len(query)
        features['DoubleSlashInPath'] = 1 if '//' in path else 0
        features['NumSensitiveWords'] = sum(1 for word in ['login', 'password', 'bank'] if word in url.lower())
        features['EmbeddedBrandName'] = 1 if any(brand in url.lower() for brand in ['paypal', 'amazon', 'bank']) else 0

        # Caractéristiques basées sur la page
        links = soup.find_all('a', href=True)
        total_links = len(links)
        ext_links = len([l for l in links if urlparse(l['href']).hostname and urlparse(l['href']).hostname != hostname])
        features['PctExtHyperlinks'] = ext_links / total_links if total_links > 0 else 0

        resources = soup.find_all(['img', 'script', 'link'], src=True)
        total_resources = len(resources)
        ext_resources = len([r for r in resources if urlparse(r['src']).hostname and urlparse(r['src']).hostname != hostname])
        features['PctExtResourceUrls'] = ext_resources / total_resources if total_resources > 0 else 0

        favicon = soup.find('link', rel='icon') or soup.find('link', rel='shortcut icon')
        if favicon:
            print(f"Favicon URL pour {url}: {favicon['href']}")
        features['ExtFavicon'] = 1 if favicon and urlparse(favicon['href']).hostname and urlparse(favicon['href']).hostname != hostname else 0

        forms = soup.find_all('form')
        features['InsecureForms'] = 1 if any(f.get('action') and not (f.get('action').startswith('https') or not urlparse(f.get('action')).scheme) for f in forms) else 0
        # features['InsecureForms'] = 1 if any(f.get('action') and not f.get('action').startswith('https') for f in forms) else 0
        features['RelativeFormAction'] = 1 if any(f.get('action') and not urlparse(f.get('action')).hostname for f in forms) else 0
        features['ExtFormAction'] = 1 if any(f.get('action') and urlparse(f.get('action')).hostname and urlparse(f.get('action')).hostname != hostname for f in forms) else 0
        features['AbnormalFormAction'] = 1 if any(f.get('action') in [None, '', 'javascript:void(0)'] for f in forms) else 0

        null_links = len([l for l in links if l['href'] in ['', '#', 'javascript:void(0)']])
        features['PctNullSelfRedirectHyperlinks'] = null_links / total_links if total_links > 0 else 0

        features['FrequentDomainNameMismatch'] = 1 if total_links > 0 and ext_links / total_links > 0.5 else 0
        features['FakeLinkInStatusBar'] = 1 if any('onmouseover' in str(l).lower() for l in links) else 0
        features['RightClickDisabled'] = 1 if 'contextmenu' in html_content.lower() else 0
        features['PopUpWindow'] = 1 if 'window.open' in html_content.lower() else 0
        features['SubmitInfoToEmail'] = 1 if any(f.get('action', '').startswith('mailto:') for f in forms) else 0
        features['IframeOrFrame'] = 1 if soup.find_all(['iframe', 'frame']) else 0
        features['MissingTitle'] = 1 if not soup.title else 0
        features['ImagesOnlyInForm'] = 1 if any(len(f.find_all('img')) > 0 and len(f.find_all(['input', 'textarea'])) == 0 for f in forms) else 0

        features['SubdomainLevelRT'] = 1 if features['SubdomainLevel'] > 2 else -1 if features['SubdomainLevel'] == 0 else 0
        features['UrlLengthRT'] = 1 if features['UrlLength'] > 100 else -1 if features['UrlLength'] < 50 else 0
        features['PctExtResourceUrlsRT'] = 1 if features['PctExtResourceUrls'] > 0.5 else -1 if features['PctExtResourceUrls'] < 0.1 else 0
        features['AbnormalExtFormActionR'] = features['ExtFormAction']
        features['ExtMetaScriptLinkRT'] = 1 if features['PctExtResourceUrls'] > 0.5 else -1 if features['PctExtResourceUrls'] < 0.1 else 0
        features['PctExtNullSelfRedirectHyperlinksRT'] = 1 if features['PctNullSelfRedirectHyperlinks'] > 0.5 else -1 if features['PctNullSelfRedirectHyperlinks'] < 0.1 else 0

    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques pour {url}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        # Remplir avec des valeurs par défaut
        feature_names = [
            'NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash',
            'NumDashInHostname', 'AtSymbol', 'TildeSymbol', 'NumUnderscore',
            'NumPercent', 'NumQueryComponents', 'NumAmpersand', 'NumHash',
            'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress',
            'DomainInSubdomains', 'DomainInPaths', 'HttpsInHostname',
            'HostnameLength', 'PathLength', 'QueryLength', 'DoubleSlashInPath',
            'NumSensitiveWords', 'EmbeddedBrandName', 'PctExtHyperlinks',
            'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms',
            'RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction',
            'PctNullSelfRedirectHyperlinks', 'FrequentDomainNameMismatch',
            'FakeLinkInStatusBar', 'RightClickDisabled', 'PopUpWindow',
            'SubmitInfoToEmail', 'IframeOrFrame', 'MissingTitle',
            'ImagesOnlyInForm', 'SubdomainLevelRT', 'UrlLengthRT',
            'PctExtResourceUrlsRT', 'AbnormalExtFormActionR',
            'ExtMetaScriptLinkRT', 'PctExtNullSelfRedirectHyperlinksRT'
        ]
        for key in feature_names:
            features[key] = features.get(key, 0)

    return features
# Fonction predict_phishing
def predict_phishing(url, model, scaler):
    features = extract_features(url)
    feature_names = [
        'NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash',
        'NumDashInHostname', 'AtSymbol', 'TildeSymbol', 'NumUnderscore',
        'NumPercent', 'NumQueryComponents', 'NumAmpersand', 'NumHash',
        'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress',
        'DomainInSubdomains', 'DomainInPaths', 'HttpsInHostname',
        'HostnameLength', 'PathLength', 'QueryLength', 'DoubleSlashInPath',
        'NumSensitiveWords', 'EmbeddedBrandName', 'PctExtHyperlinks',
        'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms',
        'RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction',
        'PctNullSelfRedirectHyperlinks', 'FrequentDomainNameMismatch',
        'FakeLinkInStatusBar', 'RightClickDisabled', 'PopUpWindow',
        'SubmitInfoToEmail', 'IframeOrFrame', 'MissingTitle',
        'ImagesOnlyInForm', 'SubdomainLevelRT', 'UrlLengthRT',
        'PctExtResourceUrlsRT', 'AbnormalExtFormActionR',
        'ExtMetaScriptLinkRT', 'PctExtNullSelfRedirectHyperlinksRT'
    ]
    df = pd.DataFrame([features], columns=feature_names)
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)
    return url, 'Phishing' if pred == 1 else 'Légitime'

# Interface utilisateur
url_input = st.text_input("Entrez l'URL à tester :", placeholder="https://www.example.com")
if st.button("Vérifier l'URL"):
    if url_input:
        with st.spinner("Analyse de l'URL en cours..."):
            try:
                if not url_input.startswith(('http://', 'https://')):
                    url_input = 'https://' + url_input
                result = predict_phishing(url_input, model, scaler)
                url, label = result
                if label == 'Phishing':
                    st.error(f"⚠️ Attention : L'URL '{url}' semble être une URL de phishing !")
                else:
                    st.success(f"✅ L'URL {url} semble légitime.")
                results = pd.DataFrame([result], columns=['URL', 'Prédiction'])
                results_path = 'C:\\Users\\Thibaut\\Documents\\MLProject\\url_pishing_classification\\predictions.csv'
                if not os.path.exists(results_path):
                    results_path = 'predictions.csv'
                try:
                    existing_results = pd.read_csv(results_path)
                    results = pd.concat([existing_results, results], ignore_index=True)
                except FileNotFoundError:
                    pass
                results.to_csv(results_path, index=False)
                st.write("Résultat sauvegardé sur Google Drive.")
            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {str(e)}")
                st.error(traceback.format_exc())
    else:
        st.warning("Veuillez entrer une URL valide.")
  