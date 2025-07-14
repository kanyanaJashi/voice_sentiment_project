# Voice Sentiment Detector — Analyse des sentiments vocaux en anglais

## Objectif du projet
Ce projet vise à transcrire des appels vocaux, corriger automatiquement la grammaire, puis détecter le sentiment exprimé par l’interlocuteur (positif, négatif ou neutre). Il exploite des modèles de pointe en traitement automatique de la langue (TAL/NLP) et en reconnaissance vocale, orchestrés via une interface Gradio ou une API FastAPI.

---

## Choix de la langue anglaise
Bien que le contexte d’usage soit multilingue, **l'anglais a été choisi comme langue principale** pour ce projet en raison de **l’abondance et de la qualité des modèles préentraînés disponibles sur Hugging Face**, facilitant une performance optimale et un accès plus direct aux dernières avancées en NLP.

---

## Modèles utilisés

### 1. **Transcription vocale (ASR)**
- **Modèle** : [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h)
- **Description** : Un modèle entraîné par Meta AI sur 960 heures de données audio Librispeech. Il repose sur des représentations auto-supervisées de haute qualité.
- **Robustesse** : Excellente pour les accents standards et les environnements peu bruités. Léger et rapide, il convient parfaitement aux applications temps réel.

---

### 2. **Correction grammaticale**
- **Modèle** : [`vennify/t5-base-grammar-correction`](https://huggingface.co/vennify/t5-base-grammar-correction)
- **Description** : Basé sur la version allégée du modèle T5 (Text-to-Text Transfer Transformer), spécialisé dans la reformulation grammaticale.
- **Robustesse** : Très efficace pour corriger des phrases issues de transcriptions brutes. Préserve le sens tout en améliorant la syntaxe.

---

### 3. **Analyse de sentiment**
- **Modèle** : [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Description** : Variante optimisée de RoBERTa, entraînée sur des millions de tweets pour détecter les sentiments dans des textes courts et informels.
- **Robustesse** : Très performant pour comprendre les nuances émotionnelles du langage, même en présence d’erreurs ou de langage familier.

---

## Fonctionnement du pipeline

1. L'utilisateur fournit un fichier audio `.wav`
2. Le fichier est transcrit en texte brut via **Wav2Vec2**
3. Le texte est corrigé grammaticalement avec **T5**
4. Le texte corrigé est analysé par **RoBERTa** pour extraire un **label de sentiment** et une **confiance** (score de probabilité)

---

## Interfaces disponibles

- **Gradio** : Interface conviviale pour tester le pipeline localement via navigateur
- **FastAPI** : Endpoint `/analyze` permettant l'intégration dans des applications tierces (via POST multipart)

---

## Technologies

- Python 3.11
- Hugging Face Transformers
- Torch & Torchaudio
- Soundfile
- Gradio / FastAPI

---

## À venir
- Dockerisation
- Multilinguisme progressif
- Authentification pour l’API

---

## Auteur

Ce projet a été développé et documenté par Jashi Kanyana, passionné par le deep learning et les applications concrètes du NLP.

