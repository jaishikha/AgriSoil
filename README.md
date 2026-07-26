# AgriSoil

🔗 **Live Demo:** [Click here to try the live web application!](https://agrisoil-fdn536qfkh6a2d6dbp8zbv.streamlit.app/)

**AgriSoil** is a smart web application built to take the confusion out of soil test reports. You simply type in your lab numbers like pH, electrical conductivity,nutrient levels, and the app instantly runs a ML model to diagnose the soil's health, display visual breakdown bars, and give you exact fertilizer prescriptions so you know precisely what your crops need.

## Features

**Bilingual Support**
- Lets you switch easily between English and Hindi across the whole website.
- Translates buttons, labels, and soil results instantly.
- Makes it simple and easy to use for farmers, students, and anyone else who prefers Hindi or English.

**Instant Lab Diagnostics**
- Uses a trained ML model to check your main soil numbers and micronutrients within seconds.
- Takes away the stress of manual calculations by handling complicated soil test math for you.

**Visual Status Breakdown & Fertilizer Prescriptions**
- Uses simple, color-coded progress bars to show you the status of each nutrient.
- Instantly points out if a nutrient is too low, just right, or dangerously high.
- Tells you exact, easy-to-follow steps to fix your soil and gives you the exact amount of fertilizer to use based on your test numbers.

**Gemini AI Advisory Assistant**
- Powered by Google's AI to chat with you and answer any questions about your soil or farming.
- Explains things like pH levels and how to take care of your fields in simple terms.
- Strictly restricted to agriculture, soil health, and farming topics so you only get relevant guidance.

## Tech stack

| Layer | Choice |
|---|---|
| Frontend | [Streamlit](https://streamlit.io/) |
| Backend | Python, Streamlit |
| Machine Learning | [Scikit-Learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/) |
| Model Training | Google Colab |
| Data Handling | Pandas,  NumPy |
| AI Integration | Google GenAI SDK [`gemini-3.1-flash-lite`](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-3.1-flash-lite) |
| Live Hosting | Streamlit Community Cloud |

### Why this stack

**Streamlit**: Lets you build an interactive website entirely using Python, so you don't have to waste time writing complicated HTML, CSS, or JavaScript.

**Scikit-Learn & Joblib**: Gives you fast and lightweight tools to run your machine learning model and easily save or load it for quick predictions.

**Google GenAI SDK (Gemini)**: Provides fast, smart AI answers and language support while being very efficient with usage limits.

**Pandas & NumPy**: Makes it easy to handle numbers, data charts, and soil test calculations smoothly.

**Streamlit Community Cloud**: Lets you host and publish your website straight from GitHub for free, with a safe place to hide your API keys.

## Getting started

**Prerequisites**
- Python 3.x installed on your machine
- A free Gemini API key generated via Google AI Studio

**Run it**
```bash
git clone <this-repository-url>
cd AgriSoil
pip install -r requirements.txt
```
Set your API key variable in your terminal:

```bash
# Windows
set GEMINI_API_KEY="your_actual_key_here"

# Mac/Linux
export GEMINI_API_KEY="your_actual_key_here"
```

Start the development app:

```bash
streamlit run app.py
```

## How to use it

1. **Enter Soil Data** — Type in your lab test values like pH, electrical conductivity, and nutrient numbers into the input fields.
2. **View Diagnostics** — Check the color-coded progress bars to instantly see your soil health and spot any nutrient imbalances.
3. **Get Prescriptions** — Read the custom soil recovery steps and exact fertilizer dosages recommended for your field.
3. **Ask the AI** — Chat with the Gemini assistant in English or Hindi to get instant answers to any farming or soil questions.

## Known limitations

- **Manual Data Entry Required** : Users must manually input their lab test numbers since automated image or document scanning is not currently integrated.
- **Single Sample Focus** : The app analyzes and generates prescriptions for one soil sample profile at a time rather than handling bulk batch datasets.
- **Region-Specific Focus** : Because the model was trained on genuine, real-world agricultural data collected directly from local fields around Jodhpur, it is highly accurate for regional use but may give incorrect predictions for completely different climates.

*Disclaimer: This project is unlicensed and created solely for educational purposes.*
