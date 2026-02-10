import gradio as gr
import librosa
import numpy as np
import joblib

# Load trained files
model = joblib.load("models/ushi_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=5)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)
    rms = np.mean(librosa.feature.rms(y=audio))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

    return np.hstack([mfcc, chroma, rms, zcr])

# Impact score mapping
impact_score = {
    "pleasant": 2,
    "neutral": 0,
    "harmful": -2
}

def predict_eco_sound(audio, latitude, longitude):
    features = extract_features(audio)
    features = scaler.transform([features])

    pred = model.predict(features)
    label = label_encoder.inverse_transform(pred)[0]

    score = impact_score[label]

    # Biodiversity stress interpretation
    if score > 0:
        stress = "🟢 Low Biodiversity Stress (Healthy Environment)"
    elif score == 0:
        stress = "🟡 Moderate Biodiversity Stress"
    else:
        stress = "🔴 High Biodiversity Stress (Noise Dominant)"

    location_info = f"Latitude: {latitude}, Longitude: {longitude}"

    return label, score, stress, location_info

interface = gr.Interface(
    fn=predict_eco_sound,
    inputs=[
        gr.Audio(type="filepath", label="Upload Environmental Audio"),
        gr.Number(label="Latitude", value=0.0),
        gr.Number(label="Longitude", value=0.0)
    ],
    outputs=[
        gr.Textbox(label="Dominant Sound Type"),
        gr.Number(label="EcoSound Health Index"),
        gr.Textbox(label="Biodiversity Stress Level"),
        gr.Textbox(label="Location Coordinates")
    ],
    title="EcoSound AI – Urban Noise & Biodiversity Health Monitor",
    description=(
        "AI-based decision-support system that analyzes environmental sounds "
        "along with location coordinates to assess noise dominance and biodiversity stress."
    )
)

if __name__ == "__main__":
    interface.launch()
