import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Load model
model = tf.keras.models.load_model('D:/MGC/notebooks/genre_classifier_model.h5')

# Genre labels (adjust based on your actual model classes)
genre_labels = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Convert audio to mel spectrogram image
def audio_to_mel_spectrogram(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig = plt.figure(figsize=(3, 3))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    
    # Save spectrogram to buffer
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image = image[:, :, :3]  # Remove alpha channel
    plt.close(fig)
    st.image(image, caption='Mel Spectrogram', use_column_width=True)

    # Resize for model
    image = Image.fromarray(image).resize((256, 256))
    image = np.array(image) / 255.0  # Normalize
    return image

# Streamlit UI
st.title("🎵 Music Genre Classifier")
st.write("Upload an audio file (WAV format) and I’ll predict the genre!")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.write("Processing...")

    try:
        # Convert to spectrogram and predict
        image = audio_to_mel_spectrogram(uploaded_file)
        image = np.expand_dims(image, axis=0)  # (1, 128, 128, 3)
        prediction = model.predict(image)
        predicted_genre = genre_labels[np.argmax(prediction)]

        st.success(f"🎧 Predicted Genre: **{predicted_genre.upper()}**")
    except Exception as e:
        st.error(f"Error: {e}")