import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def audio_to_mel_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)  # ← FIXED
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()