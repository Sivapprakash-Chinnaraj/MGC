```markdown
# 🎵 Music Genre Classification Using CNN

This project classifies music files into genres using deep learning and mel spectrograms. Spectrogram images of audio files are used as input to a Convolutional Neural Network (CNN) trained to predict the genre.

---

## 📁 Project Structure

```

MGC/
├── data/                      # Original audio files (GTZAN-like structure)
├── spectrograms/             # Generated spectrogram images by genre
├── src/
│   ├── preprocessing.py      # Code to convert audio to mel spectrograms
│   └── model.py              # CNN model for training & evaluation
├── env/                      # Virtual environment (should be in .gitignore)
├── app.py                    # Streamlit app to use the model
├── model.h5                  # Trained CNN model
├── requirements.txt          # Python dependencies
└── README.md

````

---

## 🧠 Model Overview

- Input: `64x64 RGB` spectrogram images
- Architecture: Lightweight CNN with 2 Conv layers + Dropout
- Output: Softmax over 10 music genres

---

## 🎼 Supported Genres

- Classical
- Country
- Disco
- HipHop
- Jazz
- Metal
- Pop
- Reggae
- Rock
- Blues *(if included in dataset)*

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/music-genre-classifier.git
cd music-genre-classifier
````

### 2. Create a Virtual Environment

```bash
python -m venv env
.\env\Scripts\activate      # Windows
# source env/bin/activate   # Linux/macOS
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🛠️ Preprocessing - Generate Spectrograms

Run the preprocessing script to convert audio files (`.wav`) into mel spectrogram images:

```bash
python src/preprocessing.py
```

---

## 🏋️ Train the Model

Train a CNN using spectrogram images:

```bash
python src/model.py
```

This will save `model.h5` upon training.

---

## 🖥️ Run the Streamlit App

```bash
streamlit run app.py
```

This will launch a web UI where you can upload a `.wav` file and get the predicted genre.

---

## 📦 Requirements

Sample dependencies (`requirements.txt`):

```
numpy
pillow
librosa
matplotlib
scikit-learn
tensorflow
streamlit
```

---

## 📊 Accuracy

* Final test accuracy: \~60% (on resized 64×64 images)
* Can be improved with larger spectrograms and deeper models

---

## 📌 TODOs

* [ ] Improve accuracy with data augmentation
* [ ] Add genre confidence scores in UI
* [ ] Support `.mp3` and audio trimming
* [ ] Deploy on Streamlit Cloud or HuggingFace Spaces

---

## 📜 License

MIT License – feel free to use and modify with attribution.

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Open an issue or start a discussion if you have questions or ideas.

---

## 🙌 Acknowledgments

* GTZAN Music Genre Dataset
* Librosa for audio processing
* TensorFlow & Streamlit for modeling and UI

```

---

Let me know if you'd like a version with badges (build, license, etc.) or instructions for Colab training or Hugging Face deployment.
```
