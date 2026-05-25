import librosa
import numpy as np
import os
import pandas as pd
import static_ffmpeg
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# This ensures ffmpeg is available for librosa/audioread
static_ffmpeg.add_paths()

def extract_features(file_path):
    # Load 30s of the audio
    # The 'y' is the sound wave, 'sr' is the sample rate
    y, sr = librosa.load(file_path, duration=30)
    
    # 1. Tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # 2. Spectral Centroid (Brightness)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # 3. MFCCs (The "Texture" of the audio)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    return [float(tempo), centroid] + mfccs_mean.tolist()

# --- POINT TO THE RIGHT FOLDER ---
# Using absolute path to be 100% safe
audio_dir = r"C:\Users\pault\Desktop\Projects\VibeMap\vibe_data"
features_list = []
song_names = []

print(f"Looking for files in: {audio_dir}")

if not os.path.exists(audio_dir):
    print("ERROR: Folder not found! Check your path.")
else:
    for file in os.listdir(audio_dir):
        if file.endswith(".webm"): 
            try:
                full_path = os.path.join(audio_dir, file)
                features = extract_features(full_path)
                features_list.append(features)
                song_names.append(file)
                print(f"Successfully analyzed: {file}")
            except Exception as e:
                print(f"Could not process {file}: {e}")

# --- CLUSTERING ---
if len(features_list) >= 2:
    df = pd.DataFrame(features_list)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    
    # Set clusters to 2 since we only have a few songs
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled)
    df['song'] = song_names
    
    print("\n" + "="*30)
    print("      VIBEMAP CLUSTERS")
    print("="*30)
    print(df[['song', 'cluster']].sort_values(by='cluster'))
else:
    print("Not enough songs found to create clusters. Need at least 2!")