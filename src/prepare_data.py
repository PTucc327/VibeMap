import librosa
import numpy as np
import pandas as pd
import os

def extract_features(track_key):
    # Load audio
    path = librosa.example(track_key)
    y, sr = librosa.load(path)

    # 1. MFCCs (The 'Texture')
    # We take the mean and std dev of each of the 13 coefficients
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds = np.std(mfccs, axis=1)

    # 2. Spectral Features (The 'Vibe')
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # Build the feature dictionary
    features = {
        "track_id": track_key,
        "tempo": tempo[0],
        "brightness_mean": np.mean(cent),
        "rolloff_mean": np.mean(rolloff)
    }

    # Add the 13 MFCC means as separate columns
    for i, m in enumerate(mfcc_means):
        features[f"mfcc_mean_{i}"] = m

    return features

if __name__ == "__main__":
    vibe_samples = ['nutcracker', 'choice', 'fishin']
    data_list = [extract_features(s) for s in vibe_samples]
    
    df = pd.DataFrame(data_list)
    df.to_csv("data/vibe_features.csv", index=False)
    print("✅ Created vibe_features.csv with the Sonic DNA of your tracks!")
    print(df.head())