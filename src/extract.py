import librosa
import numpy as np
import matplotlib.pyplot as plt
def analyze_vibe(track_key):
    """
    Loads a librosa example track and extracts basic 'Vibe' signatures.
    """
    print(f"\n--- Analyzing Vibe: {track_key} ---")
    
    # 1. Load the example file path
    # 'nutcracker' = Orchestral, 'choice' = Drum & Bass, 'fishin' = Folk/Pop
    path = librosa.example(track_key)
    y, sr = librosa.load(path)

    # 2. Extract Tempo (BPM)
    # Using the latest 0.10+ beat_track API
    # Extract Tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Extract Spectral Centroid (The 'Brightness')
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    avg_brightness = np.mean(cent)

    # Extract Zero Crossing Rate (The 'Energy')
    zcr = librosa.feature.zero_crossing_rate(y)
    avg_energy = np.mean(zcr)

    print(f"✅ Analysis Complete for '{track_key}'")
    # Fix: Reference the first element of the tempo array
    print(f"   > Tempo: {tempo[0]:.2f} BPM") 
    print(f"   > Brightness: {avg_brightness:.2f}")
    print(f"   > Energy: {avg_energy:.4f}")

    return {
        "track": track_key,
        "tempo": tempo[0], # Store as a float for easier clustering later
        "brightness": avg_brightness,
        "energy": avg_energy
    }


def visualize_mfccs(y, sr, track_name):
    """
    Extracts and plots MFCCs to visualize the 'texture' of the audio.
    """
    # Extract 13 MFCCs (Standard for music analysis)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title(f'MFCC Fingerprint: {track_name}')
    plt.tight_layout()
    
    # Save the visualization for your portfolio
    plt.savefig(f"data/{track_name}_mfcc.png")
    print(f"✅ Saved MFCC Heatmap to data/{track_name}_mfcc.png")
    plt.show()




if __name__ == "__main__":
    # Let's compare three very different 'Vibes'
    vibe_samples = ['nutcracker', 'choice', 'fishin']
    results = []

    for sample in vibe_samples:
        stats = analyze_vibe(sample)
        results.append(stats)
    
    print("\n--- Summary Comparison ---")
    for r in results:
        print(f"{r['track'].capitalize()}: {r['tempo']:.0f} BPM | Brightness: {r['brightness']:.0f}")

    # Visualize MFCCs for the 'choice' track (Drum & Bass)
    
    for sample in vibe_samples:
        y, sr = librosa.load(librosa.example(sample))
        visualize_mfccs(y, sr, sample)