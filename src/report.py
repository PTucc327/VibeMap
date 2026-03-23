import pandas as pd

def generate_critic_brief():
    # 1. Load your extracted features
    df = pd.read_csv("data/vibe_features.csv")
    
    # 2. Extract a summary of the 'Collective Vibe'
    avg_tempo = df['tempo'].mean()
    avg_brightness = df['brightness_mean'].mean()
    
    # 3. Find the 'Outlier' (The most unique track in this set)
    # We'll use brightness as a simple proxy for 'uniqueness' here
    brightest_track = df.loc[df['brightness_mean'].idxmax()]['track_id']
    darkest_track = df.loc[df['brightness_mean'].idxmin()]['track_id']

    # 4. Construct the prompt for the Career Brain
    prompt = f"""
    --- MUSIC CRITIC DATA BRIEF ---
    Average Tempo: {avg_tempo:.1f} BPM
    Overall Brightness: {avg_brightness:.0f} (Scale: 1000-3000)
    Sonic Range: From the 'Dark' tones of {darkest_track} to the 'Bright' energy of {brightest_track}.
    
    --- THE CRITIC'S CHALLENGE ---
    Based on this data, describe the 'Vibe' of this listener. 
    Are they a chaotic drum-and-bass enthusiast or a calculated classical lover? 
    Write a 2-sentence witty review of their 'VibeMap'.
    """
    
    print("🚀 PROMPT GENERATED FOR YOUR LLM:")
    print(prompt)
    return prompt

if __name__ == "__main__":
    generate_critic_brief()