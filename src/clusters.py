import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_vibe_distance():
    """
    Takes a DataFrame of vibe features and calculates pairwise distances.
    """
    df = pd.read_csv("data/vibe_features.csv")
    track_names = df['track_id'].tolist()


    features = df.drop(columns=['track_id'])

    scalar = StandardScaler()
    features_scaled = scalar.fit_transform(features)
    distance_matrix = euclidean_distances(features_scaled)

    # Create a DataFrame for the distance matrix
    distance_df = pd.DataFrame(distance_matrix, index=track_names, columns=track_names)
    print("\n--- Pairwise Vibe Distances ---")
    print(distance_df)
    # Visualize the distance matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(distance_df, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Pairwise Vibe Distances")
    plt.show()
    plt.savefig("data/vibe_distance_matrix.png")
    print("✅ Saved Vibe Distance Matrix Heatmap to data/vibe_distance_matrix.png")
    # Select only the numeric features for distance calculation
    
    
    return distance_df


if __name__ == "__main__":
    distance_df = calculate_vibe_distance()
    print("\n ---- The Vibe Verdict ----")
    print(distance_df)