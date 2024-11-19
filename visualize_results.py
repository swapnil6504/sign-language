import json
import matplotlib.pyplot as plt
from matplotlib import cm

# Step 1: Load Results
def load_results(file_path='results.json'):
    print("Loading results...")
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results

# Step 2: Visualize Results
def visualize_results(results):
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_results]
    scores = [x[1] for x in sorted_results]

    # Generate a color palette focusing on shades of pink and vibrant colors
    colors = cm.get_cmap('cool')(range(len(names)))

    # Bar Chart
    plt.figure(figsize=(10, 6))
    plt.barh(names, scores, color=colors)
    plt.xlabel('Accuracy', fontsize=14)
    plt.ylabel('Algorithms', fontsize=14)
    plt.title('Comparison of Algorithms', fontsize=16, fontweight='bold', color='pink')
    plt.gca().invert_yaxis()

    # Add grid lines for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add labels to each bar
    for index, value in enumerate(scores):
        plt.text(value + 0.01, index, f"{value:.2f}", va='center', fontsize=12, color='purple')

    plt.tight_layout()
    plt.savefig("model_comparison_colored.png")  # Save the plot as an image
    plt.show()

if __name__ == "__main__":
    # Load Results and Visualize
    results = load_results()
    visualize_results(results)
