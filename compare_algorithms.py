import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
import json  # To save results for visualization

warnings.filterwarnings("ignore")

# Step 1: Load Dataset
def load_data():
    print("Loading dataset...")
    data_dict = pickle.load(open('./data.pickle', 'rb'))
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])
    return data, labels

# Step 2: Train and Evaluate Models
def train_and_evaluate(x_train, x_test, y_train, y_test):
    classifiers = {
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(kernel='linear'),
        "Logistic Regression": LogisticRegression()
    }

    results = {}
    for name, model in classifiers.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)  # Train the model
        y_pred = model.predict(x_test)  # Test the model
        acc = accuracy_score(y_test, y_pred)  # Calculate accuracy
        results[name] = acc  # Save accuracy in results

        # Print Classification Report
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
    
    return results

# Main Function
if __name__ == "__main__":
    # Step 1: Load Data
    data, labels = load_data()

    # Step 2: Split Data into Train and Test Sets
    print("Splitting data...")
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )

    # Step 3: Train and Evaluate Models
    print("Training models...")
    results = train_and_evaluate(x_train, x_test, y_train, y_test)

    # Step 4: Save Results for Visualization
    print("Saving results...")
    with open('results.json', 'w') as f:
        json.dump(results, f)
    print("Results saved to 'results.json'")
