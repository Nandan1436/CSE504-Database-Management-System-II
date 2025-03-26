import math
from collections import Counter

# Load the Iris dataset (manually or replace with your loading logic)
def load_iris_data():
    data = []
    with open('./iris.data', 'r') as file:  # Replace 'iris.data' with the actual dataset path
        for line in file:
            if line.strip():
                parts = line.strip().split(',')
                data.append((list(map(float, parts[:-1])), parts[-1]))
    return data

# Calculate entropy
def calculate_entropy(data):
    labels = [row[1] for row in data]
    label_counts = Counter(labels)
    total_samples = len(data)
    entropy = 0.0
    for count in label_counts.values():
        probability = count / total_samples
        entropy -= probability * math.log2(probability)
    return entropy

# Split data based on an attribute and threshold
def split_data(data, attribute_index, threshold):
    left = [row for row in data if row[0][attribute_index] <= threshold]
    right = [row for row in data if row[0][attribute_index] > threshold]
    return left, right

# Calculate information gain
def calculate_information_gain(data, attribute_index, threshold):
    parent_entropy = calculate_entropy(data)
    left, right = split_data(data, attribute_index, threshold)
    left_weight = len(left) / len(data)
    right_weight = len(right) / len(data)
    weighted_entropy = left_weight * calculate_entropy(left) + right_weight * calculate_entropy(right)
    return parent_entropy - weighted_entropy

# Find the best split
def find_best_split(data):
    best_gain = 0
    best_attribute = None
    best_threshold = None
    n_features = len(data[0][0])

    for attribute_index in range(n_features):
        values = set(row[0][attribute_index] for row in data)
        for threshold in values:
            gain = calculate_information_gain(data, attribute_index, threshold)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute_index
                best_threshold = threshold

    return best_attribute, best_threshold, best_gain

# Create a decision tree
class DecisionTreeNode:
    def __init__(self, attribute_index=None, threshold=None, label=None, left=None, right=None):
        self.attribute_index = attribute_index
        self.threshold = threshold
        self.label = label
        self.left = left
        self.right = right

def build_decision_tree(data, depth=0):
    labels = [row[1] for row in data]
    if len(set(labels)) == 1:  # Pure node
        return DecisionTreeNode(label=labels[0])

    if not data:  # Empty dataset
        return None

    best_attribute, best_threshold, best_gain = find_best_split(data)
    if best_gain == 0:  # No information gain, return majority label
        most_common_label = Counter(labels).most_common(1)[0][0]
        return DecisionTreeNode(label=most_common_label)

    left_data, right_data = split_data(data, best_attribute, best_threshold)
    left_subtree = build_decision_tree(left_data, depth + 1)
    right_subtree = build_decision_tree(right_data, depth + 1)

    return DecisionTreeNode(attribute_index=best_attribute, threshold=best_threshold, left=left_subtree, right=right_subtree)

# Make predictions
def predict(tree, sample):
    if tree.label is not None:
        return tree.label
    attribute_value = sample[tree.attribute_index]
    if attribute_value <= tree.threshold:
        return predict(tree.left, sample)
    else:
        return predict(tree.right, sample)

# Evaluate the decision tree
def evaluate(tree, test_data):
    correct = 0
    for sample, label in test_data:
        prediction = predict(tree, sample)
        if prediction == label:
            correct += 1
    accuracy = correct / len(test_data)
    return accuracy

# Main function
if __name__ == "__main__":
    # Load dataset
    data = load_iris_data()
    train_data = data[:120]  # First 120 samples for training
    test_data = data[120:]  # Last 30 samples for testing

    # Build decision tree
    decision_tree = build_decision_tree(train_data)

    # Evaluate decision tree
    accuracy = evaluate(decision_tree, test_data)
    print(f"Accuracy: {accuracy:.2f}")
