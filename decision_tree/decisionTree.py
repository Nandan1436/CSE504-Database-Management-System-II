import math
import random
import numpy as np
from statistics import mean
from sklearn.metrics import f1_score,precision_score, recall_score
from sklearn.metrics import fbeta_score

# Load the Iris dataset
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
    
    # Manually count occurrences of each label
    label_counts = {}
    for label in labels:
        if label not in label_counts:
            label_counts[label] = 1
        else:
            label_counts[label] += 1

    total_samples = len(data)  # Total number of samples in the dataset
    entropy = 0.0
    for count in label_counts.values():
        probability = count / total_samples  # Probability of each label
        entropy -= probability * math.log2(probability)  # Calculate entropy using the formula
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

def calculate_gini_impurity(data):
    labels = [row[1] for row in data]
    label_count = {}

    for label in labels:
        if label not in label_count:
            label_count[label] = 1
        else:
            label_count[label] += 1

    total_samples = len(data)
    gini = 1.0
    for count in label_count.values():
        probability = count / total_samples
        gini -= probability ** 2
    return gini


def calculate_gini_gain(data, attribute_index, threshold):
    left, right = split_data(data, attribute_index, threshold)
    left_gini = calculate_gini_impurity(left)
    right_gini = calculate_gini_impurity(right)
    parent_gini = calculate_gini_impurity(data)
    left_w = len(left) / len(data)
    right_w = len(right) / len(data)
    weighted_gini = left_w * left_gini + right_w * right_gini
    gini_gain = parent_gini - weighted_gini
    return gini_gain

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
    
            #gain = calculate_gini_gain(data, attribute_index, threshold)
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

def build_decision_tree(data, criterion=0, depth=0,max_depth=5):
    labels = [row[1] for row in data]
    
    # Check if all labels are the same (pure node)
    if len(set(labels)) == 1:  # Pure node, all labels are the same
        return DecisionTreeNode(label=labels[0])

    if not data:  # Empty dataset
        return None

    if max_depth is not None and depth >= max_depth:
        most_common_label = max(labels, key=labels.count)
        return DecisionTreeNode(label=most_common_label)

    best_attribute, best_threshold, best_gain = find_best_split(data)
    
    # If no information gain, return majority label
    if best_gain == 0:
        # Manually count occurrences of each label
        label_counts = {}
        for label in labels:
            if label not in label_counts:
                label_counts[label] = 1
            else:
                label_counts[label] += 1
        
        # Find the label with the maximum count
        most_common_label = max(label_counts, key=label_counts.get)
        return DecisionTreeNode(label=most_common_label)

    left_data, right_data = split_data(data, best_attribute, best_threshold)
    left_subtree = build_decision_tree(left_data, criterion, depth + 1, max_depth)
    right_subtree = build_decision_tree(right_data, criterion, depth + 1, max_depth)

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
        #print(sample, label,prediction)
        if prediction == label:
            correct += 1
    accuracy = correct / len(test_data)
    return accuracy

def evaluate_f1(tree, test_data):
    y_true = [label for _, label in test_data]
    y_pred = [predict(tree, sample) for sample, _ in test_data]
    return f1_score(y_true, y_pred, average='macro')  # 'macro' averages F1-score across all classes

def evaluate_f2(tree, test_data):
    y_true = [label for _, label in test_data]
    y_pred = [predict(tree, sample) for sample, _ in test_data]
    return fbeta_score(y_true, y_pred, beta=2, average='macro')  # 'macro' averages F2-score across all classes

# Print the decision tree
def print_tree(node, depth=0):
    if node is None:
        return
    
    # Print the label if it's a leaf node
    if node.label is not None:
        print("  " * depth + f"Leaf: {node.label}")
    else:
        # Print the decision rule at the current node
        print("  " * depth + f"Attribute {node.attribute_index} <= {node.threshold}?")
        
        # Recursively print the left and right subtrees
        print("  " * depth + "Left:")
        print_tree(node.left, depth + 1)
        
        print("  " * depth + "Right:")
        print_tree(node.right, depth + 1)

def k_fold_cross_validation(data, k):
    fold_size = len(data) // k
    folds=[data[i*fold_size:(i+1)*fold_size] for i in range (k)]
    #accuracies = []
    f1_scores = []
    f2_scores = []
    d2h_scores = []

    for i in range(k):
        test_data = folds[i]
        train_data = [item for j in range(k) if j!=i for item in folds[j]]
        decision_tree=build_decision_tree(train_data)

        #accuracy=evaluate(decision_tree, train_data)
        f1=evaluate_f1(decision_tree, test_data)
        f1_scores.append(f1)
        f2=evaluate_f2(decision_tree, test_data)
        f2_scores.append(f2)
        
        y_true = [label for _, label in test_data]
        y_pred = [predict(decision_tree, sample) for sample, _ in test_data]
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        # Calculate d2h (Distance to Heaven)
        d2h = np.sqrt((1 - precision)**2 + (1 - recall)**2)
        d2h_scores.append(d2h)
        #print(round(f1,3))
        #print(round(accuracy,3))
        #accuracies.append(accuracy)
        # if i==0:
        #     print_tree(decision_tree)
    
    #average_accuracy=mean(accuracies)
    print("F1 score for each fold:")
    for i,score in enumerate(f1_scores,start=1):
        print(f"Fold {i}: {round(score,3)}")
    average_f1=mean(f1_scores)
    print("Average f1 score:",round(average_f1,3))
    print()
    print("F2 scores for each fold:")
    for i,score in enumerate(f2_scores,start=1):
        print(f"Fold {i}: {round(score,3)}")
    average_f2=mean(f2_scores)
    print("Average f2 score:",round(average_f2,3))
    print()
    for i,score in enumerate(d2h_scores,start=1):
        print(f"Fold {i}: {round(score,3)}")
    average_d2h = mean(d2h_scores)
    print("Average d2h score: ",round(average_d2h,3))


# Main function for cross-validation
if __name__ == "__main__":
    # Load dataset
    data = load_iris_data()
    #print(data)
    random.shuffle(data)

    label_mapping = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }

    for i,row in enumerate(data):
        label=row[1]
        data[i] = (row[0],label_mapping[label])

    # accuracies = []
    # num_iterations = 1
    # max_depth = 5

    k_fold_cross_validation(data,5)
