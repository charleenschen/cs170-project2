import numpy as np
from numba import njit
import time

def get_data(filename):
    return np.loadtxt(filename)

@njit
def leave_one_out(data, features): 
    correctly_classified = 0

    for i in range(data.shape[0]):
        nearest_neighbor_distance = np.inf
        nearest_neighbor_label = None

        for j in range(data.shape[0]):
            if i == j:
                continue

            distance = 0.0
            for f in features:
                diff = data[i, f] - data[j, f]
                distance += diff * diff

            if distance < nearest_neighbor_distance:
                nearest_neighbor_distance = distance
                nearest_neighbor_label = data[j, 0]

        if data[i, 0] == nearest_neighbor_label:
            correctly_classified += 1

    return correctly_classified / data.shape[0]


def forward_selection(data, num_features):
    best_accuracy = 0.0
    current_set_of_features = []
    best_set_of_features = []
    
    for i in range(num_features):
        feature_to_add = None
        best_so_far = 0
        print(f"On the {i}th level of the search tree")
        for j in range(1, num_features + 1):
                if j in current_set_of_features:
                    continue
                accuracy = leave_one_out(data, np.array(current_set_of_features + [j]))
                print(f"\tUsing features {set(current_set_of_features + [j])}: accuracy is {accuracy*100:.1f}%")

                if accuracy > best_so_far:
                    best_so_far = accuracy;
                    feature_to_add = j

        if best_so_far > best_accuracy:
            best_accuracy = best_so_far
            best_set_of_features = current_set_of_features + [feature_to_add]
    
        current_set_of_features.append(feature_to_add)
        print(f'On level {i}, added feature {feature_to_add}. Current set: {set(current_set_of_features)}, accuracy: {best_accuracy*100:.1f}%\n')
    print(f"\nFinished. Best feature set found: {set(best_set_of_features)} with accuracy {best_accuracy*100:.1f}%")

def backward_elimination(data, num_features):
    current_set_of_features = list(range(1, num_features + 1))
    best_set_of_features = list(current_set_of_features)
    best_accuracy = leave_one_out(data, np.array(current_set_of_features))

    individual_scores = {}
    for f in range(1, num_features + 1):
        score = leave_one_out(data, np.array([f]))
        individual_scores[f] = score

    print(f"Starting with all features: {set(current_set_of_features)}, accuracy: {best_accuracy*100:.1f}%\n")

    for i in range(num_features - 1):
        feature_to_remove = None
        best_so_far = -1.0
        print(f"On the {i}th level of the search tree")
        for j in current_set_of_features:
            candidate = [f for f in current_set_of_features if f != j]
            accuracy = leave_one_out(data, np.array(candidate))
            print(f"\tUsing features {set(candidate)}: accuracy is {accuracy*100:.1f}%")

            if accuracy > best_so_far:
                best_so_far = accuracy
                feature_to_remove = j
            elif accuracy == best_so_far:
                if individual_scores[j] < individual_scores.get(feature_to_remove, 1.0):
                    feature_to_remove = j

        current_set_of_features.remove(feature_to_remove)

        if best_so_far > best_accuracy:
            best_accuracy = best_so_far
            best_set_of_features =  list(current_set_of_features)
        
        print(f'On level {i}, removed feature {feature_to_remove}. Current set: {set(current_set_of_features)}, accuracy: {best_accuracy*100:.1f}%\n')
    print(f"\nFinished. Best feature set found: {set(best_set_of_features)} with accuracy {best_accuracy*100:.1f}%")

def main():
    print("Welcome to Charleen's Feature Selection Algorithm.")
    # filename = input("Type in the name of the file to test: ").strip()
    filename = "large47.txt"
    print("Type the number of the algorithm you want to run. 1. Forward Selection 2. Backward Elimination")
    # algorithm = input()
    algorithm = "2"

    # for i in range(1, data.shape[1]):
    #     col = data[:, i]
    #     mean = np.mean(col)
    #     std = np.std(col)
    
    # # Check for zero std to avoid division by zero
    # if std > 0:
    #     data[:, i] = (col - mean) / std
    # else:
    #     data[:, i] = 0.0

    data = get_data(filename)
    num_features = data.shape[1] - 1

    start = time.time()
    if algorithm == '1':
        forward_selection(data, num_features)
    elif algorithm == '2':
        backward_elimination(data, num_features)
    else:
        print ("Invalid selection, defaulting to forward selection")
        forward_selection(data, num_features)
    end = time.time()
    print(f"\nSearch completed in {(end - start) / 60:.2f} minutes")

if __name__ == "__main__":
    main()