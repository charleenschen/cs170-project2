import numpy as np
from numba import njit
import time

def get_data(filename):
    return np.loadtxt(filename)

@njit
def leave_one_out(data, current_set, feature_to_add): 
    features = np.array(current_set + [feature_to_add])
    correctly_classified = 0

    for i in range(len(data)):
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_label = None

        for j in range(len(data)):
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

    accuracy = correctly_classified / len(data)
    return accuracy


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
                accuracy = leave_one_out(data, current_set_of_features, j)
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
    current_set_of_features = []

def main():
    print("Welcome to Charleen's Feature Selection Algorithm.")
    filename = input("Type in the name of the file to test: ").strip()
    print("Type the number of the algorithm you want to run. 1. Forward Selection 2. Backward Elimination")
    algorithm = input()

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