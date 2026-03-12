import numpy as np

def get_data(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            values = list(map(float, line.split()))
            data.append(values)
    return data

def leave_one_out(data, current_set, feature_to_add): 
    features = current_set + [feature_to_add]
    correctly_classified = 0

    for i in range(len(data)):
        object_to_classify = data[i, features]
        label_object_to_classify = data[i, 0]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_label = None

        for j in range(len(data)):
            if i == j:
                continue

            distance = np.sqrt(np.sum((object_to_classify - data[j, features]) ** 2))

            if distance < nearest_neighbor_distance:
                nearest_neighbor_distance = distance
                nearest_neighbor_label = data[j, 0]

            if label_object_to_classify == nearest_neighbor_label:
                correctly_classified += 1

    accuracy = correctly_classified / len(data)
    return accuracy


def forward_selection(data, num_features):
    global best_accuracy
    best_accuracy = 0.0
    current_set_of_features = []
    best_set_of_features = []
    
    for i in range(num_features):
        feature_to_add = None
        best_so_far = 0
        print("On the %dth level of the search tree", i)
        for j in range(num_features):
                if j in current_set_of_features:
                    continue
                accuracy = leave_one_out(data, current_set_of_features, j)
                print(f"\tUsing features {set(current_set_of_features + [j])}: accuracy is {accuracy*100:.1f}%")

                if accuracy > best_so_far:
                    best_so_far = accuracy;
                    feature_to_add = j

        if best_accuracy >= global_accuracy:
            global_accuracy = best_accuracy
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
    num_features = len(data)

    if algorithm == '1':
        forward_selection(data, num_features)
    elif algorithm == '2':
        backward_elimination(data, num_features)
    else:
        print ("Invalid selection, defaulting to forward selection")
        forward_selection(data, num_features)

if __name__ == "__main__":
    main()