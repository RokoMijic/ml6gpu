from get_data import create_data_with_labels























if __name__ == "__main__":

    (train_data, train_labels) = create_data_with_labels("../data/train/")

    print(train_labels[0])
    print(train_data[0])