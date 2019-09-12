import pandas as pd
import numpy as np


def entropy(dataset:pd.DataFrame, class_label_column):
    entropy_node = 0

    values = dataset[class_label_column].unique()

    for val in values:

        # print(val, dataset[class_label_column].value_counts()[val])

        p_i = dataset[class_label_column].value_counts()[val] / len(dataset[class_label_column])
        entropy_node += (-p_i * np.log2(p_i))

    return entropy_node


def entropy_attribute(dataset:pd.DataFrame, class_label_column, attribute):
    entropy_attr = 0

    attr_vars = dataset[attribute].unique()
    dataset_size = dataset.shape[0]

    for val in attr_vars:
        # print('\nAj =', val)

        df_attr = dataset[dataset[attribute] == val]
        # print(df_attr)

        info = entropy(df_attr, class_label_column)
        # print('Info(Dj) =', info)

        # print('Dj :', df_attr.shape[0], 'D :', dataset_size)
        fraction = df_attr.shape[0] / dataset_size
        # print('Dj / D = ', fraction)

        entropy_attr += (fraction * info)

    return entropy_attr


def gain(dataset:pd.DataFrame, class_label_column, attribute):

    _gain = entropy(dataset, class_label_column) - entropy_attribute(dataset, class_label_column, attribute)

    return _gain


def ginni(data_partition:pd.DataFrame, class_labels):
    pass