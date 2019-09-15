from itertools import combinations

import numpy as np
import pandas as pd

import math


def load_dataset(filename: str):
    df = pd.read_csv(filename, header=None)

    global dataset_size
    dataset_size = df.shape[0]

    return df


def entropy(dataset: pd.DataFrame, class_label_column):
    entropy_node = 0

    values = dataset[class_label_column].unique()

    for val in values:
        # print(val, dataset[class_label_column].value_counts()[val])

        p_i = dataset[class_label_column].value_counts()[val] / len(dataset[class_label_column])
        entropy_node += (-p_i * np.log2(p_i))

    return entropy_node


def entropy_attribute(dataset: pd.DataFrame, class_label_column, attribute):
    entropy_attr = 0

    attr_vars = dataset[attribute].unique()

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


def gain(dataset: pd.DataFrame, class_label_column, attribute):
    _gain = entropy(dataset, class_label_column) - entropy_attribute(dataset, class_label_column, attribute)

    return _gain


def gini(dataset: pd.DataFrame, class_label_column):
    labels = dataset[class_label_column].unique()

    list_pi = list()

    for val in labels:
        # print(val, dataset[class_label_column].value_counts()[val])

        p_i = dataset[class_label_column].value_counts()[val] / len(dataset[class_label_column])
        list_pi.append(p_i ** 2)

    _gini = 1 - sum(list_pi)

    return _gini


def gini_attribute(dataset: pd.DataFrame, class_label_column, attribute):
    attr_vals = dataset[attribute].unique()
    # print(attr_vals)

    mx_gini = 0
    splitting_attr = list()

    for r in range(1, math.floor(len(attr_vals) / 2) + 1):
        comb_list = list(combinations(attr_vals, r))

        for subset in comb_list:
            d1 = set(attr_vals) - set(subset)
            d2 = set(subset)

            # print('D1 :', d1, 'D2 :', d2)

            g1 = dataset[dataset[attribute].isin(d1)]
            g2 = dataset[dataset[attribute].isin(d2)]

            # print('G1 :', g1.shape[0], 'g2 :', g2.shape[0])

            G1 = gini(g1, class_label_column)
            G2 = gini(g2, class_label_column)

            # print('GINI - 1 :', G1, 'GINI - 2 :', G2)

            _gini_attr = ((g1.shape[0] / dataset_size) * G1) + ((g2.shape[0] / dataset_size) * G2)

            # print('GINI_ATTR :', _gini_attr)

            if _gini_attr > mx_gini:
                mx_gini = _gini_attr
                splitting_attr = [d1, d2]

            # print('MAX GINI:', mx_gini)
            # print(splitting_attr, '\n')

    return mx_gini, splitting_attr


if __name__ == '__main__':
    data = load_dataset('Dataset/play_tennis.csv')

    print(gini(data, class_label_column=4))
    print()

    g_mx, spl_attr = gini_attribute(data, class_label_column=4, attribute=2)
    print(g_mx)
    print(spl_attr)
