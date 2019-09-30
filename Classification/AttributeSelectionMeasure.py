import math
from itertools import combinations

import numpy as np
import pandas as pd
from tabulate import tabulate


def load_dataset(filename: str):
    df = pd.read_csv(filename, header=None)

    global dataset_size
    dataset_size = df.shape[0]

    print(tabulate([['Dataset Size', dataset_size], ['# of Attributes', df.shape[1] - 1]], tablefmt='grid',
                   headers=['Dataset Summary', filename]))

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


def entropy_attribute_cont(dataset: pd.DataFrame, class_label_column, attribute):
    attr_col = dataset[attribute].sort_values()

    min_entropy = float('inf')
    split_pt = 0
    # print(len(attr_col.unique()))

    for i in range(len(attr_col) - 1):
        if attr_col.iloc[i] == attr_col.iloc[i + 1]:
            continue
        mid_pt = (attr_col.iloc[i] + attr_col.iloc[i + 1]) / 2

        d1 = dataset[dataset[attribute] <= mid_pt]
        d2 = dataset[dataset[attribute] > mid_pt]

        e1 = entropy(d1, class_label_column)
        e2 = entropy(d2, class_label_column)

        _entropy = ((d1.shape[0] / dataset_size) * e1) + ((d2.shape[0] / dataset_size) * e2)

        if _entropy < min_entropy:
            min_entropy = _entropy
            split_pt = mid_pt

    return min_entropy, split_pt


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

    min_gini = float('inf')
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

            if _gini_attr <= min_gini:
                min_gini = _gini_attr
                splitting_attr = [d1, d2]

            # print('MAX GINI:', mx_gini)
            # print(splitting_attr, '\n')

    return min_gini, splitting_attr


def gini_cont(dataset: pd.DataFrame, class_label_column, attribute):
    attr_col = dataset[attribute].sort_values()

    min_gini = float('inf')
    split_pt = 0
    # print(attr_col)

    for i in range(len(attr_col) - 1):
        if attr_col.iloc[i] == attr_col.iloc[i + 1]:
            continue
        mid_pt = (attr_col.iloc[i] + attr_col.iloc[i + 1]) / 2

        d1 = dataset[dataset[attribute] <= mid_pt]
        d2 = dataset[dataset[attribute] > mid_pt]

        g1 = gini(d1, class_label_column)
        g2 = gini(d2, class_label_column)

        _gini = ((d1.shape[0] / dataset_size) * g1) + ((d2.shape[0] / dataset_size) * g2)

        if _gini < min_gini:
            min_gini = _gini
            split_pt = mid_pt

    return min_gini, split_pt


if __name__ == '__main__':
    data = load_dataset('Dataset/Iris/iris.data')

    print(gini(data, class_label_column=4))
    print()

    # spl_attr, spl_pt = selct_attr_gini_cont(data, class_label_column=4)

    # print(spl_attr, spl_pt)
