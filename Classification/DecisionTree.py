from pprint import pprint

import pandas as pd

from AttributeSelectionMeasure import gain, load_dataset


class DecisionTree:
    def __init__(self, is_binary=False):
        self.decision_tree = None
        self.majority = None
        self.is_binary = is_binary

    def fit(self, dataset: pd.DataFrame, class_label_column):
        self.majority = dataset[class_label_column].mode()[0]
        self.decision_tree = self._train_recursive(dataset, class_label_column)

    def _train_recursive(self, dataset: pd.DataFrame, class_label_column, tree=None):
        # print(dataset.shape)
        spl_pt,spl_attr = self._find_splitting_attribute(dataset, class_label_column)
        # print(spl_attr)
        # print(spl_attr, attr_values)

        if tree is None:
            tree = {}
            if self.is_binary:
                tree[(spl_attr, spl_pt)] = dict()
            else:
                tree[(spl_attr,0)] = dict()
        
        if self.is_binary:
            d1 = dataset[dataset[spl_attr] < spl_pt]
            d2 = dataset[dataset[spl_attr] >= spl_pt]
            self._assign_node(d1,spl_attr,'l', tree,spl_pt=spl_pt)
            self._assign_node(d2,spl_attr,'ge', tree,spl_pt=spl_pt) 

        else:
            attr_values = dataset[spl_attr].unique()
            for val in attr_values:
                data_partition = dataset[dataset[spl_attr] == val]
                data_partition = data_partition.drop(columns=[spl_attr])  # remove the splitting attribute
                self._assign_node(data_partition,spl_attr,val, tree)  
            return tree


    def _assign_node(self, data_partition, spl_attr, attr_val, tree,spl_pt=0):
        class_values = data_partition[class_label_column].unique()
        if data_partition.shape[0] == 0:
            tree[(spl_attr,spl_pt)][attr_val] = self.majority
        elif data_partition.shape[1] == 1:
            tree[(spl_attr,spl_pt)][attr_val] = data_partition[class_label_column].mode()[0]
            # print('empty')
        elif len(class_values) == 1:
            tree[(spl_attr,spl_pt)][attr_val] = class_values[0]
            # print('pure class')
        else:
            # recursion
            tree[(spl_attr,spl_pt)][attr_val] = self._train_recursive(data_partition, class_label_column)


    def predict(self, data: pd.DataFrame):
        preds = []
        if self.decision_tree is None:
            print('call fit first')
            return
        for index, r in data.iterrows():
            preds.append(self.predict_single(r, self.decision_tree))
        return preds

    def predict_single(self, row, dt: dict = None):
        if not self.is_binary:
            for (key,tp) in dt:
                val = row[key]
                if val not in dt[(key,tp)]:
                    pred = self.majority
                    continue
                dt = dt[(key,tp)][val]
                if type(dt) is dict:
                    pred = self.predict_single(row, dt)
                else:
                    pred = dt
            return pred

    def _find_splitting_attribute(self, dataset: pd.DataFrame, class_label_column):
        attr_cols = list(dataset.columns)
        attr_cols.remove(class_label_column)
        # print(attr_cols)
        max_gain = 0
        splitting_attr = None
        for col in attr_cols:
            g = gain(data, class_label_column, col)
            if g > max_gain:
                splitting_attr = col
                max_gain = max(g, max_gain)
        # print(max_gain, splitting_attr)
        return None,splitting_attr

    def print_tree(self):
        pprint(self.decision_tree)


def calculate_accuracy(predictions: list, class_labels: list):
    t_len = len(predictions)
    correct = 0.0
    for i in range(t_len):
        if predictions[i] == class_labels[i]:
            correct += 1
    return 100 * (correct / t_len)


if __name__ == '__main__':
    data = load_dataset('Dataset/BreastCancer/breast-cancer.data')
    # shuffle    ``
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.reset_index(drop=True)
    k = 3
    fold_size = data.shape[0] // k
    begin_index = 0
    end_index = begin_index + fold_size
    print('dataset size', len(data))
    class_label_column = 0
    for i in range(k):
        # print(begin_index, end_index)
        test_frame = data[begin_index:end_index + 1].reset_index(drop=True)
        test_labels = test_frame[class_label_column]
        test_frame.drop(columns=class_label_column, inplace=True)
        train_frame = data.drop(data.iloc[begin_index:end_index + 1].index).reset_index(drop=True)
        # print(len(train_frame))
        dt = DecisionTree()
        dt.fit(train_frame, class_label_column=class_label_column)
        #dt.print_tree()
        preds = dt.predict(test_frame)
        print(calculate_accuracy(preds, list(test_labels)))
        # train_frame = data.drop(range(begin_index,end_index+1), axis=0)
        begin_index = end_index + 1
        if i == k - 2:
            end_index = data.shape[0] - 1
        else:
            end_index = end_index + fold_size
