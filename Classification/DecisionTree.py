from pprint import pprint

import pandas as pd

from AttributeSelectionMeasure import load_dataset, entropy_attribute, entropy_attribute_cont


class DecisionTree:
    def __init__(self, is_binary=False):
        self.decision_tree = None
        self.majority = None
        self.is_binary = is_binary
        self.numeric_cols = list()

    def fit(self, dataset: pd.DataFrame, numeric_col_list:list, class_label_column):
        self.majority = dataset[class_label_column].mode()[0]
        self.numeric_cols = numeric_col_list
        if len(numeric_col_list) == dataset.shape[1] -1:
            self.is_binary = True
        self.decision_tree = self._train_recursive(dataset, class_label_column)

    def _train_recursive(self, dataset: pd.DataFrame, class_label_column, tree=None):
        # print(dataset.shape)
        spl_pt,spl_attr = self._find_splitting_attribute(dataset, class_label_column)
        # spl_attr,spl_pt = selct_attr_gini_cont(dataset, class_label_column)
        # print(spl_attr,spl_pt)
        # print(spl_attr, attr_values)

        if tree is None:
            tree = {}
            tree[(spl_attr, spl_pt)] = dict()
        
        if spl_attr in self.numeric_cols:
            d1 = dataset[dataset[spl_attr] < spl_pt]
            d2 = dataset[dataset[spl_attr] >= spl_pt]
            # print(dataset.shape,d1.shape,d2.shape)
            if d1.shape[0] == 0 or d2.shape[0] == 0:
                tree[(spl_attr,spl_pt)]['_'] = dataset[class_label_column].mode()[0]
            else:
                self._assign_node(d1,spl_attr,'l', tree,spl_pt=spl_pt)
                self._assign_node(d2,spl_attr,'ge', tree,spl_pt=spl_pt)
            return tree

        else:
            attr_values = dataset[spl_attr].unique()
            for val in attr_values:
                data_partition = dataset[dataset[spl_attr] == val]
                data_partition = data_partition.drop(columns=[spl_attr])  # remove the splitting attribute
                if data_partition.shape[0] == 0:
                    tree[(spl_attr,spl_pt)][val] = data_partition[class_label_column].mode()[0]
                elif data_partition.shape[1] == 1:
                    tree[(spl_attr,spl_pt)][val] = data_partition[class_label_column].mode()[0]
                    # print('empty')
                else:
                    self._assign_node(data_partition,spl_attr,val, tree)  
            return tree


    def _assign_node(self, data_partition, spl_attr, attr_val, tree,spl_pt='_'):
        class_values = data_partition[class_label_column].unique()
        if len(class_values) == 1:
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
        if self.is_binary:
            for (key,pt) in dt:
                val = row[key]
                dt = dt[(key,pt)]
                if ('l' not in dt) and ('ge' not in dt):
                    dt = dt['_']
                elif val > pt:
                    dt = dt['ge']
                else:
                    dt = dt['l']
                if type(dt) is dict:
                    pred = self.predict_single(row, dt)
                else:
                    pred = dt
            return pred
        else:
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
        min_measure_val = float('inf')  #entropy or ginni
        splitting_attr = None
        splitting_pt = None
        for col in attr_cols:
            if col in self.numeric_cols:
                attr_measure_val, pt = entropy_attribute_cont(dataset,class_label_column,col)
            else:
                attr_measure_val,pt = entropy_attribute(dataset,class_label_column,col), '_'
            if attr_measure_val < min_measure_val:
                splitting_attr = col
                min_measure_val = attr_measure_val
                splitting_pt = pt

        return splitting_pt,splitting_attr

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
    data = load_dataset('Dataset/Chess_II/krkopt.data')
    # shuffle    ``
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.reset_index(drop=True)
    k = 10
    fold_size = data.shape[0] // k
    begin_index = 0
    end_index = begin_index + fold_size
    acc_list = []

    print('dataset size', len(data))
    class_label_column = 6
    iris_numeric_cols = [0,1,2,3]
    for i in range(k):
        # print(begin_index, end_index)
        test_frame = data[begin_index:end_index + 1].reset_index(drop=True)
        test_labels = test_frame[class_label_column]
        test_frame.drop(columns=class_label_column, inplace=True)
        train_frame = data.drop(data.iloc[begin_index:end_index + 1].index).reset_index(drop=True)
        # print(len(train_frame))
        dt = DecisionTree()
        dt.fit(train_frame, numeric_col_list=[], class_label_column=class_label_column)
        dt.print_tree()
        preds = dt.predict(test_frame)
        acc = calculate_accuracy(preds, list(test_labels))
        acc_list.append(acc)
        print()
        begin_index = end_index + 1
        if i == k - 2:
            end_index = data.shape[0] - 1
        else:
            end_index = end_index + fold_size
    print(acc_list)
    print(pd.Series(acc_list).mean())
