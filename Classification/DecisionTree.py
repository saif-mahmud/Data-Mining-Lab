import pandas as pd
from pprint import pprint
from attribute_selection import gain


def load_datadet(filename:str):
    df = pd.read_csv(filename, header=None)
    return df


class DecisionTree:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_tree = None
    

    def fit(self,dataset:pd.DataFrame, class_label_column):
        self.decision_tree = self._train_recursive(dataset, class_label_column)
        

    def _train_recursive(self,dataset:pd.DataFrame, class_label_column, tree=None):
        # print(dataset.shape)
        spl_attr = self._find_splitting_attribute(dataset, class_label_column)
        attr_values = dataset[spl_attr].unique()
        # print(spl_attr, attr_values)

        if tree is None:
            tree = {}
            tree[spl_attr] = {}

        for val in attr_values:
            data_partition = dataset[dataset[spl_attr] == val]
            data_partition = data_partition.drop(columns=[spl_attr]) #remove the splitting attribute
            if data_partition.shape[0] == 0:
                tree[spl_attr][val] =  'majority'
                # print('empty')
            class_values = data_partition[class_label_column].unique()
            if len(class_values ) == 1:
                tree[spl_attr][val] = class_values[0]
                # print('pure class')
            else:
                #recursion
                tree[spl_attr][val] = self._train_recursive(data_partition, class_label_column)
        return tree



    def _find_splitting_attribute(self, dataset:pd.DataFrame, class_label_column):
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
        return splitting_attr

    
    def predict(x):
        pass


    def print_tree(self):
        pprint(self.decision_tree)


if __name__ == '__main__':
    data = load_datadet('Dataset/Mushroom/agaricus-lepiota.data')

    # info = entropy(data, 0)
    # print(info)

    dt = DecisionTree()

    tree = dt.fit(data, class_label_column=0)
    dt.print_tree()

