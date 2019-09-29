import timeit

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tabulate import tabulate


def load_dataset(file: str, class_label_column):
    dataset = pd.read_csv(file, header=None)

    print(tabulate([['Dataset Size', dataset.shape[0]], ['# of Attributes', dataset.shape[1] - 1]], tablefmt='grid',
                   headers=['Dataset Summary', file]))

    feature_columns = list(dataset.columns.values)
    feature_columns.remove(class_label_column)

    X = dataset.iloc[:, feature_columns].values
    y = dataset.iloc[:, class_label_column].values

    return X, y


def compute_probability_distribution(y: np.ndarray):
    labels, cnt = np.unique(y, return_counts=True)
    cls_prob = cnt / len(y)

    prob_dict = dict(zip(labels, cls_prob))

    return prob_dict


def groupby_class(X: np.ndarray, y: np.ndarray):
    cls_split = dict()
    cls_tuples = dict()

    for label in np.unique(y):
        cls_split[label] = np.where(y == label)[0]
        cls_tuples[label] = X[cls_split[label]]

    return cls_tuples


def feature_extraction(cls_tups: dict, categorical: list):
    attr_dict = dict()
    for label in cls_tups.keys():
        attr_dict[label] = {}

    for label, features in cls_tups.items():
        for i in range(len(features[0])):
            if categorical[i]:
                attr_dict[label][i] = compute_probability_distribution(features[:, i])
            else:
                attr_dict[label][i] = {'mean': np.mean(features[:, i]), 'std': np.std(features[:, i])}

    return attr_dict


def train(X: np.ndarray, y: np.ndarray, categorical: list):
    cls_probs = compute_probability_distribution(y)
    cls_tups = groupby_class(X, y)

    attr_dick = feature_extraction(cls_tups, categorical)

    return cls_probs, attr_dick


def gaussian_distribution(x, mean, std):
    probability = (1 / (np.sqrt(2 * np.pi) * std)) * (np.exp((-(x - mean) ** 2) / (2 * (std ** 2))))

    return probability


def predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, categorical: list):
    cls_prob, cond_prob = train(X_train, y_train, categorical)

    tab = []
    for label, prob in cls_prob.items():
        tab.append([label, prob])

    print(tabulate(tab, tablefmt='psql', headers=['Class Label', 'Probability Distribution']))

    # print('Class Probability')
    # pprint(cls_prob)

    # print('Posterior Probability : ')
    # pprint(cond_prob)

    y_pred = list()

    for test_data_feature in X_test:
        # print('\nTest Data Feature :')
        # pprint(test_data_feature)

        max_prob = -float('inf')
        pred_class = None

        for label in cls_prob.keys():

            prior_prob = cls_prob[label]
            posterior_prob = 1.0

            for idx in range(len(test_data_feature)):
                if categorical[idx]:
                    if test_data_feature[idx] in cond_prob[label][idx].keys():
                        posterior_prob *= cond_prob[label][idx][test_data_feature[idx]]
                    else:
                        cls_cnt = np.count_nonzero(y_train == label)
                        uniq_cnt = len(cond_prob[label][idx].keys())
                        posterior_prob *= (1 / (cls_cnt + uniq_cnt + 1))
                else:
                    posterior_prob *= gaussian_distribution(test_data_feature[idx], cond_prob[label][idx]['mean'],
                                                            cond_prob[label][idx]['std'])

            # print('Posterior Probability [', label, '] :', posterior_prob)
            bayes_prob = prior_prob * posterior_prob
            # print('P(', test_data_feature, '|', label, ') :', bayes_prob)

            if bayes_prob > max_prob:
                max_prob = bayes_prob
                pred_class = label

        # print('Predicted Class :', pred_class)
        y_pred.append(pred_class)

    return np.array(y_pred)


if __name__ == '__main__':
    X, y = load_dataset('Classification/Dataset/Poker/poker-hand-testing.data', class_label_column=10)

    num_idx = []
    # [0, 2, 4, 10, 11, 12]
    categorical = [True] * len(X[0])

    for idx in range(len(categorical)):
        if idx in num_idx:
            categorical[idx] = False

    # print(categorical)

    cls_prob, cond_prob = train(X, y, categorical)
    # gnb = GaussianNB()

    skf = StratifiedKFold(n_splits=10, shuffle=True)
    skf.get_n_splits(X, y)

    # print(skf)

    k = 5

    for train_index, test_index in skf.split(X, y):
        print('\n\n\n')
        k = k + 1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # print('X train :')
        # pprint(X_train)

        # print('y - Train :')
        # pprint(y_train)

        # print('X - test :')
        # pprint(X_test)

        start = timeit.default_timer()
        y_pred = predict(X_train, y_train, X_test, categorical)
        # _y_pred = gnb.fit(X_train, y_train).predict(X_test)
        stop = timeit.default_timer()

        print(tabulate([['Accuracy (%)', accuracy_score(y_test, y_pred) * 100.0], ['Time Elapsed (Sec)', stop - start]],
                       tablefmt='grid',
                       headers=['k-Fold Cross Validation', k]))

        # print('Acc [Scikit] :', accuracy_score(y_test, _y_pred))
        # for i in range(len(y_pred)):
        #     if type(y_pred) != str:
        #         y_pred[i] = str(y_pred[i])
        #         y_test[i] = str(y_test[i])

        cls_label = list(np.unique(y_test))
        # print(classification_report(y_test, y_pred, target_names=cls_label))
        print('=====================================================')
