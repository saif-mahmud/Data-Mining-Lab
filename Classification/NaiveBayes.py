import timeit

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
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


def clf_report(y_true, y_pred):
    support = {}
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            if y_true[i] not in support:
                support[y_true[i]] = 0
            else:
                support[y_true[i]] += 1

    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    for label, sup_count in support.items():
        precision = sup_count / len(y_pred[y_pred == label])
        recall = sup_count / len(y_true[y_true == label])
        f1 = 2 * precision * recall / (precision + recall)
        print(label, 'precision:', precision, 'recall:', recall, 'f1: ', f1)


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
    X, y = load_dataset('Classification/Dataset/Connect4/connect-4.data',
                        class_label_column=42)

    num_idx = []
    categorical = [True] * len(X[0])

    for idx in range(len(categorical)):
        if idx in num_idx:
            categorical[idx] = False

    # print(categorical)

    cls_prob, cond_prob = train(X, y, categorical)

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    skf.get_n_splits(X, y)

    # print(skf)

    k = 0

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
        stop = timeit.default_timer()

        print(tabulate([['Accuracy (%)', accuracy_score(y_test, y_pred) * 100.0], ['Time Elapsed (Sec)', stop - start]],
                       tablefmt='grid',
                       headers=['k-Fold Cross Validation', k]))

        cls_label = list(np.unique(y_test))
        print(classification_report(y_test, y_pred, target_names=cls_label))

        # clf_report(y_test, y_pred)
        # print('\nConfusion Matrix')
        # pprint(confusion_matrix(y_test, y_pred))
        print('=====================================================')
