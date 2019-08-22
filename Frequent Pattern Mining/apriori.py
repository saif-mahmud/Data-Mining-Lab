import collections
import itertools
from trie_class import Trie
import sys
import timeit


def load_dataset(filename):

    dataset = [sorted(int(n) for n in i.strip().split())
               for i in open(filename).readlines()]
    size = len(dataset)

    print('Size of the Dataset : ', size)

    total_len = 0

    for i in range(len(dataset)):
        total_len = total_len + len(dataset[i])

    avg_len = total_len / size
    print('Average Transaction Length : ', avg_len)

    # print(dataset)
    return dataset


def find_frequent_1_itemsets(dataset, min_sup):
    min_sup = (len(dataset) * min_sup) // 100
    frequency = dict(collections.Counter(
        itertools.chain.from_iterable(dataset)))

    L1 = dict()

    for item, freq in frequency.items():
        if freq >= min_sup:
            L1[item] = freq

    # print(L1)
    return L1

# Input : L_k (k : itemset size)


def apriori_gen(L: list, k):
    # Self Join Step
    L_next = list()

    for l1 in L:
        for l2 in L:
            if len(set(l1) & set(l2)) == (k - 1):
                L_next.append(list(set(l1) | set(l2)))

    # Removing Duplicates
    L_set = set(tuple(x) for x in L_next)
    L_k1 = [list(x) for x in L_set]

    L_k1.sort(key=lambda x: L_next.index(x))
    L_k1_tuple = [tuple(i) for i in L_k1]

    # Prune Step
    for c in L_k1_tuple:
        if has_infrequent_subset(c, L):
            L_k1.remove(list(c))

    # Returns list of lists [L_k + 1]
    return L_k1


def has_infrequent_subset(candidate: tuple, L: list):
    for subset in list(itertools.combinations(candidate, len(candidate) - 1)):
        if list(subset) not in L:
            return True

    return False


def apriori(db: list, min_sup):
    min_sup = (len(db) * min_sup) // 100

    levels = list()

    L1 = find_frequent_1_itemsets(db, min_sup)

    if bool(L1) == False:
        print('No 1-Itemset Satisfies Given Minimum Support Threshold')
        return None
    
    # Creating list of 1-itemset(list itself)
    _L1 = [[k] for k in L1.keys()]
    _L1 = sorted(_L1)

    # print('L1 :', L1)

    levels.append(_L1)
    # print('Levels :', levels)
    

    while True:
        candidates = apriori_gen(levels[-1], len(levels[-1][0]))

        trie = Trie(db)
        trie.build_trie(candidates)
        trie.assign_frequency()

        L = list()

        for itemset in candidates:
            if trie.get_candidate_freq(itemset) >= min_sup:
                L.append(itemset)
                
        if not L:
            break

        levels.append(L)

    return levels


if __name__ == "__main__":
    db = load_dataset(str(sys.argv[1]))
    min_sup = float(sys.argv[2])
    
    print('Dataset :', str(sys.argv[1]))
    print('Min Support :', min_sup, '%')

    start = timeit.default_timer()

    L = apriori(db, min_sup)

    stop = timeit.default_timer()

    if L is not None:
        for i in range(len(L)):
            print((i + 1), '- Frequent Itemsets :', L[i])

    print('\nTime: ', stop - start, " seconds")
