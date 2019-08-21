import collections
import itertools

def load_dataset(filename):

    dataset = [sorted(int(n) for n in i.strip().split()) for i in open(filename).readlines()]
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
    frequency = dict(collections.Counter(itertools.chain.from_iterable(dataset)))
    
    L1 = dict()

    for item, freq in frequency.items():
        if freq > min_sup:
            L1[item] = freq


    # print(L1)
    return L1

# Input : L_k (k : itemset size)
def apriori_gen(L:list, k):
    # Self Join Step
    L_next = list()

    for l1 in L:
        for l2 in L:
            if len(set(l1) & set(l2)) == (k - 1):
                L_next.append(list(set(l1) | set(l2)))

    # Removing Duplicates
    L_set = set(tuple(x) for x in L_next)
    L_k1 = [ list(x) for x in L_set ]

    L_k1.sort(key = lambda x: L_next.index(x) )
    L_k1_tuple = [tuple(i) for i in L_k1]

    # Prune Step
    for c in L_k1_tuple:
        if has_infrequent_subset(c, L):
            L_k1.remove(list(c))

    # Returns list of lists [L_k + 1]
    return L_k1


def has_infrequent_subset(candidate:tuple, L:list):
    for subset in list(itertools.combinations(candidate, len(candidate) - 1)):
        if list(subset) not in L:
            return True

    return False

if __name__ == "__main__":
    D = load_dataset('Dataset/chess.dat')

    L = [[1, 2], [1, 3], [1, 5], [2, 3], [2, 4], [2, 5]]

    L3 = apriori_gen(L, 2)
    print(L3)