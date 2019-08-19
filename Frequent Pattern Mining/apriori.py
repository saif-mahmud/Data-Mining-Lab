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


def apriori_gen(L:list, k):
    L_next = list()

    for l1 in L:
        for l2 in L:
            if len(l1 & l2) == (k - 1):
                L_next.append(l1 | l2)

    return L_next


def has_infrequent_subset(c:list, L:list):
    for candidate in c:
        for subset in list(itertools.combinations(candidate, len(candidate - 1))):
            if subset not in L:
                return True

    return False

if __name__ == "__main__":
    # d = load_dataset('Dataset/mushroom.dat')
    # find_frequent_1_itemsets(d, 5)

    L = [set([1, 2]), set([1, 3]), set([1, 5]), set([2, 3]), set([2, 4]), set([2, 5])]

    L3 = apriori_gen(L, 2)
    print(L3)