import collections
import itertools

def load_dataset(filename):

    dataset = [i.strip().split() for i in open(filename).readlines()]
    size = len(dataset)

    print('Size of the Dataset : ', size)

    total_len = 0

    for i in range(len(dataset)):
        total_len = total_len + len(dataset[i])

    avg_len = total_len / size
    print('Average Transaction Length : ', avg_len)

    frequency = collections.Counter(itertools.chain.from_iterable(dataset))
    print(frequency)

load_dataset('Dataset/mushroom.dat')
