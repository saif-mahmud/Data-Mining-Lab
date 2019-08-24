from apriori import load_dataset, find_frequent_1_itemsets

class Node:

    def __init__(self, name):
        self.frequency = 0
        self.is_leaf = True
        self.children = {}
        self.name = name

    def add_child(self, name: int, child):
        self.is_leaf = False
        self.children[name] = child

class FP_tree:

    def __init__(self, db:list, min_sup):
        self.db = db
        self.min_sup = min_sup
        self.root = Node(-1)

    def sort_trx(self):
        L1 = find_frequent_1_itemsets(self.db, self.min_sup)
        sorted_L1 = sorted(L1, key=L1.get, reverse=True)
        
         