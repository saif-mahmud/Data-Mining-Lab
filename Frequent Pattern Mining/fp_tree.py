from apriori import load_dataset, find_frequent_1_itemsets
import itertools
import sys
import timeit


class Node:

    def __init__(self, name: int, parent):
        self.name = name
        self.parent = parent
        self.children = {}
        self.frequency = 0
        self.is_leaf = True

    def add_child(self, name: int, child):
        child.frequency = 1
        self.is_leaf = False
        self.children[name] = child


class FP_tree:

    def __init__(self, db: list, min_sup):
        self.db = db
        self.min_sup = min_sup
        self.ordered_1_itemset = self.get_ordered_1_itemset()
        self.root_node = Node(-1, None)
        self.node_link = {k: [] for k in self.ordered_1_itemset}

    def get_ordered_1_itemset(self):
        L1 = find_frequent_1_itemsets(self.db, self.min_sup)
        # print(L1)
        sorted_L1 = sorted(L1, key=L1.get, reverse=True)
        # print(sorted_L1)

        return sorted_L1

    def build_fp_tree(self):

        for trx in self.db:
            if len(trx) == 0:
                continue
            trx = set(trx)
            _trx = []
            for i in self.ordered_1_itemset:
                if i in trx:
                    _trx.append(i)
            self.create_branch(self.root_node, _trx, 0)

    def create_branch(self, current_node: Node, trx: list, index):
        if index == len(trx):
            return

        if(trx[index] in current_node.children.keys()):
            next_node = current_node.children[trx[index]]
            next_node.frequency += 1
            self.create_branch(next_node, trx, index + 1)
        else:
            child = Node(trx[index], current_node)
            current_node.add_child(trx[index], child)
            self.node_link[child.name].append(child)
            self.create_branch(
                current_node.children[trx[index]], trx, index + 1)

    def _print(self, node: Node):
        if len(node.children) == 0:
            return
        else:
            # print(len(node.children))
            for c, nn in node.children.items():
                print(c, "parent: ", nn.parent.name, "parent_freq: ",
                      nn.parent.frequency, "frequency: ", nn.frequency)
                # print(nn.children)
                self._print(nn)

    def get_conitional_db(self):
        conditional_db = dict()
        conditional_db = {k: [] for k in self.ordered_1_itemset}

        for i in reversed(self.ordered_1_itemset):
            for j in self.node_link[i]:
                cond_trx = self.get_predecessors(j, [])

                for _ in range(j.frequency):
                    conditional_db[i].append(cond_trx)


        return conditional_db

    def get_predecessors(self, current_node: Node, pred_list: list):
        if current_node.parent == self.root_node:
            return pred_list

        pred_list.append(current_node.parent.name)
        return self.get_predecessors(current_node.parent, pred_list)

cond_cnt = 0

def fp_growth(item, cond_db: list, min_sup):
    l2 = [[item]]
    global cond_cnt

    fp_tree = FP_tree(cond_db, min_sup)
    fp_tree.build_fp_tree()
    
    for li in cond_db:
        if li!=[]:
            break
    else: 
        cond_cnt += 1
    
    _cond_db_sub = fp_tree.get_conitional_db()
    
    for _item, _db in _cond_db_sub.items():
        l = fp_growth(_item, _db, min_sup)
    
        for i in l:
            i.append(item)
            l2.append(i)
    

    return l2


def driver_fp_growth(db: list, min_sup):
    freq_cnt = 0

    fp_tree = FP_tree(db, min_sup)
    fp_tree.build_fp_tree()

    proj_db_dict = fp_tree.get_conitional_db()


    for item_1, cond_db in proj_db_dict.items():
        ls = fp_growth(item_1, cond_db, min_sup)

        # print('\n')
        # print('Itemset -', item_1, ':')
        # print('Conditional Pattern Base Size:', len(cond_db))
        freq_cnt += len(ls)
        # print('Frequent Patterns Generated :', len(ls))
        # print(ls)

    print('\nSummary ')
    print('==========================================')
    print('Total # of Conditional Tree Built :', cond_cnt)
    print('Total # of Frequent Patterns :', freq_cnt)


if __name__ == "__main__":
    db = load_dataset(str(sys.argv[1]))
    min_sup = float(sys.argv[2])

    _min_sup = (len(db) * min_sup) // 100
    print('Min Sup :', _min_sup)

    start = timeit.default_timer()

    driver_fp_growth(db, _min_sup)

    stop = timeit.default_timer()
    print('\nTime: ', stop - start, " seconds")
