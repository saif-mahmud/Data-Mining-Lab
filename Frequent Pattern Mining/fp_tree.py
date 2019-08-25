from apriori import load_dataset, find_frequent_1_itemsets
import itertools
import sys

class Node:

    def __init__(self, name:int, parent):
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

    def __init__(self, db:list, min_sup):
        self.db = db
        self.min_sup = min_sup
        self.ordered_1_itemset = self.get_ordered_1_itemset() 
        self.root_node = Node(-1, None)
        self.node_link = {k:[] for k in self.ordered_1_itemset}

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
            next_node.frequency+=1
            self.create_branch(next_node, trx, index + 1)
        else:
            child = Node(trx[index],current_node)
            current_node.add_child(trx[index], child)
            self.node_link[child.name].append(child)
            self.create_branch(current_node.children[trx[index]], trx, index + 1)


    def _print(self, node: Node):
        if len(node.children) == 0:
            return
        else:
            # print(len(node.children))
            for c, nn in node.children.items():
                print(c, "parent: ", nn.parent.name, "parent_freq: ", nn.parent.frequency, "frequency: ", nn.frequency)
                # print(nn.children)
                self._print(nn)

    def get_conitional_db(self):
        conditional_db = dict()
        conditional_db = {k:[] for k in self.ordered_1_itemset}

        for i in reversed(self.ordered_1_itemset):
            for j in self.node_link[i]:
                cond_trx = self.get_predecessors(j, [])

                # if len(cond_trx) == 0:
                #     continue
                # print(cond_trx)
                for _ in range(j.frequency):
                    conditional_db[i].append(cond_trx)

                # print(i, cond_trx, j.frequency)

        return conditional_db
            
            
    def get_predecessors(self, current_node:Node, pred_list:list):
        if current_node.parent == self.root_node:
            return pred_list

        pred_list.append(current_node.parent.name)
        return self.get_predecessors(current_node.parent, pred_list)

        
    def get_branches(self):
        branches = []
        for k, v in self.node_link.items():
            for node in v:
                if node.is_leaf:
                    branches.append(list(reversed(self.get_predecessors(node,[node.name]))))
        
        # print('Branches :', branches)
        return branches



def fp_growth(item, cond_db:list, min_sup):
    l2 = [[item]]
    # cnt=0
    # for item_list in cond_db:
    #     if not item_list:
    #         cnt+=1
    #     if cnt>=min_sup:
    #         l2.append([item])
    #         break   
    #print("cond_data",cond_db,l2)
    fp_tree = FP_tree(cond_db, min_sup)
    fp_tree.build_fp_tree()
    #fp_tree._print(fp_tree.root_node)
    _cond_db_sub = fp_tree.get_conitional_db()
    #print('c_sub :', _cond_db_sub)

    for _item, _db in _cond_db_sub.items():
        l = fp_growth(_item, _db, min_sup)
        for i in l:
            # print('i :', i)
            i.append(item)
            l2.append(i)      
    return l2


def driver_fp_growth(db:list, min_sup):
    fp_tree = FP_tree(db, min_sup)
    fp_tree.build_fp_tree()

    proj_db_dict = fp_tree.get_conitional_db()

    for item_1, cond_db in proj_db_dict.items():
        ls = fp_growth(item_1, cond_db, min_sup)
        for l in ls:
            if len(l)==7:
                print(l)


if __name__ == "__main__":
    db = load_dataset(str(sys.argv[1]))
    min_sup = float(sys.argv[2])

    _min_sup = (len(db) * min_sup) // 100
    print('Min Sup :', _min_sup)

    driver_fp_growth(db, _min_sup)