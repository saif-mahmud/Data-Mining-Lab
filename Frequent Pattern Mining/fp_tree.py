from apriori import load_dataset, find_frequent_1_itemsets
import itertools

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

                if len(cond_trx) == 0:
                    continue
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



# def freq_recursive(item, pattern:list):
#     if 

def is_single_path(nodelinks:dict):
    for k,v in nodelinks.items():
        if len(v) > 1:
            return False
    return True



def get_freq_patterns(cond_pattern_base, min_sup, heads:list):

    for item, c_db in cond_pattern_base.items():
        if len(c_db) == 0:
            continue
        fp_tree = FP_tree(c_db, min_sup)
        fp_tree.build_fp_tree()
        if is_single_path(fp_tree.node_link):
            print('found single_path_tree: ')
            print('current head: ', heads)
            fp_tree._print(fp_tree.root_node)
            heads = []
        new_cond_db = fp_tree.get_conitional_db()
        heads.append(item)
        get_freq_patterns(new_cond_db, min_sup, heads)

    return




def fp_growth(item, cond_db:list, min_sup):
    freq_list = list()

    if len(cond_db) == 0:
        freq_list.append([item])
        return freq_list

    fp_tree = FP_tree(cond_db, min_sup)
    fp_tree.build_fp_tree()

    _cond_db_sub = fp_tree.get_conitional_db()

    for _item, _db in _cond_db_sub.items():
        return fp_growth(_item, _db, min_sup)




if __name__ == "__main__":
    fptree = FP_tree(load_dataset('Han.dat'), 2)
    fptree.build_fp_tree()
    # print(fptree.node_link)
    fptree._print(fptree.root_node)
    x = fptree.get_conitional_db()
    print('Cond_DB :', x)

    ft2 = FP_tree(x[3], 2)
    ft2.build_fp_tree()

    ft2._print(ft2.root_node)
    y = ft2.get_conitional_db()

    print('cdb - I3 :', y)

    ft3 = FP_tree(x[1], 2)
    ft3.build_fp_tree()

    ft3._print(ft3.root_node)
    z = ft3.get_conitional_db()

    print('cdb :', z)

    l = fp_growth(3, x[3], 2)
    print('f :', l)
    
    # get_freq_patterns(fptree.get_conitional_db(), 2, [])

    # print(f)