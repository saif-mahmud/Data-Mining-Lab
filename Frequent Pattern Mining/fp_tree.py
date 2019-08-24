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
        child.frequency=1
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
        if current_node.parent.name == -1:
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

def get_freq_patterns(cond_pattern_base, min_sup):

    freq = list()

    for item, c_db in cond_pattern_base.items():
        fp_tree = FP_tree(c_db, min_sup)
        fp_tree.build_fp_tree()
        

        branches = fp_tree.get_branches()
        # print(item, 'Branch :', branches)

        for b in branches:
            # print('b :', b)

            for l in range(1, len(b) + 1):
                comb = list(itertools.combinations(b, l))
                # print('Comb :', comb)

                for s in comb:
                    _comb = list(s)
                    # print('_c1 :', _comb)
                    _comb.append(item)
                    # print('_c2 :', _comb)

                    # print('U : ', _comb)

                    freq.append(_comb)
                
                # print('Freq :', freq)

    return freq


        



if __name__ == "__main__":
    fptree = FP_tree(load_dataset('Han.dat'), 1)
    fptree.build_fp_tree()
    # print(fptree.node_link)
    # fptree._print(fptree.root_node)
    
    f = get_freq_patterns(fptree.get_conitional_db(), 2)

    print(f)