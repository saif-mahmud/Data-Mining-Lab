class Node:

    def __init__(self, name):
        self.frequency = 0
        self.is_leaf = True
        self.children = {}
        self.name = name

    def add_child(self, name: int, child):
        self.is_leaf = False
        self.children[name] = child


class Trie:

    def __init__(self, db: list):
        self.db = db
        self.root_node = Node(-1)

    def build_trie(self, candidates: list):

        for c_list in candidates:
            self.create_branch(self.root_node, c_list, 0)

    def create_branch(self, current_node: Node, c_list: list, index):
        if index == len(c_list):
            return

        if(c_list[index] in current_node.children.keys()):
            self.create_branch(
                current_node.children[c_list[index]], c_list, index + 1)
        else:
            current_node.add_child(c_list[index], Node(c_list[index]))
            self.create_branch(
                current_node.children[c_list[index]], c_list, index + 1)

    def _print(self, node: Node):
        if len(node.children) == 0:
            return
        else:
            # print(len(node.children))
            for c, nn in node.children.items():
                print(c, "parent: ", node.name, "frequency: ",
                      nn.frequency, "leaf: ", nn.is_leaf)
                # print(nn.children)
                self._print(nn)

    def assign_frequency(self):
        for trx in self.db:
            self.traverse(self.root_node, set(trx))

    def traverse(self, current_node: Node, trx):
        if current_node.is_leaf:
            current_node.frequency += 1
            return

        for child in current_node.children.keys():
            if child in trx:
                self.traverse(current_node.children[child], trx)

    def get_candidate_freq(self, candidate):
        return self.single_traverse(self.root_node, candidate, 0)

    def single_traverse(self, current_node, candidate, index):
        if current_node.is_leaf:
            return current_node.frequency
        return self.single_traverse(current_node.children[candidate[index]], candidate, index + 1)
