import numpy as np
import pandas as pd
np.random.seed(0)

class Node:
    def __init__(self, dim):
        '''individual nodes of the tree'''
        self.left = None
        self.right = None
        self.parent = None
        self.node_label = 0
        self.dimension = dim
        self.w = np.float32(np.random.normal(0, 1, (dim)) * np.sqrt(2.0 / (dim)))
        self.data_labels = None
        self.target_labels = None
        self.x_indices = None
        self.is_leaf = True
        self.has_converged = False

        # test data
        self.test_data_labels = None
        self.test_target_labels = None
        self.test_x_indices = None

class Tree(Node):
    '''the tree'''
    def __init__(self, dim):
        super().__init__(dim)
        self.leaves = [0]

    def splitNode(self, node_label, root):
        '''splits a node into two children'''
        node = self.findNode(node_label, root)
        if node is None:
            return
        child_left = Node(self.dimension)
        child_left.node_label = 2 * node_label + 1
        child_right = Node(self.dimension)
        child_right.node_label = 2 * node_label + 2
        node.left = child_left
        node.right = child_right
        child_left.parent = node
        child_right.parent = node
        self.leaves.remove(node.node_label)
        self.leaves.append(child_left.node_label)
        self.leaves.append(child_right.node_label)

    def get_leaves(self):
        return self.leaves

    def findNode(self, node_label, root):
        '''finds a node based on label'''
        if root == None or root.node_label == node_label:
            return root
        left_findNode = self.findNode(node_label, root.left)
        right_findNode = self.findNode(node_label, root.right)
        if left_findNode == None and right_findNode == None:
            return None
        elif left_findNode == None:
            return right_findNode
        return left_findNode

    def forward(self, root, x, x_indices):
        '''forward pass for training'''
        if root is None:
            return
        if x_indices.shape[0] == 0 or x is None:
            return
        if root.left is not None and root.right is not None:
            root.is_leaf = False
        root.x_indices = x_indices
        root.target_labels = np.zeros(x.shape[1])
        data_x = x[:, x_indices]
        w_transpose_x = np.dot(root.w, data_x)
        data_labels = np.sign(w_transpose_x)
        root.data_labels = np.zeros(x.shape[1])
        root.data_labels[x_indices] = data_labels
        if root.parent is not None:
            root.data_labels[x_indices] = root.data_labels[x_indices] * root.parent.data_labels[x_indices]
        pos_indices = np.where(root.data_labels == 1)[0]
        neg_indices = np.where(root.data_labels == -1)[0]
        self.forward(root.left, x, neg_indices)
        self.forward(root.right, x, pos_indices)

    def count_nodes(self, root):
        '''finds number of nodes in the tree'''
        if root == None:
            return 0
        return 1 + self.count_nodes(root.left) + self.count_nodes(root.right)

    def find_data_labels_train(self, root, num_labels):
        if root is None or root.x_indices is None:
            return np.zeros(num_labels)
        if root.is_leaf == True and root.x_indices is not None:
            return root.data_labels
        l_labels = self.find_data_labels_train(root.left, num_labels)
        r_labels = self.find_data_labels_train(root.right, num_labels)
        return l_labels + r_labels

    def count_examples_per_node(self, root, hyp):
        if root == None:
            return
        print(root.node_label)
        print(len(root.x_indices))
        hyp.append(root.w)
        self.count_examples_per_node(root.left, hyp)
        self.count_examples_per_node(root.right, hyp)


def main():
    n_nodes = 10
    p_split = 0.9
    dim = 5 + 1
    tree = Tree(dim)
    n_points = 100000
    data_x = np.ones((dim, n_points))
    data_x[0:-1, :] = -1 + 2*np.random.random((dim-1, n_points))
    while tree.count_nodes(tree) < n_nodes:
        leaf_nodes_labels = tree.get_leaves()
        for i in range(0, len(leaf_nodes_labels)):
            should_split = np.random.random()
            if should_split<=0.1:
                continue
            tree.splitNode(leaf_nodes_labels[i], tree)
            leaf_nodes_labels = tree.get_leaves()
            tree.forward(tree, data_x, np.arange(0, data_x.shape[1]))
            for j in range(0, len(leaf_nodes_labels)):
                node = tree.findNode(leaf_nodes_labels[j], tree)
                mean = np.sum(data_x[:, node.x_indices], axis=1)/data_x[:, node.x_indices].shape[1]
                node.w[-1] = -1*np.dot(node.w[0:-1], mean[0:-1])
    tree.forward(tree, data_x, np.arange(0, data_x.shape[1]))
    data_y = tree.find_data_labels_train(tree, data_x.shape[1])
    hyp=[]
    tree.count_examples_per_node(tree, hyp)
    hyp = np.asarray(hyp).T
    print(hyp.shape)
    data = np.hstack((data_x[0:-1, :].T, data_y.reshape(-1, 1)))
    df = pd.DataFrame(data)
    df.to_csv('ten_5D_hyperplanes_million.csv', header=False, index=False)
    df = pd.DataFrame(hyp)
    df.to_csv('ten_5D_hyperplanes_hyperplanes_million.csv', header=False, index=False)

if __name__ == '__main__':
    main()