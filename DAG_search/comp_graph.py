'''
The computational graph class
'''
import numpy as np
import copy
import warnings
from sympy import simplify
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from DAG_search import config
from expr_cal.expression import *
from torch_utils.pytorch_utils import *
from torch_utils.transform_expr import *
from expr_cal.stock_data import Stock_Data
warnings.simplefilter("ignore")

class CompGraph():

    def __init__(self, m, n, node_dict = {}):
        '''
        m... input dimension
        n... output dimension
        '''

        self.inp_dim = m
        self.outp_dim = n

        self.node_dict = node_dict # key = node_idx, value = list of children , operation type

        self.R = None # reachability matrix (number of paths from node i to node j)
        self.A = None # adjacency matrix
        self.eval_order = None # index order in which nodes are evaluated

        self.outp_nodes = []
        self.inp_nodes = []
        self.successors = {}
        self.predecessors = {}

        self.eval_strs = []
        self.eval_funcs = [] # numpy lambdas
        self.eval_funcs_pt = [] # pytorch lambdas

        if len(self.node_dict) > 0:
            self.update_stats()

    def n_nodes(self):
        return len(self.node_dict)

    def n_edges(self):
        ret = 0
        for i in self.successors:
            ret += len(self.successors[i])
        return ret

    def n_operations(self):
        return len(self.node_dict) - self.inp_dim

    def depth(self):
        k = self.inp_dim
        depth_dict = {i : 0 for i in self.node_dict}

        for i in self.eval_order[k:]:
            children, node_op = self.node_dict[i]
            node_depth = max([depth_dict[j] for j in children])
            if node_op != '=':
                node_depth += 1

            depth_dict[i] = node_depth

        return max([depth_dict[i] for i in self.outp_nodes])

    def n_subexps(self):
        m, n = self.inp_dim, self.outp_dim
        counts = {i : 0 for i in self.node_dict if i >= m + n}
        for i in self.node_dict:
            parents, _ = self.node_dict[i]
            for p in parents:
                if p in counts:
                    counts[p] += 1
        return (np.array([counts[i] for i in counts]) > 1).sum()

    def copy(self):
        ret = CompGraph(self.inp_dim, self.outp_dim)
        ret.node_dict = copy.deepcopy(self.node_dict)

        ret.update_stats()
        return ret

    def update_stats(self):

        # update A and R
        n_nodes = len(self.node_dict)
        self.A = np.zeros((n_nodes, n_nodes))
        self.R = np.eye(n_nodes)
        for i in range(n_nodes):
            children = self.node_dict[i][0]
            for j in children:
                self.A[j, i] += 1
                self.R += np.outer(self.R[:, i], self.R[j,:])

        # update eval_order
        self.eval_order = []
        for order in self.get_eval_order():
            self.eval_order = self.eval_order + order

        # set node functionalities
        self.outp_nodes = [self.inp_dim + i for i in range(self.outp_dim)]
        self.inp_nodes = list(range(self.inp_dim))

        self.successors = {i : [] for i in self.node_dict}
        self.predecessors = {i : [] for i in self.node_dict}
        for i in self.node_dict:
            for j in self.node_dict[i][0]:
                self.successors[j].append(i)
                self.predecessors[i].append(j)

        # set eval function
        self.set_eval_funcs()

    def set_eval_funcs(self):
        # 将图转化为可计算的表达式
        k = self.inp_dim

        res_dict = {i : '' for i in self.node_dict}

        # inputs
        for i in range(self.inp_dim):
            res_dict[i] = f'X_full[:, :, {i}]'

        for i in self.eval_order[k:]:
            children, op = self.node_dict[i]
            node_op = config.NODE_STR[op]
            if len(children) == 2:
                if (children[0] == children[1]) and op == '*':
                    node_op = '(a)**2' # shortcut
                elif (children[0] == children[1]) and op == '+':
                    node_op = '2*(a)'
                node_result = node_op.replace('a', res_dict[children[0]]).replace('b', res_dict[children[1]])
            else:
                assert len(children) == 1
                node_result = node_op.replace('a', res_dict[children[0]])
            res_dict[i] = node_result

        eval_strs = [res_dict[i] for i in self.outp_nodes]
        self.eval_strs = eval_strs

    def get_eval_order(self):
        # get layers of nodes

        eval_order = []
        eval_idx = set() # already evaluated indices
        remain_idx = set(range(len(self.node_dict))) # yet to be evaluated

        # inputs
        round_list = []
        for i in range(self.inp_dim):
            round_list.append(i)
            eval_idx.add(i)
            remain_idx.remove(i)
        eval_order.append(round_list)

        while len(remain_idx) > 0:
            next_idx = []
            for i in remain_idx:
                # are all children evaluated?
                valid = True
                for j in self.node_dict[i][0]:
                    if j not in eval_idx:
                        valid = False
                        break
                if valid:
                    next_idx.append(i)

            for i in next_idx:
                eval_idx.add(i)
                remain_idx.remove(i)
            eval_order.append(next_idx)
        return eval_order

    def get_subexps(self):
        eval_order = self.get_eval_order()
        n_paths = (self.R[self.outp_nodes] > 0).sum(axis=0)
        node_ids = np.arange(0, len(self.node_dict), 1)
        subexp_ids = []
        for i, nodes in enumerate(eval_order):
            mask = (n_paths[nodes] > 1)
            if np.any(mask):
                subexp_ids.append(np.array(nodes)[mask])
            else:
                break
        subexps = [self.evaluate_symbolic(ret_nodes = order) for order in subexp_ids]
        return subexps

    def max_subexp_depth(self, eval_order = None):
        if eval_order is None:
            eval_order = self.get_eval_order()
        max_depth = len(eval_order)
        n_paths = (self.R[self.outp_nodes] > 0).sum(axis=0)
        current_depth = 0
        for i, nodes in enumerate(eval_order):
            mask = (n_paths[nodes] > 1)
            if np.any(mask):
                current_depth = (i+1)
        return current_depth

    def subexp_ratio(self):
        eval_order = self.get_eval_order()
        max_depth = len(eval_order)
        return self.max_subexp_depth(eval_order)/max_depth

    # Evaluation

    def get_invalids(self, X):
        '''
        Returns first occurence of a non-finite result as
        node, predecessors

        X... N x m matrix
        '''
        assert len(self.node_dict) >= self.inp_dim + self.outp_dim, 'Node dict not initialized'

        r = 1

        # we assume that eval order is valid

        N = X.shape[0]
        k = self.inp_dim

        res_dict = {i : None for i in self.node_dict}

        # inputs
        for i in range(self.inp_dim):
            res_dict[i] = np.repeat(X[np.newaxis, :, i], r, axis=0) # r x N

        # others
        for i in self.eval_order[k:]:
            children, op = self.node_dict[i]
            node_op = config.NODE_OPS[op]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if len(children) == 2:
                    node_result = node_op(res_dict[children[0]], res_dict[children[1]])
                else:
                    assert len(children) == 1
                    node_result = node_op(res_dict[children[0]])
                res_dict[i] = node_result
                if not np.all(np.isfinite(node_result)):
                    return [i for i in np.where(self.R[i] > 0)[0] if i >= self.inp_dim]
        return []

    def evaluate(self, data: Stock_Data):
        '''
        Evaluates the symbolic expression(s) represented by this graph on the given Stock_Data.

        return_grad... kept for API compatibility, ignored.
        '''
        assert len(self.node_dict) >= self.inp_dim + self.outp_dim, 'Node dict not initialized'

        expr = transform(self.evaluate_symbolic()[0])
        #print(self.evaluate_symbolic()[0], expr)
        try:
            value = (eval(expr).evaluate(data)).values
            invalues = np.isinf(value) | np.isnan(value)
            value[invalues] = 0
            final_result = normalize_by_day(value, data)
        except Exception as e:
            #print(f'error: {e}')
            final_result = Constant(0).evaluate(data).values

        return final_result

    def evaluate_symbolic(self, X = None, ret_nodes = None):
        '''
        X... list of sympy symbols
        '''
        if X is None:
            X = [f'x_{i}' for i in range(self.inp_dim)]
        if ret_nodes is None:
            ret_nodes = self.outp_nodes

        if len(self.node_dict) < self.inp_dim + self.outp_dim:
            return None

        # we assume that eval order is valid
        k = self.inp_dim

        res_dict = {i : None for i in self.node_dict}
        # inputs
        for i in range(self.inp_dim):
            res_dict[i] = X[i]

        for i in self.eval_order[k:]:
            children, op = self.node_dict[i]
            node_op = config.NODE_STR[op]
            if len(children) == 2:
                if (children[0] == children[1]) and op == '*':
                    node_op = '(a)**2' # shortcut
                elif (children[0] == children[1]) and op == '+':
                    node_op = '2*(a)'
                node_result = node_op.replace('(a)', res_dict[children[0]]).replace('(b)', res_dict[children[1]])
            else:
                assert len(children) == 1
                node_result = node_op.replace('(a)', res_dict[children[0]])
            res_dict[i] = node_result

        return [res_dict[i] for i in ret_nodes]


def is_equal(cgraph1:CompGraph, cgraph2:CompGraph, use_sympy:bool = False):
    '''
    Checks if two graphs are equal
    '''

    if not use_sympy:

        dict1 = cgraph1.node_dict
        dict2 = cgraph2.node_dict


        if len(dict1) != len(dict2):
            return False

        k1 = sorted(dict1.keys())
        k2 = sorted(dict2.keys())

        for i1, i2 in zip(k1, k2):
            if i1 != i2:
                return False
            else:
                l1, op1 = dict1[i1]
                l2, op2 = dict2[i2]
                if (sorted(l1), op1) != (sorted(l2), op2):
                    return False

        return True
    else:
        tmp1 = cgraph1.evaluate_symbolic()
        tmp2 = cgraph2.evaluate_symbolic()
        return np.all([(simplify(exp1-exp2) == 0) for exp1, exp2 in zip(tmp1, tmp2)])

def change_operation(cgraph):
    bin_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 2]
    un_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 1]

    ret_graph = cgraph.copy()

    # 1. select node
    comp_nodes = [i for i in range(cgraph.inp_dim, len(cgraph.node_dict))]
    node_idx = np.random.choice(comp_nodes)

    # 2. change operation
    parents, op = cgraph.node_dict[node_idx]
    if len(parents) == 2:
        new_op = np.random.choice([x for x in bin_ops if x != op])
    else:
        new_op = np.random.choice([x for x in un_ops if x != op])

    ret_graph.node_dict[node_idx] = (parents, new_op)
    return ret_graph

def change_edge(cgraph):
    ret_graph = cgraph.copy()

    # 1. select node
    comp_nodes = [i for i in range(cgraph.inp_dim, len(cgraph.node_dict))]
    node_idx = np.random.choice(comp_nodes)

    # 2. select new parent
    parents, op = cgraph.node_dict[node_idx]
    parent_idx = np.random.randint(len(parents))
    p_old = parents[parent_idx]
    possible_parents = [i for i in cgraph.eval_order[:cgraph.eval_order.index(node_idx)] if i != p_old]
    p_new = np.random.choice(possible_parents)

    if len(parents) == 1:
        new_parents = [p_new]
    else:
        new_parents = sorted([p_new, parents[int(1-parent_idx)]])


    ret_graph.node_dict[node_idx] = (new_parents, op)
    return reduce_graph(ret_graph)

def reduce_graph(comp_graph:CompGraph) -> CompGraph:
    '''
    Reduces the computational graph in 2 steps:
    1.  nodes with the same paths leading to them are collapsed into one,
        identity nodes that are no output are collapsed with their successor
    2.  nodes that have no path to an output node are removed
    '''
    cgraph = comp_graph.copy()

    nodes1 = [i for i in cgraph.node_dict if i >= cgraph.inp_dim + cgraph.outp_dim] # if match they are removed
    nodes2 = [i for i in cgraph.node_dict if i >= cgraph.inp_dim] # if match they take edges from counterpart

    # 1. collapse doubled nodes

    # H is a list of tuples (keep, remove)
    H = []
    for v in nodes1:
        for u in nodes2:
            if u > v:
                l1, op1 = cgraph.node_dict[u]
                l2, op2 = cgraph.node_dict[v]
                if (sorted(l1), op1) == (sorted(l2), op2):
                    H.append((u,v))

    # identity nodes that are no output nodes
    for v in cgraph.node_dict:
        children, op = cgraph.node_dict[v]

        if (op == '=') and (v not in cgraph.outp_nodes):
            u = children[0] # should be the only child
            H.append((u,v))

    if len(H) > 0:
        # 1. resolve H
        # delete v
        transl_dict = {}
        for u, v in H:
            if v in cgraph.node_dict:
                del cgraph.node_dict[v]
            transl_dict[v] = u
        for i in cgraph.node_dict:
            if i not in transl_dict:
                transl_dict[i] = i

        # replace all references to v by u
        tmp = {}
        for i in cgraph.node_dict:
            children, op = cgraph.node_dict[i]
            new_i = transl_dict[i]
            while new_i not in cgraph.node_dict:
                new_i = transl_dict[new_i]
            new_children = []
            for c in children:
                new_c = transl_dict[c]
                while new_c not in cgraph.node_dict:
                    new_c = transl_dict[new_c]
                new_children.append(new_c)

            tmp[new_i] = (new_children, op)
        # update indices
        transl_dict = {i : j for j, i in enumerate(sorted(tmp.keys()))}
        new_node_dict = {}
        for i in tmp:
            children, op = tmp[i]
            new_children = [transl_dict[c] for c in children]
            new_node_dict[transl_dict[i]] = (new_children, op)
        cgraph.node_dict = new_node_dict
        cgraph.update_stats()


    # Special case: Remove unneccessary identity nodes at output
    H = {}
    V = set()
    for v in cgraph.node_dict:
        children, op = cgraph.node_dict[v]
        if (op == '='):
            u = children[0] # should be the only child
            if (u not in cgraph.outp_nodes) and (v in cgraph.outp_nodes) and (u not in cgraph.inp_nodes) and (u not in H):
                # delete v, replace all references to u with v
                H[u] = v
                V.add(v)
    tmp = {}
    for i in cgraph.node_dict:
        if i not in V:
            children, op = cgraph.node_dict[i]
            new_children = []
            for c in children:
                if c in H:
                    new_children.append(H[c])
                elif c not in V:
                    new_children.append(c)
            if i in H:
                tmp[H[i]] = new_children, op
            else:
                tmp[i] = new_children, op
    # re-enumerate
    transl_dict = {old_idx : new_idx for new_idx, old_idx in enumerate(sorted(tmp.keys()))}
    new_node_dict = {}
    for i in tmp:
        children, op = tmp[i]

        new_children = []
        for c in children:
            new_children.append(transl_dict[c])
        new_node_dict[transl_dict[i]] = (new_children, op)


    cgraph.node_dict = new_node_dict
    cgraph.update_stats()


    # 2. remove deadends
    rem_mask = (np.sum((cgraph.R[cgraph.outp_nodes,:] > 0), axis=0) == 0)
    rem_mask[cgraph.inp_nodes] = False

    rem_idx = np.arange(0, len(cgraph.node_dict), 1)[rem_mask]

    tmp = {i : cgraph.node_dict[i] for i in cgraph.node_dict}
    for idx in rem_idx:
        del tmp[idx]
    transl_dict = {old_idx : new_idx for new_idx, old_idx in enumerate(sorted(tmp.keys()))}
    new_node_dict = {}
    for i in tmp:
        children, op = cgraph.node_dict[i]

        new_children = []
        for c in children:
            new_children.append(transl_dict[c])
        new_node_dict[transl_dict[i]] = (new_children, op)
    cgraph.node_dict = new_node_dict
    cgraph.update_stats()
    return cgraph

def plot_cgraph(cgraph:CompGraph, ax=plt.axes, radius = 0.1, buffer_factor = 5.0, curvature = 0.3) -> nx.MultiDiGraph:
    '''
    Tries to plot the given computational graph.
    '''


    # get layers of nodes
    eval_order = cgraph.get_eval_order()

    # get positions for nodes

    y_max = max([len(order) for order in eval_order])
    y_min = 0

    node_positions = {}
    for x, node_list in enumerate(eval_order):
        y_pos = np.flip(np.linspace(y_min, y_max, len(node_list) + 2)[1:-1])
        for n, y in zip(node_list, y_pos):
            node_positions[n] = (x, y)


    # create Graph

    G = nx.MultiDiGraph()

    # add nodes
    color_dict = {}
    label_dict = {}

    inp_count = 0
    for i in cgraph.node_dict:
        children, op = cgraph.node_dict[i]
        G.add_node(i)

        # color
        if i in cgraph.inp_nodes:
            color_dict[i] = 'cyan'
        elif i in cgraph.outp_nodes:
            color_dict[i] = 'red'
        else:
            color_dict[i] = 'white'


        # label
        if op == 'inp':
            op = f'x{inp_count}'
            inp_count += 1
        label_dict[i] = op


    # add edges
    for i in cgraph.node_dict:
        children, op = cgraph.node_dict[i]
        for j in children:
            G.add_edge(j,i)

    color_list = [color_dict[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, node_positions, node_color=color_list, edgecolors='black', node_size = (2*radius)*10**4, ax = ax)
    nx.draw_networkx_labels(G, node_positions, labels=label_dict, ax = ax)

    for e in G.edges:


        x0, y0 = node_positions[e[0]]
        x1, y1 = node_positions[e[1]]

        sign = (e[2]%2)*2-1
        ampl = (((e[2]+1)//2))*curvature
        ax.annotate("",
                    xy=(x0, y0), xycoords='data',
                    xytext=(x1, y1), textcoords='data',
                    arrowprops=dict(arrowstyle="<|-", color="0.0",
                                    shrinkA=100*radius, shrinkB=100*radius,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr',str(sign*ampl)
                                    ),
                                    ),
                    )



    buffer = buffer_factor*radius
    xs = []
    ys = []
    for n in node_positions:
        (x,y) = node_positions[n]
        xs.append(x)
        ys.append(y)

    ymin, ymax = np.min(ys), np.max(ys)
    xmin, xmax = np.min(xs), np.max(xs)
    ax.set_ylim(ymin-buffer, ymax+buffer)
    ax.set_xlim(xmin-buffer, xmax+buffer)

    return G
