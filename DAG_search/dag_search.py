'''
Operations for combining computational graphs
'''
import numpy as np
import itertools
import warnings
from scipy.special import logsumexp
from tqdm import tqdm
import pickle
import multiprocessing
import time
import sklearn
import sys
sys.path.append('.')
from DAG_search import config
from DAG_search import comp_graph
from torch_utils.correlation import *
from torch_utils.pytorch_utils import *
from expr_cal.stock_data import Stock_Data
from DAG_search import utils
warnings.simplefilter("ignore")
########################
# Loss Functions
########################

class DAG_Loss_fkt(object):
    '''
    Abstract class for Loss function
    '''
    def __init__(self):
        pass

    def __call__(self, cgraph:comp_graph.CompGraph) -> float:
        '''
        Lossfkt(graph)

        @Params:
            cgraph... computational Graph

        @Returns:
            Loss value for the graph
        '''
        pass

## Symbolic Regression
class IC_loss_fkt(DAG_Loss_fkt):
    def __init__(self, outp:np.ndarray, data: Stock_Data):
        '''
        Loss function for finding DAG for regression task.

        @Params:
            outp... output that DAG should match (N x n)
        '''
        super().__init__()
        self.outp = outp
        self.data = data

    def __call__(self, cgraph:comp_graph.CompGraph) -> float:
        '''
        Lossfkt(graph)

        @Returns:
            1 - |IC| of graph output against desired output
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pred = cgraph.evaluate(data=self.data)
            try:
                ic = (np.dot(self.outp, pred) / self.data.n_days).item()
            except Exception:
                ic = 0
            loss = 1 - abs(ic)

        return loss


########################
# DAG creation
########################

def get_pre_order(order:list, node:int, inp_nodes:list, inter_nodes:list, outp_nodes:list) -> tuple:
    '''
    Given a DAG creation order, returns the pre order of the subtree with a given node as root.
    '''
    if node in outp_nodes:
        idx = 2*len(inter_nodes) + 2*(node-len(inp_nodes))
    else:
        idx = 2*(node - len(inp_nodes) - len(outp_nodes))
    idx_l = idx
    idx_r = idx + 1

    v_l = order[idx_l]
    v_r = order[idx_r]

    if (v_l in inp_nodes):
        return (node, v_l, v_r)
    elif (v_r in inp_nodes) or (v_r < 0):
        return (node,) + get_pre_order(order, v_l, inp_nodes, inter_nodes, outp_nodes) + (v_r,)
    else:
        return (node, ) +get_pre_order(order, v_l, inp_nodes, inter_nodes, outp_nodes) + get_pre_order(order, v_r, inp_nodes, inter_nodes, outp_nodes)

def build_dag(build_order:list, node_ops:list, m:int, n:int) -> comp_graph.CompGraph:
    '''
    Given a build order, builds a computational DAG.

    @Params:
        build_order... list of tuples (node, parent_nodes)
        node_ops... list of operations for each node
        m... number of input nodes
        n... number of output nodes

    @Returns:
        Computational DAG
    '''

    node_dict = {}
    for i in range(m):
        node_dict[i] = ([], 'inp')

    for op, (i, parents) in zip(node_ops, build_order):
        node_dict[i] = (list(parents), op)
    return comp_graph.CompGraph(m = m, n = n, node_dict = node_dict)

def adapt_ops(cgraph:comp_graph.CompGraph, build_order:list, node_ops:list) -> comp_graph.CompGraph:
    '''
    Given a computational Graph, changes the operations at nodes (no need to reinstantiate)
    '''
    node_dict = cgraph.node_dict
    for op, (i, parents) in zip(node_ops, build_order):
        node_dict[i] = (list(parents), op)
    cgraph.node_dict = node_dict
    return cgraph

def get_build_orders(m:int, n:int, n_calc_nodes:int, max_orders:int = 10000, verbose:int = 0, fix_size : bool = False, filter_func : callable = None, **params) -> list:
    '''
    Creates empty DAG scaffolds (no operations yet).

    @Params:
        m... number of input nodes
        n... number of output nodes
        n_calc_nodes... number of intermediate nodes
        max_orders... maximum number of possible DAG orders to search trough (lower = exhaustive, higher = sampling)
        verbose... 0 - no print, 1 - status message, 2 - progress bar
        fix_size... if set, will return only build orders with n_calc_nodes intermediate nodes (not less)
        filter_func... if set, will only return build orders for which filter_func(build_order) is True
    @Returns:
        list build orders (can be used by build_dag).
        build order = list of tuples (node, parent_nodes)
    '''
    if filter_func is None:
        filter_func = lambda orders : orders

    if verbose > 0:
        print('Creating evaluation orders')
    l = n_calc_nodes
    inp_nodes = [i for i in range(m)]
    outp_nodes = [i + m for i in range(n)]
    inter_nodes = [i + m + n for i in range(l)]

    # collect possible predecessors
    predecs = {}
    for i in inp_nodes:
        predecs[i] = []

    for i in outp_nodes:
        predecs[i] = inp_nodes + inter_nodes

    for i in inter_nodes:
        predecs[i] = inp_nodes + [j for j in inter_nodes if j < i]

    # create sample space for edges
    sample_space_edges = []
    for i in inter_nodes + outp_nodes:
        sample_space_edges.append(predecs[i])
        sample_space_edges.append(predecs[i] + [-1])
    log_its_total = np.sum([np.log(len(s)) for s in sample_space_edges])

    if fix_size:
        if n_calc_nodes > 0:
            lengths_prev = []
            for i in range(2*(len(inter_nodes) - 1)):
                lengths_prev.append(len(sample_space_edges[i]))
            for i in range(2*(len(outp_nodes))):
                lengths_prev.append(len(sample_space_edges[i + 2*len(inter_nodes)]) - 1)
            log_its_prev = np.sum([np.log(l) for l in lengths_prev])
            log_its_total = logsumexp([log_its_total, log_its_prev], b = [1, -1])

    if log_its_total > np.log(max_orders):
        # just sample random orders
        possible_edges = []
        for tmp in sample_space_edges:
            possible_edges.append(np.random.choice(tmp, size = max_orders))
        possible_edges = np.column_stack(possible_edges)

    else:
        possible_edges = itertools.product(*sample_space_edges)
    valid_set = set()
    build_orders = []

    if verbose == 2:
        total_its = np.prod([len(s) for s in sample_space_edges])
        pbar = tqdm(possible_edges, total = total_its)
    else:
        pbar = possible_edges

    for order in pbar:
        # order -> ID
        valid = True
        for i in range(l + n):
            if order[2*i] < order[2*i + 1]:
                valid = False
                break
        if valid:
            order_ID = ()
            # pre orders of outputs
            for i in range(n):
                order_ID = order_ID + get_pre_order(order, m + i, inp_nodes, inter_nodes, outp_nodes)

            is_new = True
            if fix_size:
                if n_calc_nodes > 0:
                    # how many intermediate nodes occur?
                    check_set = set(order_ID)
                    is_new = np.all([i in check_set for i in inter_nodes])

            if is_new:
                # rename intermediate nodes after the order in which they appear ( = 1 naming)
                ren_dict = {}
                counter = 0
                for i in order:
                    if (i in inter_nodes) and (i not in ren_dict):
                        ren_dict[i] = inter_nodes[counter]
                        counter += 1
                tmp_ID = tuple([ren_dict[node] if node in ren_dict else node for node in order_ID])
                is_new = tmp_ID not in valid_set



            if is_new:
                valid_set.add(tmp_ID)

                # build (node, parents) order for intermediate + output nodes
                tmp = sorted([i for i in set(order_ID) if i in inter_nodes])
                ren_dict = {node : inter_nodes[i] for i, node in enumerate(tmp)}
                build_order = []
                for i in sorted(tmp + outp_nodes):
                    if i in ren_dict:
                        # intermediate node
                        i1 = 2*(i - (m + n))
                        i2 = 2*(i - (m + n)) + 1
                    elif i in outp_nodes:
                        # output node
                        i1 = 2*l + 2*(i - m)
                        i2 = 2*l + 2*(i - m) + 1
                    p1 = order[i1]
                    p2 = order[i2]
                    preds = []
                    if p1 in ren_dict:
                        preds.append(ren_dict[p1])
                    else:
                        preds.append(p1)
                    if p2 in ren_dict:
                        preds.append(ren_dict[p2])
                    elif p2 >= 0:
                        preds.append(p2)

                    if i in ren_dict:
                        build_order.append((ren_dict[i], tuple(preds)))
                    else:
                        build_order.append((i, tuple(preds)))
                build_orders.append(tuple(build_order))

    return filter_func(build_orders)


def get_build_orders_prior(m:int, n:int, n_calc_nodes:int, max_orders:int = 10000, verbose:int = 0, prob_dict:dict=None, **params) -> list:
    if prob_dict is None:
        # node type, child types
        prob_dict = {'unary': 0.3292803970223325, 'binary': 0.6707196029776675}
    else:
        assert 'unary' in prob_dict and 'binary' in prob_dict
    probs = np.array([abs(prob_dict['unary']), abs(prob_dict['binary'])])
    probs = probs/probs.sum()

    l = n_calc_nodes
    inp_nodes = [i for i in range(m)]
    outp_nodes = [i + m for i in range(n)]
    inter_nodes = [i + m + n for i in range(l)]
    # collect possible predecessors
    predecs = {}
    for i in inp_nodes:
        predecs[i] = []

    for i in outp_nodes:
        predecs[i] = inp_nodes + inter_nodes

    for i in inter_nodes:
        predecs[i] = inp_nodes + [j for j in inter_nodes if j < i]

    valid_set = set()
    build_orders = []

    # start sampling
    if verbose == 2:
        pbar = tqdm(range(max_orders))
    else:
        pbar = range(max_orders)
    for _ in pbar:
        # sample an order according to probabilities
        order = ()
        for idx in inter_nodes + outp_nodes:
            if np.random.choice([0, 1], p=probs) == 0:
                # unary
                c1 = np.random.choice(predecs[idx])
                opt = (c1, -1)
            else:
                # binary
                c1 = np.random.choice(predecs[idx])
                c2 = np.random.choice(predecs[idx])
                opt = (max(c1, c2), min(c1, c2))
            order += opt

        # order -> ID
        order_ID = ()
        # pre orders of outputs
        for i in range(n):
            order_ID = order_ID + get_pre_order(order, m + i, inp_nodes, inter_nodes, outp_nodes)
        is_new = True
        if is_new:
            # rename intermediate nodes after the order in which they appear ( = 1 naming)
            ren_dict = {}
            counter = 0
            for i in order:
                if (i in inter_nodes) and (i not in ren_dict):
                    ren_dict[i] = inter_nodes[counter]
                    counter += 1
            tmp_ID = tuple([ren_dict[node] if node in ren_dict else node for node in order_ID])
            is_new = tmp_ID not in valid_set



        if is_new:
            valid_set.add(tmp_ID)

            # build (node, parents) order for intermediate + output nodes
            tmp = sorted([i for i in set(order_ID) if i in inter_nodes])
            ren_dict = {node : inter_nodes[i] for i, node in enumerate(tmp)}
            build_order = []
            for i in sorted(tmp + outp_nodes):
                if i in ren_dict:
                    # intermediate node
                    i1 = 2*(i - (m + n))
                    i2 = 2*(i - (m + n)) + 1
                elif i in outp_nodes:
                    # output node
                    i1 = 2*l + 2*(i - m)
                    i2 = 2*l + 2*(i - m) + 1
                p1 = order[i1]
                p2 = order[i2]
                preds = []
                if p1 in ren_dict:
                    preds.append(ren_dict[p1])
                else:
                    preds.append(p1)
                if p2 in ren_dict:
                    preds.append(ren_dict[p2])
                elif p2 >= 0:
                    preds.append(p2)

                if i in ren_dict:
                    build_order.append((ren_dict[i], tuple(preds)))
                else:
                    build_order.append((i, tuple(preds)))
            build_orders.append(tuple(build_order))

    return build_orders

def evaluate_cgraph(cgraph:comp_graph.CompGraph, loss_fkt:callable, loss_thresh:float = None) -> float:
    '''
    Evaluates the loss for a computational DAG.

    @Returns:
        loss value
    '''
    if loss_thresh is not None:
        # we are in parallel mode
        global stop_var
        evaluate = not bool(stop_var)
        if not evaluate:
            return np.inf

    loss = loss_fkt(cgraph)
    if loss_thresh is not None:
        if loss <= loss_thresh:
            stop_var = True

    return loss

def evaluate_build_order(order:list, m:int, n:int, loss_fkt:callable, topk:int = 5, loss_thresh:float = None, start_time:float = None, max_time:float = 3600, expect_evals:int = None, pareto:bool = False) -> tuple:
    '''
    Given a build order (output of get_build_orders), tests all possible assignments of operators.

    @Returns:
        tuple:
            losses... list of losses for DAGs
            ops... list of ops that were tried
    '''
    evaluate = True
    if loss_thresh is not None:
        # we are in parallel mode
        global stop_var
        evaluate = not bool(stop_var)

    ret_losses = []
    ret_ops = []

    max_loss = np.inf
    max_idx = None

    if start_time is not None:
        evaluate = evaluate and (time.time() - start_time) < max_time

    if evaluate:
        bin_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 2]
        un_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 1]

        outp_nodes = [m + i for i in range(n)]
        op_spaces = []

        transl_dict = {}
        for i, (node, parents) in enumerate(order):
            if len(parents) == 2:
                op_spaces.append(bin_ops)
            else:
                if node in outp_nodes:
                    op_spaces.append(un_ops)
                else:
                    op_spaces.append([op for op in un_ops if op != '='])
            transl_dict[node] = i

        if expect_evals is not None:
            v = np.log(expect_evals)
            log_space_size = 0
            for s in op_spaces:
                log_space_size += np.log(len(s))
            log_accept_ratio = min(0, v - log_space_size)
            accept_ratio = np.exp(log_accept_ratio)
        else:
            accept_ratio = 1.0


        cgraph = None

        # keeping track of invalids
        inv_array = []
        inv_mask = []
        for ops in itertools.product(*op_spaces):

            if (start_time is not None) and ((time.time() - start_time) >= max_time):
                break
            accept = True
            if accept_ratio < 1.0:
                accept = (np.random.rand() < accept_ratio)

            if len(inv_array) > 0:
                num_ops = np.array([config.NODE_ID[op] for op in ops])
                is_inv = np.sum((abs(inv_array - num_ops))*inv_mask, axis = 1)
                is_inv = np.any(is_inv == 0)
            else:
                is_inv = False

            if not is_inv and accept:
                if cgraph is None:
                    cgraph = build_dag(order, ops, m, n)
                else:
                    cgraph = adapt_ops(cgraph, order, ops)

                loss = evaluate_cgraph(cgraph, loss_fkt, loss_thresh)

                if (len(ret_losses) == 0):
                    ret_losses.append(loss)
                    ret_ops.append(ops)
                    max_idx = np.argmax(ret_losses)
                    max_loss = ret_losses[max_idx]

                elif pareto:
                    # check which entries it dominates
                    dominated_entries = []
                    for i, (pareto_loss, pareto_ops) in enumerate(zip(ret_losses, ret_ops)):
                        if pareto_loss >= loss and len(pareto_ops) >= len(ops):
                            dominated_entries.append(i)
                    # keep only non dominated entries
                    if len(dominated_entries) > 0:
                        ret_losses = [loss] + [ret_losses[i] for i in range(len(ret_losses)) if i not in dominated_entries]
                        ret_ops = [ops] + [ret_ops[i] for i in range(len(ret_ops)) if i not in dominated_entries]

                elif (len(ret_losses) < topk):
                    # append
                    ret_losses.append(loss)
                    ret_ops.append(ops)

                    max_idx = np.argmax(ret_losses)
                    max_loss = ret_losses[max_idx]
                else:
                    if loss < max_loss:
                        # replace
                        ret_losses[max_idx] = loss
                        ret_ops[max_idx] = ops

                        max_idx = np.argmax(ret_losses)
                        max_loss = ret_losses[max_idx]
            else:
                pass
    return ret_losses, ret_ops

def sample_graph(m:int, n:int, n_calc_nodes:int, only_order:bool = False) -> comp_graph.CompGraph:
    '''
    Samples a computational DAG.
    '''


    # 1. Sample build order
    l = n_calc_nodes
    inp_nodes = [i for i in range(m)]
    outp_nodes = [i + m for i in range(n)]
    inter_nodes = [i + m + n for i in range(l)]

    bin_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 2]
    un_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 1]

    predecs = {}
    for i in inp_nodes:
        predecs[i] = []

    for i in outp_nodes:
        predecs[i] = inp_nodes + inter_nodes

    for i in inter_nodes:
        predecs[i] = inp_nodes + [j for j in inter_nodes if j < i]

    # sample order from predecessors
    order = []
    for i in inter_nodes + outp_nodes:
        l1 = predecs[i]
        order.append(np.random.choice(l1))

        l2 = [j for j in (predecs[i] + [-1]) if j < order[-1]]
        order.append(np.random.choice(l2))

    # order -> ID
    dep_entries = [l*2 + 2*i for i in range(n)] + [l*2 + 2*i + 1 for i in range(n)] # order indicies that are dependend
    tmp = set([order[i] for i in dep_entries if order[i] not in inp_nodes and order[i] >= 0]) # node indices that remain
    while len(tmp) > 0:
        tmp_deps = []
        for idx in tmp:
            tmp_deps.append(2*(idx - (m + n)))
            tmp_deps.append(2*(idx - (m + n)) + 1)
        tmp_deps = list(set(tmp_deps))
        dep_entries = dep_entries + tmp_deps
        tmp = set([order[i] for i in tmp_deps if order[i] not in inp_nodes and order[i] >= 0])
    dep_entries = sorted(dep_entries)

    ren_dict = {}
    order_ID = []
    counter = m
    for i in range(m, m + n + l):
        if i in outp_nodes:
            i1 = 2*l + 2*(i - m)
            i2 = 2*l + 2*(i - m) + 1
        else:
            i1 = 2*(i - (m + n))
            i2 = 2*(i - (m + n)) + 1
        preds = []
        if i1 in dep_entries:
            preds.append(order[i1])
        if i2 in dep_entries and order[i2] >= 0:
            preds.append(order[i2])

        if len(preds) > 0:
            ren_dict[i] = counter
            order_ID.append((i, tuple(preds)))
            counter += 1
    new_order_ID = []
    for i, preds in order_ID:
        new_preds = [j if j not in ren_dict else ren_dict[j] for j in preds]
        new_order_ID.append((ren_dict[i], tuple(new_preds)))
    order = tuple(new_order_ID)

    if only_order:
        return order

    # 2. sample operations on build order
    node_ops = []
    for _, parents in order:
        if len(parents) == 2:
            node_ops.append(np.random.choice(bin_ops))
        else:
            node_ops.append(np.random.choice(un_ops))

    # 3. create cgraph
    return build_dag(order, node_ops, m, n)

def sample_graph_prior(m:int, n:int, n_calc_nodes:int, only_order:bool = False, prob_dict:dict = None) -> comp_graph.CompGraph:
    '''
    Samples a computational DAG.
    '''

    order = get_build_orders_prior(m = m, n = n, n_calc_nodes = n_calc_nodes, max_orders = 1, prob_dict=prob_dict)[0]

    if only_order:
        return order

    bin_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 2]
    un_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 1]

    # 2. sample operations on build order
    node_ops = []
    for _, parents in order:
        if len(parents) == 2:
            node_ops.append(np.random.choice(bin_ops))
        else:
            node_ops.append(np.random.choice(un_ops))

    # 3. create cgraph
    return build_dag(order, node_ops, m, n)


########################
# Search Methods
########################

def init_process(early_stop):
    global stop_var
    stop_var = early_stop

def is_pickleable(x:object) -> bool:
    '''
    Used for multiprocessing. Loss function must be pickleable.
    '''

    try:
        pickle.dumps(x)
        return True
    except (pickle.PicklingError, AttributeError):
        return False

def exhaustive_search(n_inputs: int, n_outps: int, loss_fkt: callable, n_calc_nodes:int = 1, n_processes:int = 1, topk:int = 5, verbose:int = 0, max_orders:int = 10000, max_time:float = 900.0, stop_thresh:float = -1.0, unique_loss:bool = True, expect_evals:int = None, pareto:bool = False, order_gen:callable = get_build_orders, **params) -> dict:
    '''
    Exhaustive search for a DAG.

    @Returns:
        dictionary with:
            graphs -> list of computational DAGs
            losses -> list of losses
            total_evals -> number of evaluated graphs
    '''
    ret = {
        'graphs' : [],
        'losses' : []}
    process_start_time = time.time()

    n_processes = max(min(n_processes, multiprocessing.cpu_count()), 1)
    ctx = multiprocessing.get_context('spawn')

    if n_processes > 1:
        error_msg = 'Loss function must be serializable with pickle for > 1 processes.\n'
        error_msg += 'See dag_search.MSE_loss_fkt for an example.\n'
        error_msg += 'If this worked before, consider reloading your loss funktion.'
        assert is_pickleable(loss_fkt), error_msg

    m = n_inputs
    n = n_outps

    orders = order_gen(m, n, n_calc_nodes, max_orders = max_orders, verbose=verbose)

    if verbose > 0:
        print(f'Total orders: {len(orders)}')
        print('Evaluating orders')

    total_evals = 0 # number of evaluated graphs
    top_losses = []
    top_ops = []
    top_orders = []
    loss_thresh = np.inf

    early_stop = False
    if n_processes == 1:
        # sequential
        if verbose == 2:
            pbar = tqdm(orders)
        else:
            pbar = orders
        for order in pbar:
            losses, ops = evaluate_build_order(order, m, n, loss_fkt, topk, start_time=process_start_time, max_time=max_time, pareto=pareto, expect_evals=expect_evals)
            total_evals += len(losses)
            if pareto:
                top_losses += losses
                top_ops += ops
                top_orders += [order]*len(losses)

                complexities = [len(op) for op in top_ops]
                pareto_idxs = utils.get_pareto_idxs(top_losses, complexities)

                if unique_loss:
                    pareto_losses = set()
                    take_idxs = []
                    for j in pareto_idxs:
                        v = np.round(top_losses[j], 15)
                        if v not in pareto_losses:
                            pareto_losses.add(v)
                            take_idxs.append(j)
                else:
                    take_idxs = pareto_idxs

                top_losses = [top_losses[i] for i in take_idxs]
                top_ops = [top_ops[i] for i in take_idxs]
                top_orders = [top_orders[i] for i in take_idxs]

                if verbose == 2:
                    pbar.set_postfix({'best_loss' : np.min(top_losses), 'n_pareto' : len(top_losses)})

                early_stop = np.any(np.array(top_losses) < stop_thresh)
            else:
                for loss, op in zip(losses, ops):

                    if loss <= loss_thresh:
                        if unique_loss:
                            valid = loss not in top_losses
                        else:
                            valid = True

                        if valid:
                            if len(top_losses) >= topk:
                                repl_idx = np.argmax(top_losses)
                                top_losses[repl_idx] = loss
                                top_ops[repl_idx] = op
                                top_orders[repl_idx] = order
                            else:
                                top_losses.append(loss)
                                top_ops.append(op)
                                top_orders.append(order)

                            loss_thresh = np.max(top_losses)
                            if verbose == 2:
                                pbar.set_postfix({'best_loss' : np.min(top_losses)})
                    if loss < stop_thresh:
                        early_stop = True
                        break
            if early_stop:
                break
    else:

        args = [[order, m, n, loss_fkt, topk, stop_thresh, process_start_time, max_time, expect_evals, pareto] for order in orders]
        if verbose == 2:
            pbar = tqdm(args, total = len(args))
        else:
            pbar = args

        with ctx.Pool(processes=n_processes, initializer=init_process, initargs=(early_stop,)) as pool:
            pool_results = pool.starmap(evaluate_build_order, pbar)

        if verbose > 0:
            print('Collecting results')
        for i, (losses, ops) in enumerate(pool_results):
            total_evals += len(losses)
            if early_stop:
                break
            if pareto:
                top_losses += losses
                top_ops += ops
                top_orders += [orders[i]]*len(losses)

                complexities = [len(op) for op in top_ops]
                pareto_idxs = utils.get_pareto_idxs(top_losses, complexities)

                if unique_loss:
                    pareto_losses = set()
                    take_idxs = []
                    for j in pareto_idxs:
                        v = np.round(top_losses[j], 15)
                        if v not in pareto_losses:
                            pareto_losses.add(v)
                            take_idxs.append(j)
                else:
                    take_idxs = pareto_idxs

                top_losses = [top_losses[j] for j in take_idxs]
                top_ops = [top_ops[j] for j in take_idxs]
                top_orders = [top_orders[j] for j in take_idxs]

                if verbose == 2:
                    pbar.set_postfix({'best_loss' : np.min(top_losses), 'n_pareto' : len(top_losses)})

                early_stop = np.any(np.array(top_losses) < stop_thresh)
            else:
                for loss, op in zip(losses, ops):
                    if loss <= loss_thresh:
                        if unique_loss:
                            valid = loss not in top_losses
                        else:
                            valid = True

                        if valid:
                            if len(top_losses) >= topk:
                                repl_idx = np.argmax(top_losses)
                                top_losses[repl_idx] = loss
                                top_ops[repl_idx] = op
                                top_orders[repl_idx] = orders[i]
                            else:
                                top_losses.append(loss)
                                top_ops.append(op)
                                top_orders.append(orders[i])

                            loss_thresh = np.max(top_losses)
                            if verbose == 2:
                                pbar.set_postfix({'best_loss' : np.min(top_losses)})
                    if loss < stop_thresh:
                        early_stop = True
                        break

    sort_idx = np.argsort(top_losses)
    top_losses = [top_losses[i] for i in sort_idx]
    top_orders = [top_orders[i] for i in sort_idx]
    top_ops = [top_ops[i] for i in sort_idx]
    top_graphs = []
    for order, ops in zip(top_orders, top_ops):
        cgraph = build_dag(order, ops, m, n)
        top_graphs.append(cgraph.copy())

    ret['graphs'] = top_graphs
    ret['losses'] = top_losses
    ret['total_evals'] = total_evals

    return ret

def hierarchical_search(n_inputs: int, n_outps: int, loss_fkt: callable, n_calc_nodes:int = 1, n_processes:int = 1, topk:int = 5, verbose:int = 0, max_orders:int = 10000, max_time:float = 900.0, stop_thresh:float = -1.0, hierarchy_stop_thresh:float = -1.0, unique_loss:bool = True, pareto:bool = False, **params) -> dict:
    '''
    Exhaustive search for a DAG but hierarchical.
    '''
    ret = {
        'graphs' : [],
        'losses' : []}
    process_start_time = time.time()


    n_processes = max(min(n_processes, multiprocessing.cpu_count()), 1)
    ctx = multiprocessing.get_context('spawn')

    if n_processes > 1:
        error_msg = 'Loss function must be serializable with pickle for > 1 processes.\n'
        error_msg += 'See dag_search.MSE_loss_fkt for an example.\n'
        error_msg += 'If this worked before, consider reloading your loss funktion.'
        assert is_pickleable(loss_fkt), error_msg

    m = n_inputs
    n = n_outps

    total_evals = 0 # number of evaluated graphs
    top_losses = []
    top_ops = []
    top_orders = []
    loss_thresh = np.inf

    for calc_nodes in range(n_calc_nodes + 1): # 0, 1, ..., n_calc_nodes
        setup_time = time.time() - process_start_time
        if setup_time >= max_time:
            break

        if verbose > 0:
            print('#########################')
            print(f'# Calc Nodes: {calc_nodes}')
            print('#########################')

        # collect computational graphs (no operations on nodes yet)
        orders = get_build_orders(m, n, calc_nodes, max_orders = max_orders, verbose=verbose, fix_size=True)

        if verbose > 0:
            print(f'Total orders: {len(orders)}')
            print('Evaluating orders')

        early_stop = False
        if n_processes == 1:
            # sequential
            if verbose == 2:
                pbar = tqdm(orders)
            else:
                pbar = orders
            for order in pbar:
                if early_stop:
                    break

                losses, ops = evaluate_build_order(order, m, n, loss_fkt, topk, start_time=process_start_time, max_time=max_time, pareto=pareto)
                total_evals += len(losses)
                if pareto:
                    top_losses += losses
                    top_ops += ops
                    top_orders += [order]*len(losses)

                    complexities = [len(op) for op in top_ops]
                    pareto_idxs = utils.get_pareto_idxs(top_losses, complexities)

                    if unique_loss:
                        pareto_losses = set()
                        take_idxs = []
                        for j in pareto_idxs:
                            v = np.round(top_losses[j], 15)
                            if v not in pareto_losses:
                                pareto_losses.add(v)
                                take_idxs.append(j)
                    else:
                        take_idxs = pareto_idxs

                    top_losses = [top_losses[i] for i in take_idxs]
                    top_ops = [top_ops[i] for i in take_idxs]
                    top_orders = [top_orders[i] for i in take_idxs]

                    if verbose == 2:
                        pbar.set_postfix({'best_loss' : np.min(top_losses), 'n_pareto' : len(top_losses)})


                else:
                    for loss, op in zip(losses, ops):
                        if loss <= loss_thresh:
                            if unique_loss:
                                valid = loss not in top_losses
                            else:
                                valid = True

                            if valid:
                                if len(top_losses) >= topk:
                                    repl_idx = np.argmax(top_losses)
                                    top_losses[repl_idx] = loss
                                    top_ops[repl_idx] = op
                                    top_orders[repl_idx] = order
                                else:
                                    top_losses.append(loss)
                                    top_ops.append(op)
                                    top_orders.append(order)

                                loss_thresh = np.max(top_losses)
                                if verbose == 2:
                                    pbar.set_postfix({'best_loss' : np.min(top_losses)})
                early_stop = np.any(np.array(top_losses) < stop_thresh)


        else:
            args = [[order, m, n, loss_fkt, topk, stop_thresh, process_start_time, max_time- setup_time, None, pareto] for order in orders]
            if verbose == 2:
                pbar = tqdm(args, total = len(args))
            else:
                pbar = args

            with ctx.Pool(processes=n_processes, initializer=init_process, initargs=(early_stop,)) as pool:
                pool_results = pool.starmap(evaluate_build_order, pbar)


            for i, (losses, ops) in enumerate(pool_results):
                total_evals += len(losses)
                if early_stop:
                    break

                if pareto:
                    top_losses += losses
                    top_ops += ops
                    top_orders += [orders[i]]*len(losses)

                    complexities = [len(op) for op in top_ops]
                    pareto_idxs = utils.get_pareto_idxs(top_losses, complexities)

                    if unique_loss:
                        pareto_losses = set()
                        take_idxs = []
                        for j in pareto_idxs:
                            v = np.round(top_losses[j], 15)
                            if v not in pareto_losses:
                                pareto_losses.add(v)
                                take_idxs.append(j)
                    else:
                        take_idxs = pareto_idxs

                    top_losses = [top_losses[j] for j in take_idxs]
                    top_ops = [top_ops[j] for j in take_idxs]
                    top_orders = [top_orders[j] for j in take_idxs]

                    if verbose == 2:
                        pbar.set_postfix({'best_loss' : np.min(top_losses), 'n_pareto' : len(top_losses)})


                else:
                    for loss, op in zip(losses, ops):
                        if loss <= loss_thresh:
                            if unique_loss:
                                valid = loss not in top_losses
                            else:
                                valid = True

                            if valid:
                                if len(top_losses) >= topk:
                                    repl_idx = np.argmax(top_losses)
                                    top_losses[repl_idx] = loss
                                    top_ops[repl_idx] = op
                                    top_orders[repl_idx] = orders[i]
                                else:
                                    top_losses.append(loss)
                                    top_ops.append(op)
                                    top_orders.append(orders[i])

                                loss_thresh = np.max(top_losses)
                                if verbose == 2:
                                    pbar.set_postfix({'best_loss' : np.min(top_losses)})
                early_stop = np.any(np.array(top_losses) < stop_thresh)

        sort_idx = np.argsort(top_losses)
        top_losses = [top_losses[i] for i in sort_idx]
        top_orders = [top_orders[i] for i in sort_idx]
        top_ops = [top_ops[i] for i in sort_idx]
        top_graphs = []
        for order, ops in zip(top_orders, top_ops):
            cgraph = build_dag(order, ops, m, n)
            top_graphs.append(cgraph.copy())

        if top_losses[0] <= stop_thresh or np.any(np.array(top_losses) <= hierarchy_stop_thresh):
            if verbose > 0:
                print(f'Stopping because early stop criteria has been matched!')
            break
    ret['graphs'] = top_graphs
    ret['losses'] = top_losses
    ret['total_evals'] = total_evals

    return ret



########################
# Sklearn Interface for Symbolic Regression Task
########################

class DAGRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Symbolic DAG-Search

    Sklearn interface for exhaustive search.
    '''

    def __init__(self, n_calc_nodes:int = 5, max_orders:int = int(1e6), random_state:int = None, processes:int = 1, max_samples:int = None, stop_thresh:float = 1e-20, mode : str = 'exhaustive',
                  loss_fkt :DAG_Loss_fkt = IC_loss_fkt, max_time:float = 1800.0, positives:list = None, pareto:bool = False, data = None, topk:int = 10, **kwargs):
        '''
        @Params:
            n_calc_nodes... number of possible intermediate nodes
            max_orders... maximum number of expression - skeletons in search
            random_state... for reproducibility
            processes... number of processes for multiprocessing
            max_samples... maximum number of samples on which to fit
            stop_thresh... threshold for early stopping, set to negative value if you dont want it
            mode... one of 'exhaustive' or 'hierarchical'
            loss_fkt... loss function class
            positives... marks which X are strictly positive
            pareto... flag for exhaustive search
        '''
        self.total_evals = 0
        self.n_calc_nodes = n_calc_nodes
        self.max_orders = max_orders
        self.max_samples = max_samples
        assert mode in ['exhaustive', 'hierarchical'], f'Search mode {mode} is not supported.'
        self.mode = mode
        self.stop_thresh = stop_thresh
        self.max_time = max_time
        self.processes = max(min(processes, multiprocessing.cpu_count()), 1)
        self.topk = topk
        self.random_state = random_state
        self.loss_fkt = loss_fkt
        self.positives = positives
        self.pareto = pareto
        self.data = data

    def fit(self, n_inputs: int, y:np.ndarray, verbose:int = 1):
        '''
        Fits a model on given regression data.
        '''
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n = 1
        loss_fkt = self.loss_fkt(y, self.data)
        params = {
            'n_inputs' : n_inputs,
            'n_outps' : n,
            'loss_fkt' : loss_fkt,
            'n_calc_nodes' : self.n_calc_nodes,
            'n_processes' : self.processes,
            'topk' : self.topk,
            'verbose' : verbose,
            'max_orders' : self.max_orders,
            'stop_thresh' : self.stop_thresh,
            'max_time' : self.max_time,
            'pareto' : self.pareto
        }
        if self.mode == 'hierarchical':
            res = hierarchical_search(**params)
        else:
            res = exhaustive_search(**params)
        self.results = res
        self.total_evals = res['total_evals']
        if verbose > 0:
            print(f'Total evaluated graphs: {self.total_evals}')

        if len(res['graphs']) > 0:
            best_idx = int(np.argmin(res['losses']))
            if verbose > 0:
                print(f'Found graph with loss {res["losses"][best_idx]}')

            self.cgraph = res['graphs'][best_idx]
            self.loss = res['losses'][best_idx]
        return self

    def predict(self, data: Stock_Data):
        '''
        Predicts values for given samples.
        '''
        assert hasattr(self, 'cgraph'), 'No graph found yet. Call .fit first!'
        pred = self.cgraph.evaluate(data)
        return pred

    def complexity(self):
        '''
        Complexity of expression (number of calculations)
        '''
        assert hasattr(self, 'cgraph'), 'No graph found yet. Call .fit first!'
        return self.cgraph.n_operations()
