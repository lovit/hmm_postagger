def sent_to_graph(sent, sent_len):
    idx2node = [node for words in sent for node in words]
    node2idx = {node:idx for idx, node in enumerate(idx2node)}
    sent_idx = [[node2idx[word] for word in words] for words in sent]
    sent_idx_observed = [
        [node2idx[word] for word in words if word[3] > word[2]]
        for words in sent]
    edges = {}

    for b, words in enumerate(sent):
        for word in words:
            from_idx = node2idx[word]
            if word[3] > sent_len: # EOS
                continue
            if word[3] > word[2]:
                edges[from_idx] = sent_idx[word[3]]
            else:
                edges[from_idx] = sent_idx_observed[word[3]]

    return edges, idx2node

def ford(g, start, end):
    from_nodes, to_nodes = _set_nodes_dict(g)
    if not ((start in from_nodes) and (end in to_nodes)):
        raise ValueError('There is no path {} - ... - {}'.format(start, end))

    n_nodes, n_edges, cost = _initialize_dict(g, start)

    for _iter in range(n_nodes * n_edges):
        cost, changed = _update_ford_dict(g, cost)
        if not changed:
            break
    paths = _find_shortest_path_dict(g, start, end, cost, n_nodes)
    return {'paths': paths, 'cost': cost[end]}

def _update_ford_dict(g, cost):
    changed = False
    for from_, to_weight in g.items():
        for to_, weight in to_weight.items():
            if cost[to_] > cost[from_] + weight:
                before = cost[to_]
                after = cost[from_] + weight
                cost[to_] = after
                changed = True
    return cost, changed

def _set_nodes_dict(g):
    from_nodes = set(g)
    to_nodes = {node for nw in g.values() for node in nw.keys()}
    return from_nodes, to_nodes

def _initialize_dict(g, start):
    nodes = set(g.keys())
    nodes.update(set({n for nw in g.values() for n in nw.keys()}))
    n_nodes = len(nodes)
    n_edges = sum((len(nw) for nw in g.values()))
    max_weight = max(w for nw in g.values() for w in nw.values())

    init_cost = n_nodes * (max_weight + 1)
    cost = {node:(0 if node == start else init_cost) for node in nodes}
    return n_nodes, n_edges, cost