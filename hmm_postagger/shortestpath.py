def _ford_list(E, V, S, T):

    ## Initialize ##
    # (max weight + 1) * num of nodes
    inf = (max((weight for from_, to_, weight in E)) + 1) * len(V)

    # distance
    d = {node:0 if node == S else inf for node in V}
    # previous node
    prev = {node:None for node in V}

    ## Iteration ##
    # preventing infinite loop
    for _ in range(len(V)):
        # for early stop
        changed = False
        for u, v, Wuv in E:
            d_new = d[u] + Wuv
            if d_new < d[v]:
                d[v] = d_new
                prev[v] = u
                changed = True
        if not changed:
            break

    # Checking negative cycle loop
    for u, v, Wuv in E:
        if d[u] + Wuv < d[v]:
            raise ValueError('Negative cycle exists')

    # Finding path
    prev_ = prev[T]
    if prev_ == S:
        return {'paths':[[prev_, S][::-1]], 'cost':d[T]}

    path = []
    while prev_ != S:
        path.append(prev_)
        prev_ = prev[prev_]
    path.append(S)

    return {'paths':[path[::-1]], 'cost':d[T]}