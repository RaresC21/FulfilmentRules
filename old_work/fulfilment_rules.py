import torch 


def rule_loss(q, d):
    aa = torch.zeros(q.shape[0], n_nodes, n_nodes)

    for i in range(n_nodes):
        for j in range(n_nodes):
            curq = q[:,i]
            curd = d[:,j]
            for edge in cross_costs_ordered:
                l,k = edge[1]
                if l == i:
                    curq = curq - aa[:,l,k]
                if k == j:   
                    curd = curd - aa[:,l,k]
            aa[:,i,j] = torch.minimum(curq, curd)

    holding_cost = torch.sum(torch.sum(H * (q - torch.sum(aa, dim=2)), dim = 1), dim = 0)
    backorder_cost = torch.sum(torch.sum(B * (d - torch.sum(aa, dim=1)), dim = 1), dim = 0)
    edge_cost = torch.sum(aa * cross_costs)
    return holding_cost + backorder_cost + edge_cost 
