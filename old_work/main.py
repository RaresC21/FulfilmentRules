import numpy as np
import gurobipy as gp
from gurobipy import GRB

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from problem import Problem
from data import DataGen
from models import * 

import matplotlib.pyplot as plt 

def eval_forecast_model(X_train, model, D):    
    q = model(X_train) 
    return fulfilment_loss(q, D) 

def generate_samples(n_samples, means, rho): 
    noises = torch.rand(n_samples, means.shape[0]) * rho
    return (torch.tensor(means) * ( 1 + noises)).detach().numpy()

def eval_model(forecast, X_test, Y_test, n_samples = 10):
    all_costs = [] 
    all_decisions = []

    saa_rho = 0.1
    saa_n_samples = n_samples

    if n_samples == 0: 
        saa_rho = 0 

    t = 0
    for x,d in zip(X_test, Y_test): 
        if t > 300: break
        t += 1
        
        pred = forecast(x)
        samples = generate_samples(saa_n_samples, pred, saa_rho)
        decision, _ = saa(samples, H, B, cross_costs, test=False) 
        _, cost = saa(d.unsqueeze(0).detach().numpy(), H, B, cross_costs, decision[0].detach().numpy(), test=True)
        all_costs.append(cost)
        print(t, " ", np.mean(all_costs))
        all_decisions.append(decision[0].numpy())
    all_decisions = np.array(all_decisions)
    all_decisions = torch.tensor(all_decisions) 
    return all_costs, all_decisions

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
     
    # make random distances
    n_nodes = 20
    n_features = 10

    H = torch.tensor([1 for i in range(n_nodes)])
    B = torch.tensor([10 for i in range(n_nodes)])  

    problem_ = Problem(H, B, n_nodes)
    cross_costs = problem_.cross_costs

    n_data = 1000
    n_test = 1000
    data_generator = DataGen(n_features, n_nodes)
    X_train, Y_train, X_test, Y_test = data_generator.get_test_train(n_data, n_test)

    print("Train Two-Stage -------------------------------------")
    two_stage_forecast = train_two_stage(problem_, X_train, Y_train, X_test, Y_test)

    
    print("Train end-to-end -------------------------------------")
    task_forecast = train_task_loss(problem_, X_train, Y_train, X_test, Y_test, two_stage_forecast)

    
    two_cost, two_dec = eval_model(two_stage_forecast, X_test, Y_test)
    task_cost, task_dec = eval_model(task_forecast, X_test, Y_test, n_samples = 0)

    qs = np.arange(0.9,1,0.0001)
    two_stage_results = [np.quantile(two_cost, q) for q in qs]
    task_results = [np.quantile(task_cost, q) for q in qs]

    plt.plot(qs, task_results, label = 'end-to-end')
    plt.plot(qs, two_stage_results, label = '2-stage')
    plt.legend()
    plt.xlabel("Quantile")
    plt.ylabel("Cost")
    plt.title("Cost Distribution (20 locations)")
    plt.show()