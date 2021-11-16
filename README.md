# physics-informed-learners

Two optimizers to choose from:

optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=0.01, 
    max_iter = 3500,
    tolerance_grad = 1.0 * np.finfo(float).eps, 
    tolerance_change = 1.0 * np.finfo(float).eps, 
    history_size = 100
)
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=0.01, 
    max_iter=50000, 
    max_eval=50000, 
    history_size=50,
    tolerance_grad=1.0 * np.finfo(float).eps, 
    tolerance_change=1.0 * np.finfo(float).eps,
    line_search_fn="strong_wolfe"       # can be "strong_wolfe"
)
