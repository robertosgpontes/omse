import pulp
import numpy as np

def create_x_variables():
    x1 = pulp.LpVariable("x1",lowBound = 0) 
    x2 = pulp.LpVariable("x2",lowBound = 0)
    x3 = pulp.LpVariable("x3",lowBound = 0) 
    x4 = pulp.LpVariable("x4",lowBound = 0) 
    return np.array([[x1], [x2], [x3], [x4]]) 
  
def load_model(lpm, A, X, b):
    Ax = A.dot(X)

    for i in range(0,4):
        lpm += Ax[i,0] <= b[i,0]

    lpm += Ax[4,0] == b[4,0]

    for i in range(5,7):
        lpm += Ax[i,0] >= b[i,0]

    return lpm
  
f_real = lambda c, x: c.dot(x)

f = lambda c, x: f_real(c, x)[0]

sig = lambda sigma: 1/(1 + np.exp(sigma*(np.linspace(0, NUM_ITER, NUM_ITER)-(NUM_ITER/2))))

def f_alpha(alpha, f1, f2):
  return (alpha*f1 + (1-alpha)*f2)

def run_models(f_alphas):
  solution_lst = []
  i = 0
  for alpha in f_alphas:
    model = pulp.LpProblem("MultiObjetivo", pulp.LpMaximize)

    X = create_x_variables()

    model += f_alpha(alpha, f(C[0], X), -f(C[1], X))

    model = load_model(model, A, X, b)

    solution = model.solve()

    X2 = np.array([pulp.value(X[0][0]), 
                    pulp.value(X[1][0]), 
                    pulp.value(X[2][0]), 
                    pulp.value(X[3][0])])
    
    solution_lst.append([i,
                    alpha,
                    str(pulp.LpStatus[solution])] +
                    X2.tolist() +
                    [pulp.value(model.objective),
                    f_real(C[0], X2),
                    -f_real(C[1], X2)]
                    )
    i += 1
    
  return pd.DataFrame(solution_lst, columns=["iter","alpha","status","x1","x2", "x3", "x4", "obj_value", "f1", "f2"])
