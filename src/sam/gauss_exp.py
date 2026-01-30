import numpy as np
import torch
import math
from .activations import get_activation

dtype = torch.float64
clip_th = 1 - 1e-9

def compute_grid_weigths(n=40,device="cuda"):
    """
    Compute 2D Gauss-Hermite quadrature grid and weights

    Inputs:
    -----------
    n : int
        Number of quadrature points per dimension
    device : str
        Device to use ('cuda' or 'cpu')

    Returns:
    -----------
    Z : torch.Tensor
        2D grid points of shape (n*n, 2)
    ws : torch.Tensor
        Weights of shape (n,)
    """
    zs, ws = np.polynomial.hermite.hermgauss(n)
    zs = torch.tensor(zs, dtype=dtype, device=device)
    ws = torch.tensor(ws, dtype=dtype, device=device)
    # Create 2D quadrature grid
    Z1, Z2 = torch.meshgrid(zs, zs, indexing="ij")
    return (Z1 , Z2)  , ws

def cholesky_pd(C, max_tries=6, eps=1e-12):
    C = (C + C.T) / 2  # enforce symmetry
    C = C.to(dtype=dtype)  # improve numerical stability
    I = torch.eye(C.shape[-1], device=C.device, dtype=C.dtype)
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(C + eps * I)
        except RuntimeError:
            eps *= 10
    # fallback: eigen clamp
    w, V = torch.linalg.eigh(C)
    w = torch.clamp(w, min=1e-15)
    return V @ torch.diag(torch.sqrt(w))


# def expectation_gauss_hermite_torch(F, Q, Q_star, M, grid , weigths, device="cuda",kwargs={}):
#     """ Compute expectation using 2D Gauss-Hermite quadrature
    
#     Inputs:
#     -----------
#     F : function
#         Function to compute expectation of. Should take two arguments (x,y, **kwargs)
#     Q : float
#         Variance of x
#     Q_star : float
#         Variance of y
#     M : float
#         Covariance between x and y
#     n : int
#         Number of quadrature points per dimension
#     device : str
#         Device to use ('cuda' or 'cpu') 
#     **kwargs : dict
#         Additional keyword arguments to pass to function F
#     Returns:
#     -----------
#     exp : float
#         Computed expectation
#     """
    
#     C = torch.tensor([[Q, M],
#                       [M, Q_star]], dtype=dtype, device=device)
#     L = cholesky_pd(C)
#     # Transform
#     XY = math.sqrt(2) * (grid @ L.T)
#     x, y = XY[:, 0], XY[:, 1]
#     vals = F(x, y,**kwargs)
#     W = torch.outer(weigths, weigths).reshape(-1)
#     return torch.sum(W * vals).item() / math.pi


def expectation_gauss_hermite(F, Q, Q_star, M, grid , weigths, kwargs={}):
    Z1, Z2 = grid

    z1 = math.sqrt(2) * Z1.reshape(-1)
    z2 = math.sqrt(2) * Z2.reshape(-1)

    Q = float(max(Q, 1e-12))
    Q_star = float(max(Q_star, 1e-12))

    vals = F(z1, z2, M, Q, Q_star, **kwargs)

    W = torch.outer(weigths, weigths).reshape(-1)

    return torch.sum(W * vals).item() / torch.pi



def get_gradient_function(tch_act,std_act):
    """
    Get the gradient function for the square loss between teacher and student activations
    Inputs:
    -----------
    tch_act : specs for teacher activation
    std_act : specs for student activation
    Returns:
    -----------
    gradient_function : function
        Function that computes the gradient of the square loss between teacher and student activations

    """
    tch_f, _ = get_activation(tch_act)
    std_f, std_deriv = get_activation(std_act)

    def gradient_function(h, h_star):
        return std_deriv(h) * (std_f(h) - tch_f(h_star))
    return gradient_function


def get_expectation_functions(tch_act,std_act):

    gradient_function  = get_gradient_function(tch_act,std_act)
    def F_M(z1,z2, M, Q, Q_star, rho = None):
        c = M / math.sqrt(Q_star*Q)
        c = float(np.clip(c, -clip_th, clip_th))
        h_star = math.sqrt(Q_star)*z1
        h = math.sqrt(Q)*(c*z1 + math.sqrt(1 - c**2)*z2)
        a = gradient_function(h,h_star)
        h_new = h + rho*torch.sign(a)
        a_new = gradient_function(h_new,h_star)
        return a_new*h_star

    def F_Q(z1,z2, M, Q, Q_star, rho = None, lr = None):
        c = M / math.sqrt(Q_star*Q)
        c = float(np.clip(c, -clip_th, clip_th))
        h_star = math.sqrt(Q_star)*z1
        h = math.sqrt(Q)*(c*z1 + math.sqrt(1 - c**2)*z2)
        a = gradient_function(h,h_star)
        h_new = h + rho*torch.sign(a)
        a_new = gradient_function(h_new,h_star)
        return 2*a_new*h + lr*a_new*a_new
    return F_M, F_Q


def derivatives(t,X,grid,weigths,funcs,parameters): # Q_star , grid , weigths , F_M, F_Q, device  , rho ,lr

    F_M, F_Q = funcs
    Q_star = parameters['Q_star']
    rho = parameters['rho']
    lr = parameters['lr']
    gamma = parameters['gamma']

    M , Q = X
    M_exp = expectation_gauss_hermite(F_M, Q, Q_star, M, grid , weigths, kwargs={'rho':rho})
    Q_exp = expectation_gauss_hermite(F_Q, Q, Q_star, M, grid , weigths, kwargs={'rho':rho,'lr':lr})

    dM = -gamma*M - lr * M_exp
    dQ = -2*gamma*Q - lr * Q_exp
    return np.array([dM,dQ])


def get_loss_function(tch_act,std_act):
    tch_f, _ = get_activation(tch_act)
    std_f, _ = get_activation(std_act)

    def loss_function(z1,z2, M, Q, Q_star):

        c = M / math.sqrt(Q_star*Q)
        c = float(np.clip(c, -clip_th, clip_th))
        h_star = math.sqrt(Q_star)*z1
        h = math.sqrt(Q)*(c*z1 + math.sqrt(1 - c**2)*z2)

        return 0.5 * (std_f(h) - tch_f(h_star))**2
    
    return loss_function

