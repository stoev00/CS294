import torch
import torch.nn.functional as nnF
import numpy as np

def burgers_pde_residual(x, t, u):
    # x: (B, Nx, Nt)
    # t: (B, Nx, Nt)
    # u: (B, Nx, Nt)

    print(x.shape,t.shape)
    nu = 0.01
    dt = 1/(t.size()[2]-1)
    dx = 1/(x.size()[1]-1)
    # TODO
    dudx = torch.zeros_like(u)
    dudt = torch.zeros_like(u)
    du2dx2 = torch.zeros_like(u)
    zero_m = torch.zeros_like(u)

    #print(dudx.shape)
    '''
    for k in range(0,x.size()[0]-1):
        for i in range(1,t.size()[2]-1):
            for j in range(1,x.size()[1]-1):

                dudx[k][j][i] = (u[k, j + 1, i] - u[k, j - 1, i]) / (2*dx)
                du2dx2[k][j][i] = (u[k, j + 1, i] + u[k, j - 1, i] - 2 * u[k, j, i]) / dx ** 2
                dudt[k][j][i] = (u[k, j, i + 1] - u[k, j, i - 1]) / (2*dt)
                #Edges using forward euler
                dudx[k][0][i] = (u[k, 1, i] - u[k, 0, i]) / dx
                dudx[k][-1][i] = (u[k, -1, i] - u[k, -2, i]) / dx
                dudt[k][j][0] = (u[k, j,  1] - u[k, j, 0]) / dt
                dudt[k][j][-1] = (u[k, j, - 1] - u[k, j, -2]) / dt '''






    resid = dudt +0.5* dudx**2 - nu * du2dx2
    loss = torch.nn.MSELoss()
    loss = loss(resid,zero_m)

    bc_loss = torch.nn.MSELoss()
    bc_loss = bc_loss(u[:,0,:],u[:,x.size()[1]-1,:])

    bc_loss2 = torch.nn.MSELoss()
    bc_loss2 = bc_loss2(dudx[:,0,:],dudx[:,x.size()[1]-1,:])

    return loss+bc_loss+bc_loss2


def burgers_data_loss(predicted, target):
    # Relative L2 Loss
    # Predicted: (B, Nx, Nt)
    # Target: (B, Nx, Nt)
    # TODO
    num = torch.norm(predicted - target, p =2)
    den = torch.norm(target,p=2)
    return num/den

