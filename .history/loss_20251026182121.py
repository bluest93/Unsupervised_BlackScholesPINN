import torch
import torch.autograd as autograd

def pde_residual(model, S, t, r, sigma):
    C = model(S, t)
    dC_dt = autograd.grad(C, t, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True)[0]
    dC_dS = autograd.grad(C, S, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True)[0]
    d2C_dS2 = autograd.grad(dC_dS, S, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True)[0]
    return dC_dt + 0.5 * sigma**2 * S**2 * d2C_dS2 + r * S * dC_dS - r * C

def total_loss(model, S_terminal, t_terminal, C_terminal,
               S_boundary, t_boundary, C_boundary, S_colloc, t_colloc, r, sigma):
  
    pde = pde_residual(model, S_colloc, t_colloc, r, sigma)
    loss_pde = torch.mean(pde**2)

    # Terminal condition loss: C(S, T)
    C_pred_terminal = model(S_terminal, t_terminal)
    loss_terminal = torch.mean((C_pred_terminal - C_terminal)**2)

    # Boundary condition loss: C(S, 0) = 0 and C(S, T) = BS(S)
    C_pred_boundary = model(S_boundary, t_boundary)
    loss_boundary = torch.mean((C_pred_boundary - C_boundary)**2)

    return 2*loss_terminal + loss_boundary + loss_pde , loss_terminal, loss_boundary, loss_pde
