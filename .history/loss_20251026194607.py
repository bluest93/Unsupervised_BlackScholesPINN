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
    # C_pred_boundary = model(S_boundary, t_boundary)
    # loss_boundary = torch.mean((C_pred_boundary - C_boundary)**2)

    N = S_boundary.shape[0] // 2
    S_min, S_max = S_boundary[:N], S_boundary[N:]
    t_min, t_max = t_boundary[:N], t_boundary[N:]
    C_min, C_max = C_boundary[:N], C_boundary[N:]

    # Lower boundary: C(min_S, t) = 0
    C_pred_min = model(S_min, t_min)
    loss_bc_min = torch.mean((C_pred_min-C_min)**2)

    # Upper boundary: ∂C/∂S (S_max, t) = 1
    S_max.requires_grad_(True)
    C_pred_max = model(S_max, t_max)
    dC_dS = torch.autograd.grad(
        C_pred_max, S_max,
        grad_outputs=torch.ones_like(C_pred_max),
        create_graph=True
    )[0]

    # Neumann condition: derivative should be 1
    loss_bc_max = torch.mean((dC_dS - C_max)**2)
    
    # Combine both boundary losses
    loss_boundary = loss_bc_min + loss_bc_max

    return loss_terminal + loss_boundary + loss_pde , loss_terminal, loss_boundary, loss_pde
