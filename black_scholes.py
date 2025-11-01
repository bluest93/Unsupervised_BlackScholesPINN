from model import PINN
from data import *
from loss import *
import torch
from abc import ABC, abstractmethod

# class BlackScholesPINN:
#     def __init__(self, config):
#         self.config = config
#         self.model = PINN()
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

#         # Prepare data
#         self.S_terminal, self.t_terminal, self.C_terminal = generate_terminal_data(config) 
#         self.S_boundary, self.t_boundary, self.C_boundary = generate_boundary_data(config)
#         self.S_colloc, self.t_colloc = generate_collocation_points(self.config)

#     def train(self):
        
#         for epoch in range(self.config["epochs"]):
#             self.optimizer.zero_grad()

            
#             # Losses
#             loss, loss_terminal, loss_boundary, loss_pde = total_loss_euro(
#                 self.model,
#                 self.S_terminal, self.t_terminal, self.C_terminal,
#                 self.S_boundary, self.t_boundary, self.C_boundary,
#                 self.S_colloc, self.t_colloc,
#                 self.config["r"],
#                 self.config["sigma"]
#             )
#             loss.backward()
#             self.optimizer.step()

#             if epoch % self.config["log_interval"] == 0:
#                 print(f"Epoch {epoch} | Total: {loss.item():.6f} | Terminal: {loss_terminal.item():.6f} | Boundary: {loss_boundary.item():.6f} | PDE: {loss_pde.item():.6f}")

    
#     def export(self):
#         torch.save(self.model.state_dict(), self.config.get("model_path", "model.pth"))

#     def predict(self, S_eval, t_eval):
#         with torch.no_grad():
#             return self.model(S_eval, t_eval)


class BaseBlackScholesPINN(ABC):
    def __init__(self, config):
        self.config = config
        self.model = PINN(hidden_layers= self.config["hidden_layers"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

        # Prepare data
        self.S_colloc, self.t_colloc = generate_collocation_points(self.config)

    @abstractmethod
    def train(self):
        """Compute total loss depending on different market."""
        pass

    def export(self):
        torch.save(self.model.state_dict(), self.config.get("model_path", "model.pth"))

    def predict(self, S_eval, t_eval):
        with torch.no_grad():
            return self.model(S_eval, t_eval)
        
class EuropeanPINN(BaseBlackScholesPINN):
    def __init__(self, config):
        super().__init__(config)
        self.S_terminal, self.t_terminal, self.C_terminal = generate_terminal_call_data(config) 
        self.S_boundary, self.t_boundary, self.C_boundary = generate_boundary_call_data(config)


    def train(self):
        
        for epoch in range(self.config["epochs"]):
            self.optimizer.zero_grad()
            
            # Losses
            loss, loss_terminal, loss_boundary, loss_pde = total_loss_euro(
                self.model,
                self.S_terminal, self.t_terminal, self.C_terminal,
                self.S_boundary, self.t_boundary, self.C_boundary,
                self.S_colloc, self.t_colloc,
                self.config["r"],
                self.config["sigma"]
            )
            loss.backward()
            self.optimizer.step()

            if epoch % self.config["log_interval"] == 0:
                print(f"Epoch {epoch} | Total: {loss.item():.6f} | Terminal: {loss_terminal.item():.6f} | Boundary: {loss_boundary.item():.6f} | PDE: {loss_pde.item():.6f}")

class AmericanPutPINN(BaseBlackScholesPINN):
    def __init__(self, config):
        super().__init__(config)
        self.S_terminal, self.t_terminal, self.C_terminal = generate_terminal_put_data(config) 
        self.S_boundary, self.t_boundary, self.C_boundary = generate_boundary_put_data(config)

    def train(self):
        for epoch in range(self.config["epochs"]):
            self.optimizer.zero_grad()

            # Losses
            loss, loss_terminal, loss_boundary, loss_pde, loss_payoff, loss_comp = total_loss_us(
                self.model,
                self.S_terminal, self.t_terminal, self.C_terminal,
                self.S_boundary, self.t_boundary, self.C_boundary,
                self.S_colloc, self.t_colloc,
                self.config
            )
            loss.backward()
            self.optimizer.step()

            if epoch % self.config["log_interval"] == 0:
                print(f"Epoch {epoch} | Total: {loss.item():.6f} | Terminal: {loss_terminal.item():.6f} | Boundary: {loss_boundary.item():.6f} | PDE: {loss_pde.item():.6f} | Payoff: {loss_payoff.item():.6f} | Complementarity: {loss_comp.item():.6f}")
