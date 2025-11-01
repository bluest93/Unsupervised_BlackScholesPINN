# Unsupervised Black-Scholes PINN

This repository extends the original Physics-Informed Neural Network (PINN) framework to solve both **European** and **American** option pricing problems under the Black–Scholes model.  
The training is entirely **unsupervised**, relying only on the governing PDE and boundary conditions — without any labeled or synthetic data.

This work builds on the [original notebook](https://github.com/PieroPaialungaAI/BlackScholesPINN/blob/main/example/BlackScholesModel.ipynb) by [**Piero Paialunga**](https://github.com/PieroPaialungaAI/BlackScholesPINN), simplifying the training process, introducing free-boundary logic, and incorporating the **American option formulation** based on [**American Option Lecture Notes (Byott, 2005)**](https://empslocal.ex.ac.uk/people/staff/NPByott/teaching/FinMaths/2005/american.pdf).


## Key Modifications


- **Unsupervised learning framework**  
  No synthetic or labeled data is used. The model learns purely from the PDE residuals and boundary/terminal constraints.

- **European (call) and American (put) option support**  
  A unified PINN structure handles both formulations:
  - European call option — solved directly from the Black–Scholes PDE.
  - American put option — solved as a **free-boundary problem** using complementarity conditions.

### European Market (Call Option)

The European call option is solved directly from the **Black–Scholes PDE**:

![equation](https://latex.codecogs.com/svg.image?\frac{\partial&space;V}{\partial&space;t}&plus;\frac{1}{2}\sigma^2&space;S^2\frac{\partial^2&space;V}{\partial&space;S^2}&plus;rS\frac{\partial&space;V}{\partial&space;S}-rV=0,)

with terminal condition

![equation](https://latex.codecogs.com/svg.image?V(S,T)=\max(S-K,0).)


The PINN learns to satisfy this PDE and boundary behavior without any labeled data.  
The results closely match the analytical **Black–Scholes formula**.


### American Market (Put Option)

The American put introduces a **free-boundary problem**, governed by the inequalities:


![equation](https://latex.codecogs.com/svg.image?\frac{\partial&space;V}{\partial&space;t}&plus;\frac{1}{2}\sigma^2&space;S^2\frac{\partial^2&space;V}{\partial&space;S^2}&plus;rS\frac{\partial&space;V}{\partial&space;S}-rV\le&space;0,\quad&space;V\ge\max(K-S,0),)


where at least one condition holds as equality at every point.  
This can be reformulated as the **linear complementarity condition**:

![equation](https://latex.codecogs.com/svg.image?(V-\max(K-S,0))\left(\frac{\partial&space;V}{\partial&space;t}&plus;\frac{1}{2}\sigma^2&space;S^2\frac{\partial^2&space;V}{\partial&space;S^2}&plus;rS\frac{\partial&space;V}{\partial&space;S}-rV\right)=0.)

The PINN loss incorporates these relations using ReLU and $\min(\cdot)$ operators to maintain the exercise and continuation region logic.


- **Analytical and numerical comparison**  
  The PINN results are compared against both:
  - The **Black–Scholes analytical solution** (European option), and  
  - The **QuantLib benchmark** (American option).

- **Notebook-only workflow**  
  Training and evaluation are performed entirely within a Jupyter notebook — no `main.py` or `train.py` scripts are used.


## Why These Changes Matter

- Eliminates the need for synthetic or labeled data.  
- Accurately captures **free-boundary behavior** in American options.  
- Improves PDE compliance and financial interpretability.  
- Demonstrates the potential of PINNs for **variational inequality problems** in quantitative finance.


## Credits

This project is based on the original [BlackScholesPINN repository](https://github.com/PieroPaialungaAI/BlackScholesPINN) by **Piero Paialunga**, which provided a clean and modular framework for PINN-based option pricing. All modifications here are intended for research and educational purposes.


## Author

**Blues**  
Exploring unsupervised learning and financial PDEs.

