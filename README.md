# BlackScholesPINN

A Python implementation of Physics-Informed Neural Networks (PINNs) for solving the Black-Scholes partial differential equation used in option pricing.

---

## 📌 What is this?

This repository demonstrates how to use Physics-Informed Neural Networks (PINNs) to learn the solution of the **Black-Scholes equation** — a foundational model in financial mathematics for pricing European call options.

PINNs are neural networks that are trained not just on data, but also on the **underlying physical (or financial) laws** described by differential equations.

---

## 🚀 Features

- ✅ Clean modular design
- ✅ Configurable via `config.json`
- ✅ Supports noisy synthetic data generation
- ✅ Enforces PDE constraint using autograd
- ✅ Lightweight and dependency-free (only PyTorch + NumPy + matplotlib)
- ✅ Fully reproducible

---

## 🧠 What You’ll Learn

- How to generate synthetic financial data using the Black-Scholes formula
- How to train a neural network to obey a PDE using automatic differentiation
- How to combine **data loss** and **PDE loss** in a single objective
- How to modularize ML code for experimentation and reuse

---

## 🗂 Project Structure

. ├── black_scholes.py # Main wrapper class for training/evaluation ├── config.json # All key hyperparameters ├── data.py # Synthetic data and collocation point generation ├── loss.py # PDE residual and total loss function ├── model.py # Neural network architecture (PINN) ├── train.py # Training loop ├── utils.py # Black-Scholes analytical solution ├── temp.ipynb # Notebook for dev or exploration └── README.md # This file
