import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
from .models import SimpleCNN
from .data import get_data_loaders
from .optimizers import HonestFractionalMomentum, FastFractionalMomentum, fit_fast_to_honest

def get_optimizer_memory(optimizer):
    """Calcule la taille réelle de l'état de l'optimiseur en MB."""
    total_bytes = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                total_bytes += v.element_size() * v.nelement()
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, torch.Tensor):
                        total_bytes += item.element_size() * item.nelement()
    return total_bytes / (1024 ** 2)

def train_and_evaluate(optimizer_name, optimizer, model, trainloader, testloader, epochs=3):
    criterion = nn.CrossEntropyLoss()
    history = {'loss': [], 'acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                outputs = model(images)
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        avg_loss = running_loss / len(trainloader)
        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        
    mem = get_optimizer_memory(optimizer)
    return history, mem

def run_experiment(optimizer_class, name, model, trainloader, testloader, **kwargs):
    model_copy = copy.deepcopy(model)
    if name == "SGD":
        opt = optim.SGD(model_copy.parameters(), lr=kwargs['lr'], momentum=0.9)
    elif name == "Fast":
        betas, coeffs = fit_fast_to_honest(kwargs['alpha'], kwargs['history_size'], K=kwargs['K'])
        opt = optimizer_class(model_copy.parameters(), lr=kwargs['lr'], betas=betas, coeffs=coeffs)
    else:
        opt = optimizer_class(model_copy.parameters(), lr=kwargs['lr'], alpha=kwargs['alpha'], history_size=kwargs['history_size'])
        
    hist, mem = train_and_evaluate(name, opt, model_copy, trainloader, testloader, epochs=kwargs['epochs'])
    return hist, mem

def plot_kernels(alpha, history_size, K, save_path='../results/kernels.png'):
    """Trace les noyaux (poids) de la méthode Honnête vs Rapide."""
    w_honest = [1.0]
    for k in range(1, history_size + 1):
        w_honest.append(w_honest[-1] * (k - 1 + alpha) / k)
    w_honest = np.array(w_honest)
    w_honest = w_honest / w_honest.sum()
    
    betas, coeffs = fit_fast_to_honest(alpha, history_size, K)
    w_fast = np.zeros_like(w_honest)
    for b, c in zip(betas, coeffs):
        for k in range(history_size + 1):
            w_fast[k] += c * (1 - b) * (b ** k)
            
    plt.figure(figsize=(8, 5))
    plt.plot(w_honest, label="Honnête (Intégrale frac.)", marker='o', linestyle='--')
    plt.plot(w_fast, label=f"Rapide (K={K})", marker='x')
    plt.title(f"Comparaison des noyaux de mémoire (α={alpha}, N={history_size})")
    plt.xlabel("k (pas de temps en arrière)")
    plt.ylabel("Poids du gradient")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()