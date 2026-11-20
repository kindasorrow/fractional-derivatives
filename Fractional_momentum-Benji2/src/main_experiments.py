import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from src.models import SimpleCNN
from src.data import get_data_loaders
from src.train import run_experiment, plot_kernels
from src.optimizers import HonestFractionalMomentum, FastFractionalMomentum

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    EPOCHS = 5
    RUNS = 5
    DATASET = 'MNIST'
    ALPHA = 0.5
    N = 20
    K = 4

    os.makedirs('results', exist_ok=True)
    
    # 1. Tracer les noyaux (preuve mathématique de l'approximation)
    print("Génération du graphique des noyaux...")
    plot_kernels(ALPHA, N, K, save_path='results/kernels.png')
    
    trainloader, testloader = get_data_loaders(DATASET)
    base_model = SimpleCNN(in_channels=1 if DATASET == 'MNIST' else 3)
    
    # Dictionnaire pour stocker les résultats (Moyenne et écart-type)
    results = {
        "SGD": {"acc": [], "mem": []},
        "Honnête": {"acc": [], "mem": []},
        "Rapide": {"acc": [], "mem": []}
    }

    print(f"Lancement des expériences ({RUNS} runs, {EPOCHS} époques)...")
    
    for run in range(RUNS):
        print(f"--- Run {run+1}/{RUNS} ---")
        set_seed(42 + run) # Seed différent par run
        
        # Note sur l'équité : 
        # Le SGD a un momentum de 0.9. Le pas effectif est ~ (1/(1-0.9)) = 10.
        # Pour les fractions, comme les poids somment à 1, le pas effectif est 1.
        # On multiplie donc le LR des fractions par 10 pour égaliser la taille des pas.
        lr_sgd = 0.01
        lr_frac = 0.1 
        
        # SGD
        hist_sgd, mem_sgd = run_experiment(torch.optim.SGD, "SGD", base_model, trainloader, testloader, 
                                           lr=lr_sgd, epochs=EPOCHS)
        results["SGD"]["acc"].append(hist_sgd['acc'][-1])
        results["SGD"]["mem"].append(mem_sgd)
        
        # Honnête
        hist_honest, mem_honest = run_experiment(HonestFractionalMomentum, "Honnête", base_model, trainloader, testloader,
                                                 lr=lr_frac, alpha=ALPHA, history_size=N, epochs=EPOCHS)
        results["Honnête"]["acc"].append(hist_honest['acc'][-1])
        results["Honnête"]["mem"].append(mem_honest)
        
        # Rapide
        hist_fast, mem_fast = run_experiment(FastFractionalMomentum, "Fast", base_model, trainloader, testloader,
                                             lr=lr_frac, alpha=ALPHA, history_size=N, K=K, epochs=EPOCHS)
        results["Rapide"]["acc"].append(hist_fast['acc'][-1])
        results["Rapide"]["mem"].append(mem_fast)

    # --- AFFICHAGE DU TABLEAU FINAL ---
    print("\n" + "="*60)
    print(" RESULTATS FINAUX (Moyenne ± Écart-type sur {} runs)".format(RUNS))
    print("="*60)
    print(f"{'Méthode':<15} | {'Accuracy (%)':<20} | {'Mémoire Opt. (MB)'}")
    print("-" * 60)
    for name, data in results.items():
        acc_mean = np.mean(data['acc'])
        acc_std = np.std(data['acc'])
        mem_mean = np.mean(data['mem'])
        print(f"{name:<15} | {acc_mean:.2f} ± {acc_std:.2f}       | {mem_mean:.2f} MB")
    print("="*60)
    
    print("\nLes graphiques (noyaux) ont été sauvegardés dans /results/.")