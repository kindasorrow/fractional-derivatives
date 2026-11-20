import torch
import math
from collections import deque
from torch.optim.optimizer import Optimizer
import numpy as np

class HonestFractionalMomentum(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.5, history_size=20):
        # Ajout d'un facteur de normalisation pour que le pas effectif soit similaire au SGD
        defaults = dict(lr=lr, alpha=alpha, history_size=history_size)
        super(HonestFractionalMomentum, self).__init__(params, defaults)

    def _get_fractional_integral_weights(self, alpha, k_max):
        """
        Calcule les poids de l'intégrale fractionnaire (tous positifs).
        Utilise la récursion stable proposée par le correcteur.
        """
        w = [1.0]
        for k in range(1, k_max + 1):
            w.append(w[-1] * (k - 1 + alpha) / k)
        # Normalisation : on divise par la somme pour garder un pas équitable
        w_tensor = torch.tensor(w, dtype=torch.float32)
        return w_tensor / w_tensor.sum()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                alpha = group['alpha']
                hist_size = group['history_size']
                
                # Calcul des coeffs une seule fois par groupe si pas déjà fait
                if 'coeffs' not in group:
                    group['coeffs'] = self._get_fractional_integral_weights(alpha, hist_size)

                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    grad = p.grad
                    state = self.state[p]

                    # Initialisation
                    if len(state) == 0:
                        state['history'] = deque(maxlen=hist_size + 1)

                    # Ajout du gradient actuel
                    state['history'].appendleft(grad.clone())

                    # Calcul de la mise à jour
                    update = torch.zeros_like(p)
                    coeffs = group['coeffs'].to(p.device)
                    
                    for i, g_old in enumerate(state['history']):
                        if i < len(coeffs):
                            update.add_(g_old, alpha=coeffs[i])

                    # Mise à jour des poids
                    p.add_(update, alpha=-lr)

        return loss


class FastFractionalMomentum(Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.99, 0.999), coeffs=(0.5, 0.3, 0.2)):
        # On met betas et coeffs dans defaults pour qu'ils soient dans le state_dict
        defaults = dict(lr=lr, betas=betas, coeffs=coeffs)
        super(FastFractionalMomentum, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                betas = group['betas']
                coeffs = group['coeffs']

                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]

                    if len(state) == 0:
                        state['step'] = 0
                        state['buffers'] = [torch.zeros_like(p) for _ in betas]

                    state['step'] += 1

                    # Mise à jour des buffers
                    for i, (buf, beta) in enumerate(zip(state['buffers'], betas)):
                        buf.mul_(beta).add_(grad, alpha=1 - beta)

                    # Somme pondérée
                    update = torch.zeros_like(p)
                    for buf, coeff in zip(state['buffers'], coeffs):
                        update.add_(buf, alpha=coeff)

                    p.add_(update, alpha=-lr)

        return loss

# --- OUTIL POUR TROUVER LES MEILLEURS BETA/COEFFS ---
def fit_fast_to_honest(alpha, history_size, K=3):
    """
    Trouve les meilleurs betas et coeffs pour la méthode rapide
    afin d'approximer le noyau de la méthode honnête.
    """
    # 1. Générer le noyau "Honnête"
    w = [1.0]
    for k in range(1, history_size + 1):
        w.append(w[-1] * (k - 1 + alpha) / k)
    w = np.array(w)
    w = w / w.sum()
    
    # 2. Choisir des betas fixes espacés
    betas = np.linspace(0.5, 0.99, K)
    
    # 3. Matrice A : colonnes = (1-beta) * beta^k
    A = np.zeros((history_size + 1, K))
    for j, beta in enumerate(betas):
        for k in range(history_size + 1):
            A[k, j] = (1 - beta) * (beta ** k)
            
    # 4. Résolution par moindres carrés (A * c = w)
    coeffs, _, _, _ = np.linalg.lstsq(A, w, rcond=None)
    
    return tuple(betas), tuple(coeffs)