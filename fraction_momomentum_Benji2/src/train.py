import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import os
import matplotlib.pyplot as plt
from .models import SimpleCNN
from .data import get_data_loaders
from .optimizers import HonestFractionalMomentum, FastFractionalMomentum

def train_and_evaluate(optimizer_name, optimizer, model, trainloader, testloader, epochs=3):
    criterion = nn.CrossEntropyLoss()
    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss / 1024 ** 2

    history = {'loss': [], 'acc': []}
    print(f"\n--- Обучение: {optimizer_name} ---")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
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
            for data in testloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        avg_loss = running_loss / len(trainloader)
        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    end_time = time.time()
    mem_end = process.memory_info().rss / 1024 ** 2
    
    return history, (end_time - start_time), (mem_end - mem_start)

def plot_results(results_dict, save_path='../results/comparison.png'):
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    for name, data in results_dict.items():
        plt.plot(data['history']['loss'], label=name, marker='o')
    plt.title('Кривая функции потерь (Loss)')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    for name, data in results_dict.items():
        plt.plot(data['history']['acc'], label=name, marker='s')
    plt.title('Точность (Accuracy) на тесте')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\nГрафики сохранены в {save_path}")
    plt.show()

if __name__ == "__main__":
    EPOCHS = 3
    LR = 0.01
    DATASET = 'MNIST'
    
    trainloader, testloader = get_data_loaders(DATASET)
    results = {}

    # 1. Baseline
    model_sgd = SimpleCNN()
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=LR, momentum=0.9)
    hist_sgd, time_sgd, mem_sgd = train_and_evaluate("SGD Classique", opt_sgd, model_sgd, trainloader, testloader, EPOCHS)
    results["SGD"] = {'history': hist_sgd, 'time': time_sgd, 'mem': mem_sgd}

    # 2. Honest
    model_honest = SimpleCNN()
    opt_honest = HonestFractionalMomentum(model_honest.parameters(), lr=LR, alpha=0.5, history_size=20)
    hist_honest, time_honest, mem_honest = train_and_evaluate("Fractionnaire Honnête", opt_honest, model_honest, trainloader, testloader, EPOCHS)
    results["Frac. Honnête"] = {'history': hist_honest, 'time': time_honest, 'mem': mem_honest}

    # 3. Fast
    model_fast = SimpleCNN()
    opt_fast = FastFractionalMomentum(model_fast.parameters(), lr=LR, betas=(0.5, 0.9, 0.99), coeffs=(0.6, 0.3, 0.1))
    hist_fast, time_fast, mem_fast = train_and_evaluate("Fractionnaire Rapide", opt_fast, model_fast, trainloader, testloader, EPOCHS)
    results["Frac. Rapide"] = {'history': hist_fast, 'time': time_fast, 'mem': mem_fast}

    plot_results(results, save_path='../results/comparison.png')

    # Таблица результатов
    print("\n" + "="*50)
    print(" ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*50)
    print(f"{'Метод':<25} | {'Время (с)':<10} | {'Память (MB)':<12} | {'Acc (%)'}")
    print("-" * 65)
    for name, data in results.items():
        print(f"{name:<25} | {data['time']:<10.2f} | {data['mem']:<12.2f} | {data['history']['acc'][-1]:.2f}")