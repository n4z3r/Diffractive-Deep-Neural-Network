# -*- coding: utf-8 -*-
"""
STABLE TRAINING VERSION – dynamic layers + example images saved in runs/.../examples/
"""

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
from model import Onn
from torch import optim
from label_generator import label_generator, eval_accuracy
import matplotlib.pyplot as plt
from loss import npcc_loss
import time
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchvision.datasets as datasets
from datetime import datetime


def train(onn, criterion, optimizer, train_loader, val_loader, save_dir, epoch_num=50, device='cuda:0'):
    label_set = label_generator()
    train_losses = []
    train_accies = []
    val_losses = []
    val_accies = []

    for epoch in range(epoch_num):
        train_loss = 0.0
        acc_sum = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels_long = labels.to(torch.long)
            targets = label_set[labels_long].to(device)

            optimizer.zero_grad()
            outputs = onn(inputs)
            I = torch.abs(outputs)**2

            loss = criterion(I, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc, _ = eval_accuracy(I, labels)
            acc_sum += train_acc.item()

            if (i + 1) % 32 == 0:
                train_log = f'epoch {epoch + 1} {i+1}, train loss: {train_loss/32: 5f}, train accuracy: {acc_sum/32: 5f}'
                train_losses.append(train_loss / 32)
                train_accies.append(acc_sum / 32)
                train_loss = 0.0
                acc_sum = 0.0

                torch.save(onn, f'{save_dir}/models/onn{epoch+1}.pt')

                with torch.no_grad():
                    val_loss, val_acc, I_val, labels_val = validation(onn, val_loader, criterion, device)
                    val_log = f'validation loss: {val_loss:5f}, validation accuracy: {val_acc:5f}'
                    val_losses.append(val_loss)
                    val_accies.append(val_acc)

                print(train_log, '\n', val_log)
                with open(f'{save_dir}/log.txt', "a", encoding='utf-8') as f:
                    f.write(train_log + '\n' + val_log + '\n')

    return onn, train_losses, train_accies, val_losses, val_accies, I_val, labels_val


def validation(onn, val_loader, criterion, device='cuda:0'):
    label_set = label_generator()
    val_loss_sum = 0.0
    val_acc_sum = 0.0
    last_I = None
    last_labels = None
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels_long = labels.to(torch.long)
        targets = label_set[labels_long].to(device)

        outputs = onn(inputs)
        I = torch.abs(outputs)**2
        val_loss = criterion(I, targets)
        val_acc, _ = eval_accuracy(I, labels)
        val_loss_sum += val_loss.item()
        val_acc_sum += val_acc.item()
        last_I = I
        last_labels = labels

    return val_loss_sum / (i + 1), val_acc_sum / (i + 1), last_I, last_labels


def save_all_results(onn, train_losses, train_accies, val_losses, val_accies, 
                     I_val, labels_val, save_dir, device, num_examples=16):
    os.makedirs(f'{save_dir}/examples', exist_ok=True)

    # Training curves
    epochs = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    ax1.plot(epochs, train_losses, '-o', label='train')
    ax1.plot(epochs, val_losses, '-s', label='validation')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()
    ax2.plot(epochs, train_accies, '-o', label='train')
    ax2.plot(epochs, val_accies, '-s', label='validation')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Phase masks
    phases = onn.get_phase_masks()
    ncols = min(4, len(phases))
    nrows = (len(phases) + ncols - 1) // ncols
    plt.figure(figsize=(5 * ncols, 5 * nrows), dpi=200)
    for i, phase_np in enumerate(phases):
        plt.subplot(nrows, ncols, i + 1)
        im = plt.imshow(phase_np % (2 * np.pi), cmap='twilight_shifted')
        plt.colorbar(im)
        plt.title(f'Layer {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/phase_masks.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Test set + examples + confusion matrix
    print("Evaluating full test set + saving examples...")
    trans = transforms.Compose([Resize(256), ToTensor()])
    test_dataset = datasets.MNIST(root="../data", train=False, transform=trans, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    label_set = label_generator().to(device)
    all_preds = []
    all_labels = []
    example_inputs = []
    example_outputs = []
    example_labels_true = []

    onn.eval()
    with torch.no_grad():
        collected = 0
        for inputs, labels in test_loader:
            inputs_device = inputs.to(device)
            outputs = onn(inputs_device)
            I = torch.abs(outputs)**2
            _, preds = eval_accuracy(I, labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

            if collected < num_examples:
                can_take = min(num_examples - collected, inputs.size(0))
                example_inputs.extend(inputs[:can_take].cpu())
                example_outputs.extend(I[:can_take].cpu())
                example_labels_true.extend(labels[:can_take].cpu().numpy())
                collected += can_take
                if collected >= num_examples:
                    break

    final_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    with open(f'{save_dir}/accuracy.txt', 'w') as f:
        f.write(f'Final test accuracy: {final_acc*100:.4f}%\n')
    print(f'Final test accuracy: {final_acc*100:.4f}%')

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Accuracy {final_acc*100:.2f}%')
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save example images: input MNIST digit vs output intensity pattern
    for i in range(min(num_examples, len(example_inputs))):
        inp = example_inputs[i].squeeze().numpy()
        out = example_outputs[i].squeeze().numpy()
        true_label = example_labels_true[i]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
        ax1.imshow(inp, cmap='gray')
        ax1.set_title(f'Input - Label: {true_label}')
        ax1.axis('off')

        ax2.imshow(out, cmap='gray')
        ax2.set_title('Output Intensity')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/examples/example_{i:02d}.png', dpi=200, bbox_inches='tight')
        plt.close()

    print(f"All results + {min(num_examples, len(example_inputs))} example images saved to: {save_dir}/")


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"runs/run_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/examples", exist_ok=True)
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    print(f"Starting new run → {save_dir}/")

    # ==================== CHANGE NUMBER OF LAYERS HERE ====================
    num_phase_layers = 4                      # ← change this number
    z_list = [30] * (num_phase_layers + 1)    # uniform spacing
    # z_list = [20, 30, 40, 50, 60][:num_phase_layers+1]  # example: different distances
    # ===================================================================

    c = 3e8 * 1e3
    f = 400e9
    lambda0 = c / f
    L = 80

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    M = 256
    batch_size = 128
    epoch_num = 50

    trans = transforms.Compose([Resize(M), ToTensor()])

    mnist_train = torchvision.datasets.MNIST(root="../data", train=True, transform=trans, download=True)
    train_set, val_set, _ = torch.utils.data.random_split(mnist_train, [4096, 512, 60000 - 4096 - 512])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)

    onn = Onn(M, L, lambda0, z_list).to(device)
    optimizer = optim.Adam(onn.parameters(), lr=1e-2)
    criterion = npcc_loss

    start_time = time.time()
    onn, train_losses, train_accies, val_losses, val_accies, I_val, labels_val = train(
        onn, criterion, optimizer, train_loader, val_loader, save_dir, epoch_num, device
    )
    end_time = time.time()
    print(f'Total training time: {end_time - start_time:.1f} seconds')

    save_all_results(onn, train_losses, train_accies, val_losses, val_accies,
                     I_val, labels_val, save_dir, device, num_examples=16)

    print("✅ Training complete! Now run `python export_results.py` for fabrication files.")