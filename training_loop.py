from utils import log_experiment
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
import wandb

def train_model(net, train_loader, val_loader, test_loader, device, eval_batchsize, error_minimizer, criterion, scheduler, patience, epochs, model_type='resnet'):
    best_auc = 0.0
    early_stopping_patience = patience
    counter = 0
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    epochs_list = []

    # Deep copy the network to save the best model
    net_final = deepcopy(net)

    for epoch in range(epochs):
        net.train() # Set the network to training mode
        
        print("\n### Epoch {}:".format(epoch + 1))

        total_train_examples = 0
        num_correct_train = 0
        total_train_loss = 0

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            # Move images and labels to the device
            images = images.to(device)
            labels = labels.to(device)

            # Zero the gradients
            error_minimizer.zero_grad()

            # Perform forward pass
            if model_type == 'resnet':
                predictions = net(images)
            elif model_type == 'vit':
               predictions = net(images).logits 

            # Compute loss
            loss = criterion(predictions, labels)

            # Perform backward pass
            loss.backward()

            # Update weights
            error_minimizer.step()

            # Accumulate the loss and correct predictions
            total_train_loss += loss.item() * images.size(0)
            _, predicted_class = predictions.max(1)
            total_train_examples += predicted_class.size(0)
            num_correct_train += predicted_class.eq(labels).sum().item()

        # Calculate average training accuracy and loss
        train_acc = num_correct_train / total_train_examples
        train_accs.append(train_acc)

        train_loss_avg = total_train_loss / total_train_examples
        train_losses.append(train_loss_avg)

        print("Training Metrics: Accuracy: {:.4f} | Loss: {:.4f}".format(train_acc, train_loss_avg))

        total_val_examples = 0
        num_correct_val = 0
        total_val_loss = 0

        val_labels = []
        val_preds = []

        # Validation Phase
        net.eval()  # Set the network to evaluation mode

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                # Move images and labels to the device
                images = images.to(device)
                labels = labels.to(device)

                # Perform forward pass
                if model_type == 'resnet':
                    predictions = net(images)
                elif model_type == 'vit':
                    predictions = net(images).logits

                # Compute loss
                val_loss = criterion(predictions, labels)
                total_val_loss += val_loss.item() * images.size(0)
                
                _, predicted_class = predictions.max(1)
                total_val_examples += labels.size(0)
                num_correct_val += predicted_class.eq(labels).sum().item()

                # Store true labels and predictions for AUC calculation
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predictions[:, 1].detach().cpu().numpy())


        # Calculate average validation accuracy and loss
        val_acc = num_correct_val / total_val_examples
        val_accs.append(val_acc)

        val_loss_avg = total_val_loss / total_val_examples
        val_losses.append(val_loss_avg)

        # Calculate AUC for validation
        val_auc = roc_auc_score(val_labels, val_preds)

        # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step()

        print("Validation Metrics: Accuracy: {:.4f} | Loss: {:.4f} | AUC: {:.4f}".format(val_acc, val_loss_avg, val_auc))

        epochs_list.append(epoch + 1)

        # Check if validation AUC has improved
        if val_auc > best_auc:
            best_auc = val_auc
            print("Validation AUC improved --> Saving model...")
            # Save the best model
            net_final = deepcopy(net)
            counter = 0
        
        else:
            counter += 1

            # Check if early stopping should be triggered
            if counter >= early_stopping_patience:
                print("Early stopping triggered after {} epochs.".format(epoch))
                break

        # Log to wandb
        wandb.log({
            "train_loss": train_loss_avg,
            "train_accuracy": train_acc,
            "val_loss": val_loss_avg,
            "val_accuracy": val_acc,
            "val_auc": val_auc,
            "epoch": epoch
        })

    # Log final experiment results
    log_experiment(net_final, epochs_list, train_accs, val_accs, train_losses, val_losses, test_loader, device, eval_batchsize, model_type)