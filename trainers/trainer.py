import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import io
import tqdm

from torchvision import transforms
from PIL import Image
from datetime import datetime
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

class MultiTaskLoss:
    def __init__(self, weights={'pathology': 1.0, 'region': 1.0, 'depth': 1.0}):
        self.weights = weights
        self.criterion = nn.CrossEntropyLoss()
        
    def __call__(self, predictions, targets):
        """
        predictions: dict or tuple of (pathology_pred, region_pred, depth_pred)
        targets: dict of { 'pathology': ..., 'region': ..., 'depth': ... }
        """
        losses = {}
        total_loss = 0
        
        for task, weight in self.weights.items():
            losses[task] = self.criterion(predictions[task], targets[task])
            total_loss += weight * losses[task]
            
        return total_loss, losses

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 device, num_epochs=100, early_stopping_patience=10,
                 exp_name=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.patience = early_stopping_patience
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Initialize TensorBoard writer
        if exp_name is None:
            exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f'runs/{exp_name}')
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': {}, 'val_acc': {}
        }
        
    def log_images(self, images, epoch):
        """Log a batch of images to TensorBoard."""
        img_grid = torchvision.utils.make_grid(images, normalize=True)
        self.writer.add_image('Sample Images', img_grid, epoch)
    
    def log_confusion_matrices(self, epoch):
        self.model.eval()
        predictions = {
            'pathology': [], 'region': [], 'depth': []
        }
        targets = {
            'pathology': [], 'region': [], 'depth': []
        }
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                pathology_pred, region_pred, depth_pred = self.model(images)
                outputs = {
                    'pathology': pathology_pred,
                    'region': region_pred,
                    'depth': depth_pred
                }
                
                for task in ['pathology', 'region', 'depth']:
                    pred = torch.argmax(outputs[task], dim=1).cpu().numpy()
                    target = batch[task].cpu().numpy()
                    predictions[task].extend(pred)
                    targets[task].extend(target)
        
        # Create and log confusion matrices
        for task in ['pathology', 'region', 'depth']:
            cm = confusion_matrix(targets[task], predictions[task])
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_title(f'{task.capitalize()} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            img = transforms.ToTensor()(img)
            
            self.writer.add_image(f'{task}/confusion_matrix', img, epoch)
            plt.close()
    
    def log_model_weights_and_grads(self, epoch):
        """
        Log histograms of the parameters and their gradients.
        """
        for name, param in self.model.named_parameters():
            # Log parameter values
            self.writer.add_histogram(f'weights/{name}', param, epoch)
            # Log gradients (if they exist)
            if param.grad is not None:
                self.writer.add_histogram(f'grads/{name}', param.grad, epoch)
    
    def log_pr_curves(self, predictions, targets, epoch, prefix=''):
        """
        Log precision-recall curves for each task, if multi-class we do it for class 0 or 1, 
        you can adapt for each class as needed.
        """
        for task in ['pathology', 'region', 'depth']:
            preds_softmax = F.softmax(predictions[task], dim=1)
            # Let's just pick the probability of 'class 1' for demonstration
            if preds_softmax.shape[1] > 1:
                positive_probs = preds_softmax[:, 1]
            else:
                positive_probs = preds_softmax[:, 0]
            
            self.writer.add_pr_curve(
                f'{prefix}PR/{task}', 
                labels=targets[task], 
                predictions=positive_probs, 
                global_step=epoch
            )
    
    def train_epoch(self, epoch):
        scaler = torch.amp.GradScaler('cuda')
        self.model.train()
        total_loss = 0
        task_correct = {'pathology': 0, 'region': 0, 'depth': 0}
        task_total = {'pathology': 0, 'region': 0, 'depth': 0}
        task_losses = {'pathology': 0, 'region': 0, 'depth': 0}
        
        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}", position=0, leave=True)
        ):
            images = batch['image'].to(self.device)
            targets = {
                task: batch[task].to(self.device)
                for task in ['pathology', 'region', 'depth']
            }
            
            if epoch == 0 and batch_idx == 0:
                self.log_images(images, epoch)
            
            self.optimizer.zero_grad()
            
            pathology_pred, region_pred, depth_pred = self.model(images)
            predictions = {
                'pathology': pathology_pred,
                'region': region_pred,
                'depth': depth_pred
            }
            
            loss, individual_losses = self.criterion(predictions, targets)
            with torch.amp.autocast('cuda'):
                pathology_pred, region_pred, depth_pred = self.model(images)
                predictions = {
                'pathology': pathology_pred,
                'region': region_pred,
                'depth': depth_pred
            }
            loss, individual_losses = self.criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            total_loss += loss.item()
            for task, t_loss in individual_losses.items():
                task_losses[task] += t_loss.item()
            
            # Calculate accuracy
            for task in ['pathology', 'region', 'depth']:
                pred = torch.argmax(predictions[task], dim=1)
                task_correct[task] += (pred == targets[task]).sum().item()
                task_total[task] += targets[task].size(0)
            
            step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Batch/Loss', loss.item(), step)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracies = {
            task: task_correct[task] / task_total[task]
            for task in task_correct.keys()
        }
        avg_task_losses = {
            task: val / len(self.train_loader)
            for task, val in task_losses.items()
        }
        
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        for task in accuracies.keys():
            self.writer.add_scalar(f'Accuracy/train/{task}', accuracies[task], epoch)
            self.writer.add_scalar(f'Loss/train/{task}', avg_task_losses[task], epoch)
        
        return avg_loss, accuracies, predictions, targets
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        task_correct = {'pathology': 0, 'region': 0, 'depth': 0}
        task_total = {'pathology': 0, 'region': 0, 'depth': 0}
        task_losses = {'pathology': 0, 'region': 0, 'depth': 0}
        
        # We'll store final predictions across the entire val set for PR curves
        final_preds = { 'pathology': [], 'region': [], 'depth': [] }
        final_targets = { 'pathology': [], 'region': [], 'depth': [] }
        
        for batch_idx, batch in enumerate(
            tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}", position=0, leave=True)
        ):
            images = batch['image'].to(self.device)
            targets = {
                task: batch[task].to(self.device)
                for task in ['pathology', 'region', 'depth']
            }
            
            with torch.no_grad():
                pathology_pred, region_pred, depth_pred = self.model(images)
            
            predictions = {
                'pathology': pathology_pred,
                'region': region_pred,
                'depth': depth_pred
            }
            
            loss, individual_losses = self.criterion(predictions, targets)
            total_loss += loss.item()
            
            for task, t_loss in individual_losses.items():
                task_losses[task] += t_loss.item()
            
            # Accuracy
            for task in ['pathology', 'region', 'depth']:
                pred = torch.argmax(predictions[task], dim=1)
                task_correct[task] += (pred == targets[task]).sum().item()
                task_total[task] += targets[task].size(0)
            
            # Save preds/targets for entire epoch
            for task in ['pathology', 'region', 'depth']:
                final_preds[task].append(predictions[task].cpu())
                final_targets[task].append(targets[task].cpu())
        
        # Convert final_preds, final_targets to single tensors
        for task in ['pathology', 'region', 'depth']:
            final_preds[task] = torch.cat(final_preds[task], dim=0)
            final_targets[task] = torch.cat(final_targets[task], dim=0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracies = {
            task: task_correct[task] / task_total[task]
            for task in task_correct.keys()
        }
        avg_task_losses = {
            task: val / len(self.val_loader)
            for task, val in task_losses.items()
        }
        
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        for task in accuracies.keys():
            self.writer.add_scalar(f'Accuracy/val/{task}', accuracies[task], epoch)
            self.writer.add_scalar(f'Loss/val/{task}', avg_task_losses[task], epoch)
        
        return avg_loss, accuracies, final_preds, final_targets
    
    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Log model graph (try-catch if it doesn't support dict outputs)
        sample_input = next(iter(self.train_loader))['image'].to(self.device)
        try:
            self.writer.add_graph(self.model, sample_input, use_strict_trace=False)
        except Exception as e:
            print(f"Warning: Could not add model graph to TensorBoard: {str(e)}")
        
        for epoch in range(self.num_epochs):
            # Training
            train_loss, train_acc, train_preds, train_targets = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate(epoch)
            
            # Log confusion matrices every 5 epochs
            if epoch % 5 == 0:
                self.log_confusion_matrices(epoch)
            
                self.log_model_weights_and_grads(epoch)
                self.log_pr_curves(val_preds, val_targets, epoch, prefix='val_')
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            for task in train_acc.keys():
                if task not in self.history['train_acc']:
                    self.history['train_acc'][task] = []
                    self.history['val_acc'][task] = []
                self.history['train_acc'][task].append(train_acc[task])
                self.history['val_acc'][task].append(val_acc[task])
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            for task in train_acc.keys():
                print(f"{task.capitalize()} - Train Acc: {train_acc[task]:.4f}, "
                      f"Val Acc: {val_acc[task]:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("\nEarly stopping triggered")
                    break
        
        self.writer.close()
        return self.history