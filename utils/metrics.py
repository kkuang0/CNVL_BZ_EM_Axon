import torch
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
def test_model(model, test_loader, device, 
               pathology_num_classes=2, 
               region_num_classes=3, 
               depth_num_classes=2):
    """
    Evaluate a multi-task model on a test_loader and print extensive metrics.
    
    Args:
        model: PyTorch model returning (pathology_pred, region_pred, depth_pred)
        test_loader: DataLoader for the test dataset
        device: 'cpu' or 'cuda'
        pathology_num_classes: number of classes for pathology
        region_num_classes: number of classes for region
        depth_num_classes: number of classes for depth
    """
    
    model.eval()
    
    from sklearn.metrics import accuracy_score
    
    # We'll store predictions and labels for each task
    all_pathology_preds = []
    all_pathology_labels = []
    
    all_region_preds = []
    all_region_labels = []
    
    all_depth_preds = []
    all_depth_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            pathology_labels = batch['pathology'].to(device)
            region_labels    = batch['region'].to(device)
            depth_labels     = batch['depth'].to(device)
            
            # Forward pass
            pathology_pred, region_pred, depth_pred = model(images)
            
            # Take argmax across classes to get predicted labels
            pathology_pred_labels = torch.argmax(pathology_pred, dim=1)
            region_pred_labels    = torch.argmax(region_pred,    dim=1)
            depth_pred_labels     = torch.argmax(depth_pred,     dim=1)
            
            # Store predictions and ground truth on CPU
            all_pathology_preds.extend(pathology_pred_labels.cpu().numpy())
            all_pathology_labels.extend(pathology_labels.cpu().numpy())
            
            all_region_preds.extend(region_pred_labels.cpu().numpy())
            all_region_labels.extend(region_labels.cpu().numpy())
            
            all_depth_preds.extend(depth_pred_labels.cpu().numpy())
            all_depth_labels.extend(depth_labels.cpu().numpy())
    
    # Convert to NumPy arrays
    all_pathology_preds = np.array(all_pathology_preds)
    all_pathology_labels = np.array(all_pathology_labels)
    
    all_region_preds = np.array(all_region_preds)
    all_region_labels = np.array(all_region_labels)
    
    all_depth_preds = np.array(all_depth_preds)
    all_depth_labels = np.array(all_depth_labels)
    
    # Calculate accuracies
    pathology_acc = accuracy_score(all_pathology_labels, all_pathology_preds)
    region_acc    = accuracy_score(all_region_labels,    all_region_preds)
    depth_acc     = accuracy_score(all_depth_labels,     all_depth_preds)
    
    # Confusion matrices
    cm_pathology = confusion_matrix(all_pathology_labels, all_pathology_preds)
    cm_region    = confusion_matrix(all_region_labels,    all_region_preds)
    cm_depth     = confusion_matrix(all_depth_labels,     all_depth_preds)
    
    # Classification reports
    report_pathology = classification_report(
        all_pathology_labels, all_pathology_preds, 
        labels=range(pathology_num_classes),
        target_names=[f"class_{i}" for i in range(pathology_num_classes)], 
        zero_division=0
    )
    report_region = classification_report(
        all_region_labels, all_region_preds,
        labels=range(region_num_classes),
        target_names=[f"class_{i}" for i in range(region_num_classes)],
        zero_division=0
    )
    report_depth = classification_report(
        all_depth_labels, all_depth_preds,
        labels=range(depth_num_classes),
        target_names=[f"class_{i}" for i in range(depth_num_classes)],
        zero_division=0
    )
    
    # Print results
    print("===== TEST RESULTS =====")
    
    print("pathology Results")
    print(f"  Accuracy: {pathology_acc:.4f}")
    print("  Confusion Matrix:\n", cm_pathology)
    print("  Classification Report:\n", report_pathology)
    
    print("\nRegion Results")
    print(f"  Accuracy: {region_acc:.4f}")
    print("  Confusion Matrix:\n", cm_region)
    print("  Classification Report:\n", report_region)
    
    print("\nDepth Results")
    print(f"  Accuracy: {depth_acc:.4f}")
    print("  Confusion Matrix:\n", cm_depth)
    print("  Classification Report:\n", report_depth)
    
    return {
        'pathology_acc': pathology_acc,
        'region_acc': region_acc,
        'depth_acc': depth_acc,
        'cm_pathology': cm_pathology,
        'cm_region': cm_region,
        'cm_depth': cm_depth,
        'report_pathology': report_pathology,
        'report_region': report_region,
        'report_depth': report_depth
    }