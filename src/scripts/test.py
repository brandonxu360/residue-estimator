import argparse
import torch
from torch.utils.data import DataLoader
from src.data_loader import MultiTaskSegmentationDataset, find_multitask_image_mask_pairs, split_dataset
from src.models.unet.multitask_unet import MultiTaskUNet
from pathlib import Path
import os

def calculate_metrics(pred, target):
    pred_flat = pred.view(-1).cpu()
    target_flat = target.view(-1).cpu()

    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()

    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    return accuracy, precision, recall, iou, dice

def main():
    parser = argparse.ArgumentParser(description='Test Multi-task UNet')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    data_root = Path(args.data_root)
    triples = find_multitask_image_mask_pairs(data_root)
    _, _, test_triples = split_dataset(triples)
    test_dataset = MultiTaskSegmentationDataset(test_triples, resize=(512, 512))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = MultiTaskUNet(n_channels=3, base_channels=64).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Accumulate metrics
    metrics_res = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'iou': 0.0, 'dice': 0.0}
    metrics_sun = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'iou': 0.0, 'dice': 0.0}
    total = 0

    with torch.no_grad():
        for images, residue_masks, sunlit_masks in test_loader:
            images = images.to(device)
            residue_masks = residue_masks.unsqueeze(1).float().to(device)
            sunlit_masks = sunlit_masks.unsqueeze(1).float().to(device)

            res_out, sun_out = model(images)
            res_pred = (torch.sigmoid(res_out) > 0.5).float()
            sun_pred = (torch.sigmoid(sun_out) > 0.5).float()

            # Calculate metrics
            for pred, target, metrics in zip([res_pred, sun_pred], [residue_masks, sunlit_masks], [metrics_res, metrics_sun]):
                acc, pre, rec, iou, dice = calculate_metrics(pred, target)
                metrics['accuracy'] += acc * images.size(0)
                metrics['precision'] += pre * images.size(0)
                metrics['recall'] += rec * images.size(0)
                metrics['iou'] += iou * images.size(0)
                metrics['dice'] += dice * images.size(0)
            total += images.size(0)

    # Print results
    for name, metrics in zip(['Residue', 'Sunlit'], [metrics_res, metrics_sun]):
        print(f"\n{name} Metrics:")
        print(f"Accuracy: {metrics['accuracy']/total:.4f}")
        print(f"Precision: {metrics['precision']/total:.4f}")
        print(f"Recall: {metrics['recall']/total:.4f}")
        print(f"IoU: {metrics['iou']/total:.4f}")
        print(f"Dice: {metrics['dice']/total:.4f}")

if __name__ == '__main__':
    main()