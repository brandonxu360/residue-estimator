import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_loader import MultiTaskSegmentationDataset, find_multitask_image_mask_pairs, split_dataset
from src.models.unet.multitask_unet import MultiTaskUNet
from pathlib import Path
import os

def main():
    parser = argparse.ArgumentParser(description='Train Multi-task UNet')
    parser.add_argument('--job_id', type=str, default=None, help='Slurm job ID for model naming')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='experiments/unet', help='Directory to save model checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and split dataset
    data_root = Path(args.data_root)
    triples = find_multitask_image_mask_pairs(data_root)
    train_triples, val_triples, test_triples = split_dataset(triples)

    # Create datasets and dataloaders
    train_dataset = MultiTaskSegmentationDataset(train_triples, resize=(512, 512))
    val_dataset = MultiTaskSegmentationDataset(val_triples, resize=(512, 512))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, loss, optimizer
    model = MultiTaskUNet(n_channels=3, base_channels=64).to(device)
    criterion_res = nn.BCEWithLogitsLoss()
    criterion_sun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        for images, residue_masks, sunlit_masks in train_loader:
            images = images.to(device)
            residue_masks = residue_masks.unsqueeze(1).float().to(device)
            sunlit_masks = sunlit_masks.unsqueeze(1).float().to(device)

            optimizer.zero_grad()
            res_out, sun_out = model(images)
            loss_res = criterion_res(res_out, residue_masks)
            loss_sun = criterion_sun(sun_out, sunlit_masks)
            total_loss = loss_res + loss_sun
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, residue_masks, sunlit_masks in val_loader:
                images = images.to(device)
                residue_masks = residue_masks.unsqueeze(1).float().to(device)
                sunlit_masks = sunlit_masks.unsqueeze(1).float().to(device)

                res_out, sun_out = model(images)
                val_loss += (criterion_res(res_out, residue_masks) + criterion_sun(sun_out, sunlit_masks)).item() * images.size(0)
            val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_name = f'best_model_{args.job_id}.pth' if args.job_id else 'best_model.pth'
            torch.save(model.state_dict(), os.path.join(args.output_dir, model_name))

    print(f'Training complete. Best model saved to {args.output_dir}/{model_name}')

if __name__ == '__main__':
    main()