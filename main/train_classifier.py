"""
train_classifier.py - Train MobileNetV2 binary classifier for desk occupancy

This script trains a SINGLE shared model for ALL desks using ROI-based
image classification.

IMPORTANT: Why a shared model?
- Fixed cameras have consistent viewpoints per angle
- "Empty" = similar background across all desks
- "Occupied" = person silhouette (similar pattern across desks)
- More training data = better generalization
- Model learns "occupied" pattern, not desk identity

Dataset structure:
    dataset/crops/
    ├── desk_a/
    │   ├── empty/     ← Label: 0
    │   └── occupied/  ← Label: 1
    ├── desk_b/
    │   ├── empty/
    │   └── occupied/
    └── desk_c/
        ├── empty/
        └── occupied/

The model learns to classify ANY desk crop as empty (0) or occupied (1).

Usage:
    python train_classifier.py
    python train_classifier.py --epochs 30 --batch-size 16 --lr 0.0001
"""

import os
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split


def extract_timestamp_from_filename(filename):
    """
    Extract timestamp from filename for temporal grouping.

    Filename format: 2026.05.08_02_24_14_3772_esp32.jpg
    Parts: YYYY.MM.DD_HH_MM_SS_MSS_xxx.jpg

    Returns:
        tuple: (session_key, full_datetime)
        - session_key: "YYYY.MM.DD_HH" for grouping by hour
        - full_datetime: datetime object for sorting
    """
    import re
    match = re.match(r'(\d{4})\.(\d{2})\.(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d+)_', filename)
    if match:
        year, month, day, hour, minute, second, ms = match.groups()
        dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        session_key = f"{year}.{month}.{day}_{hour}"
        return session_key, dt
    return None, None

IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 20

LEARNING_RATE = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_image_files(directory):
    """
    Get all image files from directory with case-insensitive extension matching.
    Handles .jpg, .JPG, .jpeg, .JPEG, .png, .PNG extensions.
    """
    if not directory.exists():
        return []

    extensions = {'.jpg', '.jpeg', '.png'}
    files = []
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in extensions:
            files.append(item)
    return sorted(files)


class SharedDeskDataset(Dataset):
    """
    Dataset that loads images from ALL desks and merges them into ONE dataset.

    This is intentional - we want ONE binary classifier for all desks.
    The model learns "empty vs occupied" pattern, NOT desk identity.

    Directory structure expected:
        dataset/crops/{desk}/empty/
        dataset/crops/{desk}/occupied/

    Also tracks temporal information for temporal splitting.
    """

    def __init__(self, crops_dir, transform=None, include_desks=None, raw_images_dir=None):
        """
        Args:
            crops_dir: Base directory containing desk subdirectories
            transform: Optional transform to apply to images
            include_desks: List of desk names to include (None = all)
            raw_images_dir: Optional path to raw images for temporal grouping
        """
        self.crops_dir = Path(crops_dir)
        self.transform = transform
        self.samples = []
        self.sessions = {}  
        self.class_to_idx = {'empty': 0, 'occupied': 1}
        self.idx_to_class = {0: 'empty', 1: 'occupied'}

        
        stats = {
            'total': 0,
            'empty': 0,
            'occupied': 0,
            'per_desk': {},
            'sessions': []
        }

       
        if include_desks is None:
            desk_dirs = [d for d in self.crops_dir.iterdir()
                        if d.is_dir() and d.name.startswith('desk_')]
        else:
            desk_dirs = [self.crops_dir / d for d in include_desks
                        if (self.crops_dir / d).exists()]

        desk_dirs = sorted(desk_dirs)

        for desk_dir in desk_dirs:
            desk_name = desk_dir.name
            stats['per_desk'][desk_name] = {'empty': 0, 'occupied': 0}

            for class_name in ['empty', 'occupied']:
                class_dir = desk_dir / class_name
                if class_dir.exists():
                    img_files = get_image_files(class_dir)
                    for img_path in img_files:
                        filename = img_path.name
                        session_key, _ = extract_timestamp_from_filename(filename)

                        self.samples.append((
                            str(img_path),
                            self.class_to_idx[class_name],
                            desk_name,  
                            session_key  
                        ))

                       
                        if session_key:
                            if session_key not in self.sessions:
                                self.sessions[session_key] = []
                            self.sessions[session_key].append(len(self.samples) - 1)

                        stats[class_name] += 1
                        stats['per_desk'][desk_name][class_name] += 1
                        stats['total'] += 1

        stats['sessions'] = list(self.sessions.keys())
        stats['num_sessions'] = len(self.sessions)

        if len(self.samples) == 0:
            raise ValueError(
                f"No labeled images found. Check:\n"
                f"  {crops_dir}/desk_a/empty/\n"
                f"  {crops_dir}/desk_a/occupied/\n"
                f"  ... etc"
            )

        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"Total samples: {stats['total']}")
        print(f"Empty samples: {stats['empty']} ({100*stats['empty']/stats['total']:.1f}%)")
        print(f"Occupied samples: {stats['occupied']} ({100*stats['occupied']/stats['total']:.1f}%)")
        print(f"\nTemporal split info:")
        print(f"  Number of sessions: {stats['num_sessions']}")
        print(f"  Session key format: YYYY.MM.DD_HH (grouped by hour)")
        print("\nPer desk breakdown:")
        for desk, counts in stats['per_desk'].items():
            total_desk = counts['empty'] + counts['occupied']
            print(f"  {desk}: {total_desk} samples "
                  f"(empty={counts['empty']}, occupied={counts['occupied']})")
        print("=" * 60)

        self.stats = stats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, desk_name, session_key = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class MobileNetV2Classifier(nn.Module):

    def __init__(self, num_classes=1, pretrained=True):
        super(MobileNetV2Classifier, self).__init__()
        try:
            from torchvision.models import MobileNet_V2_Weights
            self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        except ImportError:
            self.model = models.mobilenet_v2(pretrained=pretrained)

        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),  
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc='Training', leave=False):
        images, labels = images.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)

        outputs = outputs.view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.float().to(device)

            outputs = model(images)

           
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), correct / total


def train_classifier(crops_dir='dataset/crops', output_dir='models', epochs=20,
                    batch_size=32, lr=LEARNING_RATE, include_desks=None,
                    temporal_split=True, val_ratio=0.2):
    """
    Train the shared MobileNetV2 classifier.

    Args:
        crops_dir: Base directory containing desk subdirectories
        output_dir: Output directory for trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate (default: 0.0001 for transfer learning)
        include_desks: List of desk names to include (None = all)
        temporal_split: If True, split by sessions (not random). Prevents data leakage.
        val_ratio: Ratio of sessions to use for validation

    TEMPORAL SPLITTING:
        Images from the same capture session (same hour) are grouped together.
        All images from a session go to EITHER train OR val, never both.
        This prevents data leakage from temporally correlated frames.
    """
    print(f"\nUsing device: {DEVICE}")
    print(f"Training on data from: {crops_dir}")
    print(f"Shared model for all desks: {include_desks or 'all desks'}")
    print(f"Learning rate: {lr} (optimized for transfer learning)")
    print(f"Temporal split: {temporal_split}")

    transforms_dict = get_transforms()
    full_dataset = SharedDeskDataset(crops_dir, transform=transforms_dict['train'],
                                     include_desks=include_desks)

    if temporal_split:
        print("\n" + "-" * 60)
        print("USING TEMPORAL SPLIT")
        print("-" * 60)

        sessions = list(full_dataset.sessions.keys())
        print(f"Total sessions: {len(sessions)}")

        random.seed(42)
        shuffled_sessions = sessions.copy()
        random.shuffle(shuffled_sessions)

        num_val_sessions = max(1, int(len(shuffled_sessions) * val_ratio))
        val_sessions = set(shuffled_sessions[:num_val_sessions])
        train_sessions = set(shuffled_sessions[num_val_sessions:])

        print(f"Train sessions: {len(train_sessions)}")
        print(f"Validation sessions: {len(val_sessions)}")

        train_indices = []
        val_indices = []

        for session, indices in full_dataset.sessions.items():
            if session in val_sessions:
                val_indices.extend(indices)
            else:
                train_indices.extend(indices)

        train_samples = [full_dataset.samples[i] for i in train_indices]
        val_samples = [full_dataset.samples[i] for i in val_indices]

        print(f"\nTrain samples: {len(train_samples)}")
        print(f"Validation samples: {len(val_samples)}")

        print(f"\nValidation sessions breakdown:")
        for session in sorted(val_sessions)[:5]:
            print(f"  - {session} ({len(full_dataset.sessions[session])} samples)")
        if len(val_sessions) > 5:
            print(f"  ... and {len(val_sessions) - 5} more sessions")

    else:
        print("\n" + "-" * 60)
        print("WARNING: Using random split (may cause data leakage)")
        print("-" * 60)

        train_samples, val_samples = train_test_split(
            full_dataset.samples,
            test_size=val_ratio,
            random_state=42,
            stratify=[s[1] for s in full_dataset.samples]
        )

        print(f"\nTrain samples: {len(train_samples)}")
        print(f"Validation samples: {len(val_samples)}")

    class TrainDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label, desk_name, session_key = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label

    train_dataset = TrainDataset(train_samples, transforms_dict['train'])
    val_dataset = TrainDataset(val_samples, transforms_dict['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = MobileNetV2Classifier().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    best_val_acc = 0.0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = output_path / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': {
                    'img_size': IMG_SIZE,
                    'classes': ['empty', 'occupied'],
                    'include_desks': include_desks,
                    'shared_model': True  
                }
            }, model_path)
            print(f"*** Saved best model with val_acc: {val_acc:.4f} ***")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_path / 'best_model.pth'}")

    training_info = {
        'timestamp': timestamp,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'best_val_acc': best_val_acc,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'classes': ['empty', 'occupied'],
        'include_desks': include_desks,
        'shared_model': True,
        'dataset_stats': full_dataset.stats,
        'temporal_split': {
            'enabled': temporal_split,
            'num_train_sessions': len(train_sessions) if temporal_split else None,
            'num_val_sessions': len(val_sessions) if temporal_split else None,
            'val_sessions': sorted(val_sessions) if temporal_split else None
        }
    }
    with open(output_path / 'training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)

    return model, best_val_acc


def main():
    parser = argparse.ArgumentParser(
        description='Train shared MobileNetV2 desk occupancy classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_classifier.py                    # Train with temporal split (recommended)
  python train_classifier.py --no-temporal      # Disable temporal split (random split)
  python train_classifier.py --epochs 30        # More epochs
  python train_classifier.py --batch-size 16    # Smaller batch
  python train_classifier.py --lr 0.0001        # Lower LR for transfer learning

Recommended command:
  python train_classifier.py --epochs 20 --batch-size 16 --lr 0.0001

Note:
  - Temporal split is ENABLED by default (prevents data leakage)
  - This trains ONE shared model for all desks
  - The model learns "empty vs occupied" pattern, not desk identity
        """
    )
    parser.add_argument('--crops-dir', default='dataset/crops',
                        help='Base directory with desk subdirectories')
    parser.add_argument('--output-dir', default='models',
                        help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--include-desks', nargs='+',
                        help='Desk names to include (e.g., desk_a desk_b)')
    parser.add_argument('--temporal-split', action='store_true', default=True,
                        help='Split by session/time instead of random (recommended, default: True)')
    parser.add_argument('--no-temporal', action='store_true',
                        help='Disable temporal split (use random split instead)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Ratio of sessions to use for validation (default: 0.2)')

    args = parser.parse_args()

    temporal_split = not args.no_temporal

    train_classifier(
        crops_dir=args.crops_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        include_desks=args.include_desks,
        temporal_split=temporal_split,
        val_ratio=args.val_ratio
    )


if __name__ == '__main__':
    main()
