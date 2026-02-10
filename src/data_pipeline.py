"""
Advanced Data Pipeline for Glaucoma Detection
Features:
- Advanced augmentation with MixUp, CutMix, and Mosaic
- Retinal-specific preprocessing (CLAHE, Ben Graham, illumination normalization)
- Class-aware sampling for imbalanced data
- Multi-scale preprocessing per model
- Cross-validation support
- Test-Time Augmentation (TTA)
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.core.composition import Compose

from src.data_preprocessing import (
    apply_preprocessing_pipeline, MODEL_PREPROCESSING,
    remove_black_border, normalize_illumination, apply_clahe
)


def get_augmentation_pipeline(image_size, mode='train', strength='medium'):
    if mode == 'train':
        if strength == 'light':
            return Compose([
                A.RandomResizedCrop(size=(image_size[0], image_size[1]),
                                   scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        elif strength == 'heavy':
            return Compose([
                A.RandomResizedCrop(size=(image_size[0], image_size[1]),
                                   scale=(0.7, 1.0), ratio=(0.85, 1.15)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=20, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.4),
                A.OneOf([
                    A.GaussianBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.CLAHE(clip_limit=3.0, p=1.0),
                    A.Equalize(p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                ], p=0.4),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08, p=0.5),
                A.GaussNoise(std_range=(0.01, 0.06), p=0.25),
                A.CoarseDropout(
                    num_holes_range=(1, 10),
                    hole_height_range=(image_size[0]//40, image_size[0]//12),
                    hole_width_range=(image_size[1]//40, image_size[1]//12),
                    p=0.3
                ),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
                A.ElasticTransform(alpha=50, sigma=50 * 0.05, p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            return Compose([
                A.RandomResizedCrop(size=(image_size[0], image_size[1]),
                                   scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2.0, p=1.0),
                    A.Equalize(p=1.0),
                ], p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(image_size[0]//32, image_size[0]//16),
                    hole_width_range=(image_size[1]//32, image_size[1]//16),
                    p=0.2
                ),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
    else:
        return Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


def get_tta_augmentations(image_size):
    return [
        Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]),
        Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]),
        Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]),
        Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Rotate(limit=(90, 90), p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]),
        Compose([
            A.RandomResizedCrop(size=(image_size[0], image_size[1]), scale=(0.9, 1.0)),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]),
    ]


class GlaucomaDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, image_size=(224, 224),
                 augment=True, shuffle=True, mixup_alpha=0.2, cutmix_alpha=0.2,
                 preprocessing_pipeline='standard', oversample_minority=False,
                 aug_strength='medium'):
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.shuffle = shuffle
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.preprocessing_pipeline = preprocessing_pipeline

        mode = 'train' if augment else 'val'
        self.aug_pipeline = get_augmentation_pipeline(image_size, mode, strength=aug_strength)

        if oversample_minority and augment:
            self.indices = self._create_oversampled_indices()
        else:
            self.indices = np.arange(len(self.image_paths))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def _create_oversampled_indices(self):
        pos_indices = np.where(self.labels == 1)[0]
        neg_indices = np.where(self.labels == 0)[0]
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return np.arange(len(self.image_paths))
        ratio = len(neg_indices) // max(len(pos_indices), 1)
        oversampled_pos = np.tile(pos_indices, ratio)
        np.random.shuffle(oversampled_pos)
        oversampled_pos = oversampled_pos[:len(neg_indices)]
        all_indices = np.concatenate([neg_indices, oversampled_pos])
        np.random.shuffle(all_indices)
        return all_indices

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_images = []
        batch_labels = []

        for i in batch_indices:
            img = self._load_and_preprocess(self.image_paths[i])
            if img is not None:
                augmented = self.aug_pipeline(image=img)
                batch_images.append(augmented['image'])
                batch_labels.append(self.labels[i])

        if len(batch_images) == 0:
            return np.zeros((1, *self.image_size, 3)), np.zeros((1, 1))

        X = np.array(batch_images)
        y = np.array(batch_labels).reshape(-1, 1)

        if self.augment and len(X) >= 2 and np.random.random() < 0.3:
            if np.random.random() < 0.5:
                X, y = self._mixup(X, y)
            else:
                X, y = self._cutmix(X, y)

        return X, y

    def _load_and_preprocess(self, path):
        try:
            img = Image.open(path).convert('RGB')
            img = np.array(img)
            img = apply_preprocessing_pipeline(img, self.preprocessing_pipeline)
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            return img
        except Exception as e:
            return None

    def _mixup(self, X, y):
        if len(X) < 2:
            return X, y
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        lam = max(lam, 1 - lam)
        indices = np.random.permutation(len(X))
        X_mixed = lam * X + (1 - lam) * X[indices]
        y_mixed = lam * y + (1 - lam) * y[indices]
        return X_mixed, y_mixed

    def _cutmix(self, X, y):
        if len(X) < 2:
            return X, y
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        indices = np.random.permutation(len(X))
        h, w = X.shape[1], X.shape[2]
        cut_h = int(h * np.sqrt(1 - lam))
        cut_w = int(w * np.sqrt(1 - lam))
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        X_cut = X.copy()
        X_cut[:, y1:y2, x1:x2, :] = X[indices, y1:y2, x1:x2, :]
        lam_adjusted = 1 - ((x2 - x1) * (y2 - y1)) / (h * w)
        y_mixed = lam_adjusted * y + (1 - lam_adjusted) * y[indices]
        return X_cut, y_mixed

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def prepare_rfmid_dataset(data_dir, label_column='ODC'):
    train_dir = os.path.join(data_dir, 'Training_set')
    val_dir = os.path.join(data_dir, 'Validation_set')
    test_dir = os.path.join(data_dir, 'Test_set')

    train_csv = os.path.join(train_dir, 'RFMiD_Training_Labels.csv')
    val_csv = os.path.join(val_dir, 'RFMiD_Validation_Labels.csv')
    test_csv = os.path.join(test_dir, 'RFMiD_Testing_Labels.csv')

    def load_split(csv_path, image_dir):
        df = pd.read_csv(csv_path)
        image_paths = []
        labels = []
        for _, row in df.iterrows():
            img_id = row['ID']
            img_path = os.path.join(image_dir, f"{img_id}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(image_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(int(row[label_column]))
        return image_paths, labels

    train_paths, train_labels = load_split(train_csv, train_dir)
    val_paths, val_labels = load_split(val_csv, val_dir)
    test_paths, test_labels = load_split(test_csv, test_dir)

    print(f"Training set: {len(train_paths)} images")
    print(f"  - Normal: {sum(1 for l in train_labels if l == 0)}")
    print(f"  - Glaucoma: {sum(1 for l in train_labels if l == 1)}")
    print(f"Validation set: {len(val_paths)} images")
    print(f"Test set: {len(test_paths)} images")

    return {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }


def create_tf_dataset(image_paths, labels, batch_size=32, image_size=(224, 224),
                      augment=True, shuffle=True, model_name=None,
                      oversample_minority=False, aug_strength='medium'):
    pipeline = MODEL_PREPROCESSING.get(model_name, 'standard') if model_name else 'standard'

    generator = GlaucomaDataGenerator(
        image_paths=image_paths,
        labels=labels,
        batch_size=batch_size,
        image_size=image_size,
        augment=augment,
        shuffle=shuffle,
        preprocessing_pipeline=pipeline,
        oversample_minority=oversample_minority,
        aug_strength=aug_strength,
    )

    return generator


def get_class_weights(labels):
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = {}
    for cls, count in zip(unique, counts):
        weights[cls] = total / (len(unique) * count)
    return weights


def create_kfold_splits(image_paths, labels, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    for train_idx, val_idx in skf.split(image_paths, labels):
        train_paths = [image_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        splits.append({
            'train': (train_paths, train_labels),
            'val': (val_paths, val_labels)
        })
    return splits


def predict_with_tta(model, image, image_size=(224, 224)):
    tta_augs = get_tta_augmentations(image_size)
    predictions = []
    for aug in tta_augs:
        augmented = aug(image=image)
        img_tensor = np.expand_dims(augmented['image'], axis=0)
        pred = model.predict(img_tensor, verbose=0)
        predictions.append(float(pred[0][0]))
    return float(np.mean(predictions))
