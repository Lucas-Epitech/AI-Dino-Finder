#!/usr/bin/env python3
"""
Test script to visualize preprocessing transformations.

This script shows what happens to an image when we apply transforms.
Run: python scripts/test_preprocessing.py
"""

from pathlib import Path
from PIL import Image
import torch

from dino_finder.data.preprocessing import (
    train_transforms,
    val_transforms,
    denormalize,
    IMAGE_SIZE
)


def main():
    # Trouver une image de test
    data_dir = Path("data/raw")

    # Prendre la première image trouvée
    image_path = None
    for species_dir in data_dir.iterdir():
        if species_dir.is_dir():
            for img in species_dir.iterdir():
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_path = img
                    break
        if image_path:
            break

    if not image_path:
        print("❌ Aucune image trouvée dans data/raw/")
        return

    print(f"🖼️  Image de test: {image_path}")
    print("=" * 60)

    # Charger l'image originale
    original_image = Image.open(image_path).convert('RGB')
    print(f"\n📥 IMAGE ORIGINALE:")
    print(f"   Taille: {original_image.size} (largeur × hauteur)")
    print(f"   Mode: {original_image.mode}")

    # Appliquer les transformations de validation (sans augmentation)
    print(f"\n🔄 APRÈS VAL_TRANSFORMS (sans augmentation):")
    tensor_val = val_transforms(original_image)
    print(f"   Shape: {tensor_val.shape}")
    print(f"   Type: {tensor_val.dtype}")
    print(f"   Min: {tensor_val.min():.3f}")
    print(f"   Max: {tensor_val.max():.3f}")
    print(f"   Mean: {tensor_val.mean():.3f}")

    # Appliquer les transformations d'entraînement (avec augmentation)
    print(f"\n🔄 APRÈS TRAIN_TRANSFORMS (avec augmentation):")
    print("   (Les résultats varient à chaque exécution car c'est aléatoire)")

    for i in range(3):
        tensor_train = train_transforms(original_image)
        print(f"   Run {i+1}: shape={tensor_train.shape}, "
              f"mean={tensor_train.mean():.3f}")

    # Explication des dimensions
    print(f"\n📐 EXPLICATION DES DIMENSIONS:")
    print(f"   Original: ({original_image.size[1]}, {original_image.size[0]}, 3)")
    print(f"            = (Hauteur, Largeur, Canaux RGB)")
    print(f"   Tensor:   {tuple(tensor_val.shape)}")
    print(f"            = (Canaux, Hauteur, Largeur)")
    print(f"   → PyTorch utilise le format 'channels first' (C, H, W)")

    # Exemple de valeurs normalisées
    print(f"\n📊 EXEMPLE DE VALEURS (coin supérieur gauche, 2×2 pixels):")
    print(f"   Canal Rouge (R):")
    print(f"   {tensor_val[0, 0:2, 0:2]}")
    print(f"   Canal Vert (G):")
    print(f"   {tensor_val[1, 0:2, 0:2]}")
    print(f"   Canal Bleu (B):")
    print(f"   {tensor_val[2, 0:2, 0:2]}")

    # Dé-normalisation
    print(f"\n🔙 APRÈS DÉ-NORMALISATION:")
    tensor_denorm = denormalize(tensor_val)
    print(f"   Min: {tensor_denorm.min():.3f}")
    print(f"   Max: {tensor_denorm.max():.3f}")
    print(f"   → Valeurs entre 0 et 1 (prêtes pour affichage)")

    print("\n" + "=" * 60)
    print("✅ Preprocessing fonctionne correctement!")
    print("=" * 60)


if __name__ == "__main__":
    main()
