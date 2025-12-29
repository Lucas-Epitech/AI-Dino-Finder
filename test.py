from dino_finder.data.dataset import DinoDataset

dataset = DinoDataset("data/raw")

print(f"Nombre d'images: {len(dataset)}")
print(f"Classes trouvées: {dataset.class_to_idx}")
print(f"\nPremiers échantillons:")
for i in range(min(30, len(dataset))):
    image_path, label = dataset.samples[i]
    print(f"Image path: {image_path.name}, Label index: {label}")

if len(dataset) > 0:
    print(f"\n✅ Test de chargement d'une image:")
    image, label = dataset[0]
    print(f"   Taille: {image.size}")
    print(f"   Mode: {image.mode}")
    print(f"   Label: {label}")
else:
    print("❌ Aucune image trouvée!")
