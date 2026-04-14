# scripts/train.py
import torch, os
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT/"data/processed"
BATCH_SIZE = 32
# ⭐️ Change 1: Epochs badhayein
EPOCHS = 25 
# ⭐️ Change 2: Learning rate kam karein (fine-tuning ke liye)
LR = 0.0001 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("📂 DATA DIR:", DATA_DIR)

    # ⭐️ Change 3: Data Augmentation ko aur powerful banayein
    transform = {
        "train": transforms.Compose([
            # Resize ki jagah RandomResizedCrop (zoom/crop variation)
            transforms.RandomResizedCrop((224,224)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # Lighting variations
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # datasets
    image_datasets = {x: datasets.ImageFolder(DATA_DIR/x, transform=transform[x])
                      for x in ["train","val","test"]}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
                   for x in ["train","val","test"]}

    class_names = image_datasets["train"].classes
    print("📦 classes:", class_names)

    # ⭐️ Change 4: ResNet18 ki jagah ResNet50 (behtar model)
    model = models.resnet50(pretrained=True) 
    
    # ⭐️ Change 5: Freeze karne wala code HATA diya hai. 
    # Hum saari layers ko fine-tune karenge.
    
    # Aakhiri layer ko badlein (ResNet50 ke hisab se)
    num_ftrs = model.fc.in_features # Yeh 2048 dega (ResNet18 512 deta hai)
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # ⭐️ Change 6: Optimizer ko saari layers train karne ko bolein
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # training loop
    for epoch in range(EPOCHS):
        print(f"\n🔄 Epoch {epoch+1}/{EPOCHS}")
        for phase in ["train","val"]:
            if phase == "train": model.train()
            else: model.eval()

            running_loss, running_corrects = 0.0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=="train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, 1)
                    if phase=="train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

    # save model
    MODEL_PATH = PROJECT/"models/waste_classifier.pth"
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Model saved at:", MODEL_PATH)

if __name__ == "__main__":
    main()