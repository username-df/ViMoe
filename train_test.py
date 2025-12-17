import torch
from transformers import get_cosine_schedule_with_warmup
from torch import nn
from ViT import ViT
import matplotlib.pyplot as plt
from dataset import train_data, test_data
from torchmetrics.classification import MulticlassAccuracy

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
model = ViT(img_size=224, patch_size=32, embed_dim=512, num_heads=8, num_blocks=12, num_classes=2, bn=train_data.batch_size)
model = model.to(device) 

epochs = 25
LR = 1e-3
optimizer = torch.optim.AdamW(params=model.parameters(), betas=(0.9, 0.999), weight_decay=0.1)
lossfn = nn.CrossEntropyLoss()

schedule = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=10000,
    num_training_steps=epochs*len(train_data)
)

accfn = MulticlassAccuracy(num_classes=2).to(device)

def loss_curves(epochs, train, test):
    xs = range(1, epochs+1)
    plt.plot(xs, train, 'b-', label="Train Loss")
    plt.plot(xs, test, 'b--', label="Test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

train_hst, test_hst = [], []

for epoch in range(epochs):
    #------------- Train ----------------
    train_loss, train_acc = 0, 0
    
    for X,y in train_data:
        X, y = X.to(device), y.to(device)

        model.train()

        trainprd = model(X)
        
        loss = lossfn(trainprd, y)

        train_loss += loss
        train_acc += accfn(trainprd, y).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()

    train_loss /= len(train_data)
    train_acc /= len(train_data)

    # --------------- Test ----------------------
    test_loss, test_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for X,y in test_data:
            X, y = X.to(device), y.to(device)

            testprd = model(X)

            test_loss += lossfn(testprd, y)
            test_acc += accfn(testprd, y).item()

        test_loss /= len(test_data)
        test_acc /= len(test_data)

    model.save(file_name=f"saved{epoch}.pth")
    train_hst.append(train_loss.item())
    test_hst.append(test_loss.item())

    print(f"---------------- Epoch {epoch} ---------------------")

    print(f"Train Loss: {train_loss:.2f} | Train Accuracy: {train_acc*100:.2f}%\n")
    
    print(f"Test loss: {test_loss:.2f} | Test Accuracy: {test_acc*100:.2f}%\n")

    print()

loss_curves(epochs, train_hst, test_hst)