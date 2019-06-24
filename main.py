import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
from model import NADE

def train(train_loader, loss_function, optimizer, model, device):
    model.train()
    total_loss = 0.0
    for i, (imgs, _) in enumerate(train_loader):
        
        optimizer.zero_grad()
        # preprocess to binary
        inputs = imgs.view(imgs.size(0), -1).gt(0.).float().to(device)
        x_hat = model(inputs)
        loss = loss_function(x_hat, inputs)
        loss.backward()
        optimizer.step()
        
        # record
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f"\t[{i*imgs.size(0)/len(train_loader.dataset)*100:.2f}%] loss: {loss/imgs.size(0):.4f}")
            
    return total_loss

def test(test_loader, loss_function, model, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, _ in test_loader:
            # preprocess to binary
            inputs = imgs.view(imgs.size(0), -1).gt(0.).float().to(device)
            x_hat = model(inputs)
            loss = loss_function(x_hat, inputs)
            total_loss += loss.item()
            
        print(f"\t[Test Result] loss: {total_loss/len(test_loader.dataset):.4f}")
    return total_loss/len(test_loader.dataset)
        
def draw_sampling(model):
    model.eval()
    x_hat, xs, nll_loss = model.sample(n=1, only_prob=True)
    fig, axes = plt.subplots(1, 2)
    for x, ax in zip([x_hat, xs], axes):
        ax.matshow(x.cpu().detach().squeeze().view(28, 28).numpy())
        ax.axis('off')
    plt.title(f"\t[Random Sampling] NLL loss: {nll_loss:.4f}", fontdict={"fontsize": 16})
    plt.show()
    
def non_decreasing(L):
    """for early stopping"""
    return all(x<=y for x, y in zip(L, L[1:]))

def main(draw=False):
    data_path = Path(".").absolute().parent / "data"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_step = 25
    batch_size = 256

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.MNIST(root=str(data_path), train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.MNIST(root=str(data_path), train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model = NADE(input_dim=784, hidden_dim=500).to(device)
    loss_function = nn.BCELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)
    
    # start main
    train_losses = []
    test_losses = []
    best_loss = 99999999
    wait = 0
    for step in range(n_step):
        print(f"Running Step: [{step+1}/{n_step}]")
        train_loss = train(train_loader, loss_function, optimizer, model, device)
        test_loss = test(test_loader, loss_function, model, device)
        scheduler.step()
        # sampling
        if draw:
            draw_sampling(model)
        # record
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss <= best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "nade-binary.pt")
            print(f"\t[Model Saved]")
            if (step >= 2) and (wait <= 3) and (non_decreasing(test_losses[-3:])):
                wait += 1
            elif wait > 3:
                print(f"[Early Stopped]")
                break
            else:
                continue
                
                
if __name__ == "__main__":
    main(draw=False)