import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from src.dataset import GCPDataset
from src.model import GCPModel


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset = GCPDataset(
        image_dir=os.path.join(BASE_DIR, "data/train_dataset"),
        label_path=os.path.join(BASE_DIR, "data/curated_gcp_marks.json"),
        img_size=256
    )

    # train / validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = GCPModel().to(device)

    coord_loss_fn = nn.MSELoss()

    # class imbalance handling
    class_counts = torch.tensor([105, 892, 1], dtype=torch.float)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    shape_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 8

    best_val_loss = float("inf")

    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for i, (imgs, coords, shapes) in enumerate(train_loader):

            imgs = imgs.to(device)
            coords = coords.to(device)
            shapes = shapes.to(device)

            pred_coords, pred_shape = model(imgs)

            coord_loss = coord_loss_fn(pred_coords, coords)
            shape_loss = shape_loss_fn(pred_shape, shapes)

            loss = coord_loss + shape_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch {epoch + 1} Batch {i}/{len(train_loader)} Loss {loss.item():.4f}")

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0

        with torch.no_grad():

            for imgs, coords, shapes in val_loader:

                imgs = imgs.to(device)
                coords = coords.to(device)
                shapes = shapes.to(device)

                pred_coords, pred_shape = model(imgs)

                coord_loss = coord_loss_fn(pred_coords, coords)
                shape_loss = shape_loss_fn(pred_shape, shapes)

                loss = coord_loss + shape_loss

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

            torch.save(
                model.state_dict(),
                os.path.join(BASE_DIR, "outputs/model.pth")
            )

            print("Saved best model")


if __name__ == "__main__":
    train()