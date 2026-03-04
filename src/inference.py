import os
import json
import cv2
import torch

from src.model import GCPModel


def run_inference():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    test_dir = os.path.join(BASE_DIR, "data/test_dataset")
    model_path = os.path.join(BASE_DIR, "outputs/model.pth")

    model = GCPModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    shape_map = {0: "Cross", 1: "Square", 2: "L-Shaped"}

    predictions = {}

    for root, dirs, files in os.walk(test_dir):

        for file in files:

            if not file.lower().endswith(".jpg"):
                continue

            full_path = os.path.join(root, file)

            rel_path = os.path.relpath(full_path, test_dir)

            img = cv2.imread(full_path)
            h, w = img.shape[:2]

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))

            img = img / 255.0
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            img = img.unsqueeze(0).to(device)

            with torch.no_grad():
                coords, shape_logits = model(img)

            coords = coords.cpu().numpy()[0]
            shape = torch.argmax(shape_logits, dim=1).item()

            x = float(coords[0] * w)
            y = float(coords[1] * h)

            predictions[rel_path] = {
                "mark": {
                    "x": x,
                    "y": y
                },
                "verified_shape": shape_map[shape]
            }

    output_path = os.path.join(BASE_DIR, "outputs/predictions.json")

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)

    print("Predictions saved to:", output_path)


if __name__ == "__main__":
    run_inference()