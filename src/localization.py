import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset_utils import numeric_key
from src.models import CNNModel


DEFAULT_SUBJECT_DIR = os.path.join(
    "data", "topomap_sequence", "test", "stroke", "stroke_05"
)
DEFAULT_MODEL_PATH = os.path.join("outputs", "trained_models", "cnn", "cnn_model.pth")
DEFAULT_OUTPUT_PATH = os.path.join("outputs", "locate", "gradcam_subject_stroke_05.png")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate(self, x):
        self.model.zero_grad()
        output = self.model(x)
        score = output.view(-1)[0]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, score.detach()


def load_topomap(path):
    x = np.load(path).astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x


def load_subject_segments(subject_dir, max_segments=None):
    files = [
        os.path.join(subject_dir, file)
        for file in os.listdir(subject_dir)
        if file.endswith(".npy")
    ]
    files = sorted(files, key=numeric_key)

    if max_segments is not None:
        files = files[:max_segments]

    if not files:
        raise ValueError(f"No .npy segment files found in {subject_dir}")

    return files


def estimate_region(cam):
    height, width = cam.shape
    _, scalp_mask = apply_scalp_mask(cam)
    masked_cam = cam.copy()
    masked_cam[~scalp_mask] = np.nan

    if np.all(np.isnan(masked_cam)):
        return "Unknown"

    peak_y, peak_x = np.unravel_index(np.nanargmax(masked_cam), masked_cam.shape)
    radius = 4
    y_min = max(0, peak_y - radius)
    y_max = min(height, peak_y + radius + 1)
    x_min = max(0, peak_x - radius)
    x_max = min(width, peak_x + radius + 1)

    local_patch = masked_cam[y_min:y_max, x_min:x_max]
    local_ys, local_xs = np.where(~np.isnan(local_patch))

    if len(local_xs) == 0:
        x_mean = peak_x
        y_mean = peak_y
    else:
        weights = local_patch[local_ys, local_xs]
        x_mean = np.average(local_xs + x_min, weights=weights)
        y_mean = np.average(local_ys + y_min, weights=weights)

    center_margin = width * 0.12

    if x_mean < (width / 2) - center_margin:
        side = "Left"
    elif x_mean > (width / 2) + center_margin:
        side = "Right"
    else:
        side = "Midline"

    if y_mean < height * 0.40:
        area = "frontal"
    elif y_mean < height * 0.68:
        area = "central/temporal"
    else:
        area = "posterior/occipital"

    return f"{side} {area}"


def apply_scalp_mask(image):
    height, width = image.shape
    y_grid, x_grid = np.ogrid[:height, :width]
    center = (height // 2, width // 2)
    radius = min(height, width) // 2 - 2

    mask = (x_grid - center[1]) ** 2 + (y_grid - center[0]) ** 2 <= radius ** 2
    masked = image.copy()
    masked[~mask] = np.nan
    return masked, mask


def save_overlay(topomap, cam, output_path, title):
    base = topomap.mean(axis=0)
    base, mask = apply_scalp_mask(base)
    cam = cam.copy()
    cam[~mask] = np.nan

    plt.figure(figsize=(7, 6))
    plt.imshow(base, cmap="gray", interpolation="bilinear")
    plt.imshow(cam, cmap="jet", alpha=0.45, interpolation="bilinear")
    plt.gca().add_patch(
        plt.Circle((32, 32), 30, color="black", fill=False, linewidth=1.5)
    )
    plt.colorbar(label="Grad-CAM importance")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run subject-level Grad-CAM localization.")
    parser.add_argument(
        "--subject",
        default=DEFAULT_SUBJECT_DIR,
        help="Path to a subject folder containing ordered seg_*.npy topomap files.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-segments", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    gradcam = GradCAM(model, model.conv[6])
    segment_files = load_subject_segments(args.subject, args.max_segments)

    cams = []
    probabilities = []
    base_maps = []

    try:
        for segment_path in segment_files:
            topomap = load_topomap(segment_path)
            x = torch.tensor(topomap, dtype=torch.float32).unsqueeze(0).to(device)
            cam, logit = gradcam.generate(x)

            cams.append(cam)
            probabilities.append(torch.sigmoid(logit).item())
            base_maps.append(topomap)
    finally:
        gradcam.close()

    subject_cam = np.mean(np.stack(cams), axis=0)
    subject_topomap = np.mean(np.stack(base_maps), axis=0)
    probability = float(np.mean(probabilities))
    prediction = "stroke" if probability >= 0.5 else "non_stroke"
    region = estimate_region(subject_cam)

    if args.output is None:
        subject_name = os.path.basename(os.path.normpath(args.subject))
        args.output = os.path.join("outputs", f"gradcam_{subject_name}.png")

    title = f"Prediction: {prediction} ({probability:.2%}) | Region: {region}"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_overlay(subject_topomap, subject_cam, args.output, title)

    print("Subject:", args.subject)
    print("Segments used:", len(segment_files))
    print("Prediction:", prediction)
    print("Average stroke probability:", f"{probability:.4f}")
    print("Approximate affected region:", region)
    print("Saved Grad-CAM overlay:", args.output)


if __name__ == "__main__":
    main()
