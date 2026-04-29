
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from torchvision import transforms as T
from PIL import Image
from utils_helper import crop_and_resize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- TRANSFORM --------------------
transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])
# ========= special transformer =========
img_size = 300
g_mean = (0.485, 0.456, 0.406)
g_std = (0.229, 0.224, 0.225)
validation_transform_pipeline = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=g_mean, std=g_std),
    ToTensorV2(),
])
# ======== classification transofrmer =========== 
classification_transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# -------------------- PREDICTION --------------------
def cnn_predict_bbox(image_pil, model): 
    img = np.array(image_pil)

    transformed = validation_transform_pipeline(image=img)
    img_tensor = transformed["image"].to(device)
    with torch.no_grad():
        pred = model(img_tensor.unsqueeze(0).to(device))

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    h_img, w_img, _ = img_np.shape

    x, y, w, h, obj_prob = pred[0].detach().cpu()

    x = int(x.item() * w_img)
    y = int(y.item() * h_img)
    w = int(w.item() * w_img)
    h = int(h.item() * h_img)

    x, y = max(0, x), max(0, y)
    w, h = max(1, w), max(1, h)

    x = min(x, w_img - 1)
    y = min(y, h_img - 1)
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    # denormalize
    img_np = img_np * np.array(g_std) + np.array(g_mean)
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_np, (x, y, w, h)


def predict_bbox(image_pil, model, model_name): 
    if model_name == "CNN (Unet)":
        return cnn_predict_bbox(image_pil, model)
    else:
        with torch.no_grad():
            pred = model(image_pil)[0] 
            
        if pred.boxes is None or len(pred.boxes) == 0:
            return image, None   # handle if no box found
        # take highest confidence box
        pred_box = pred.boxes.xyxy[0].cpu().numpy().astype(int).tolist()
        img = np.array(image_pil)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw predicted box (red)
        cv2.rectangle(img_rgb, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (255,0,0), 2)

        return img_rgb, pred_box




def predict_contour(image_pil, model, bbox, device): 
    image_np = np.array(image_pil)

    crop_img = crop_and_resize(image_np, bbox)

    augmented = transform(image=crop_img)
    x = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)

    pred = torch.sigmoid(pred)
    mask = (pred > 0.5).float().squeeze().cpu().numpy()

    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1

    mask = cv2.resize(mask, (box_w, box_h))

    full_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = (mask * 255).astype(np.uint8)

    # CREATE OVERLAY 

    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # green mask
    mask_color = np.zeros_like(img_rgb)
    mask_color[:, :, 1] = full_mask  # green channel

    overlay = cv2.addWeighted(img_rgb, 0.7, mask_color, 0.3, 0)

    return overlay


def predict_contour_cnn_resnet(image_pil, model):
    model.eval()

    # PIL → numpy
    img = np.array(image_pil)

    # apply SAME transform as training
    transformed = validation_transform_pipeline(image=img)
    img_tensor = transformed["image"].to(device)

    with torch.no_grad():
        pred = model(img_tensor.unsqueeze(0))
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

    # prepare image for drawing
    img_vis = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_vis = img_vis * np.array(g_std) + np.array(g_mean)
    img_vis = (img_vis * 255).clip(0, 255).astype(np.uint8).copy()

    # mask → binary
    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    # find contours
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours (green)
    cv2.drawContours(img_vis, contours, -1, (0, 255, 0), 2)

    return img_vis

def predict_classes(image_pil, model, bbox, device): 
    image_np = np.array(image_pil)
    print(f"bbox: {bbox}")
    crop_img = crop_and_resize(image_np, bbox)

    # augmented = classification_transform(image=crop_img)
    # x = augmented["image"].unsqueeze(0).to(device)
    crop_pil = Image.fromarray(crop_img)   # VERY IMPORTANT
    x = classification_transform(crop_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).squeeze()
        # print(f'predicted classes: {pred}')
    prob = torch.sigmoid(pred).item()
    classes = ["Benign", "Malignant"]

    index = 1 if prob > 0.5 else 0
    confidence = prob if prob > 0.5 else (1.0 - prob)
    probs = [round(1.0 - prob, 4), round(prob, 4)]
    return {
        "label": classes[index],
        "confidence": round(confidence, 4),
        "all_probs": dict(zip(classes, probs))
    }
