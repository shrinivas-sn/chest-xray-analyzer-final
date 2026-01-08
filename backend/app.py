import cv2
import gradio as gr
import numpy as np
import spaces
import torch
import torch.nn.functional as F
import pydicom
from PIL import Image 
from einops import rearrange
from transformers import AutoModel

# --- 1. HELPER FUNCTIONS ---

def calculate_ctr(mask: np.ndarray) -> float:
    lungs = np.zeros_like(mask)
    lungs[mask == 1] = 1
    lungs[mask == 2] = 1
    heart = (mask == 3).astype("int")
    
    # Safety check for empty masks
    if not np.any(lungs == 1) or not np.any(heart == 1):
        return 0.0
        
    y, x = np.stack(np.where(lungs == 1))
    lung_min, lung_max = x.min(), x.max()
    y, x = np.stack(np.where(heart == 1))
    heart_min, heart_max = x.min(), x.max()
    
    if (lung_max - lung_min) == 0: 
        return 0.0
        
    return (heart_max - heart_min) / (lung_max - lung_min)

def make_overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    overlay = alpha * img + (1 - alpha) * mask
    return overlay.astype(np.uint8)

def load_image_safe(filepath):
    """
    Universal Loader:
    1. Tries to read as DICOM (ignoring extension).
    2. Falls back to Standard Image (OpenCV).
    3. STANDARDIZES everything to Grayscale uint8 (0-255).
    """
    try:
        # ATTEMPT 1: DICOM
        # force=True allows reading files even if they lack the ".dcm" extension
        ds = pydicom.dcmread(filepath, force=True)
        
        # Check if it actually has pixel data
        if 'PixelData' not in ds:
            raise ValueError("No pixel data in DICOM")

        pixel_array = ds.pixel_array.astype(float)
        
        # Normalize (Windowing) to 0-255 range
        if np.max(pixel_array) != np.min(pixel_array):
            pixel_array = pixel_array - np.min(pixel_array)
            pixel_array = pixel_array / np.max(pixel_array)
            pixel_array = (pixel_array * 255).astype(np.uint8)
        else:
            pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
            
        return pixel_array
            
    except Exception:
        # ATTEMPT 2: Standard Image (PNG/JPG)
        try:
            # imread_grayscale ensures we get the same 2D shape as DICOM
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            return img
        except Exception as e:
            print(f"Failed to load file {filepath}: {e}")
            return None

# --- 2. MAIN PREDICTION FUNCTION ---
@spaces.GPU
def predict(Radiograph_path):
    # Step A: Load and Standardize
    Radiograph = load_image_safe(Radiograph_path)
    
    if Radiograph is None:
        return ["Error: The file is corrupted or not a valid X-ray.", None, None, None, None]

    # Step B: Prepare for Model (Convert to RGB)
    rg = cv2.cvtColor(Radiograph, cv2.COLOR_GRAY2RGB)
    
    # Step C: AI Inference
    x = cxr_info_model.preprocess(Radiograph)
    x = torch.from_numpy(x).float().to(device)
    x = rearrange(x, "h w -> 1 1 h w")

    with torch.inference_mode():
        info_out = cxr_info_model(x)

    # --- Process Output 1: Heart & Lungs ---
    info_mask = info_out["mask"]
    h, w = rg.shape[:2]
    info_mask = F.interpolate(info_mask, size=(h, w), mode="bilinear")
    info_mask = info_mask.argmax(1)[0]
    info_mask_3ch = F.one_hot(info_mask, num_classes=4)[..., 1:]
    info_mask_3ch = (info_mask_3ch.cpu().numpy() * 255).astype(np.uint8)
    info_overlay = make_overlay(rg, info_mask_3ch[..., ::-1])

    view = info_out["view"].argmax(1).item()
    info_string = ""
    if view in {0, 1}:
        info_string += "This is a frontal chest radiograph "
        info_string += "(AP projection)." if view == 0 else "(PA projection)."
    elif view == 2:
        info_string += "This is a lateral chest radiograph."

    age = info_out["age"].item()
    info_string += f"\nThe patient's predicted age is {round(age)} years."
    sex = "male" if info_out["female"].item() < 0.5 else "female"
    info_string += f"\nThe patient's predicted sex is {sex}."

    if view in {0, 1}:
        ctr = calculate_ctr(info_mask.cpu().numpy())
        info_string += f"\nThe estimated cardiothoracic ratio (CTR) is {ctr:0.2f}."

    # --- Process Output 2: Pneumonia ---
    x = pna_model.preprocess(Radiograph)
    x = torch.from_numpy(x).float().to(device)
    x = rearrange(x, "h w -> 1 1 h w")

    with torch.inference_mode():
        pna_out = pna_model(x)

    pna_mask = pna_out["mask"]
    pna_mask = F.interpolate(pna_mask, size=(h, w), mode="bilinear")
    pna_mask = (pna_mask.cpu().numpy()[0, 0] * 255).astype(np.uint8)
    pna_mask = cv2.applyColorMap(pna_mask, cv2.COLORMAP_JET)
    pna_overlay = make_overlay(rg, pna_mask[..., ::-1])

    # --- Process Output 3: Pneumothorax ---
    x = ptx_model.preprocess(Radiograph)
    x = torch.from_numpy(x).float().to(device)
    x = rearrange(x, "h w -> 1 1 h w")

    with torch.inference_mode():
        ptx_out = ptx_model(x)

    ptx_mask = ptx_out["mask"]
    ptx_mask = F.interpolate(ptx_mask, size=(h, w), mode="bilinear")
    ptx_mask = (ptx_mask.cpu().numpy()[0, 0] * 255).astype(np.uint8)
    ptx_mask = cv2.applyColorMap(ptx_mask, cv2.COLORMAP_JET)
    ptx_overlay = make_overlay(rg, ptx_mask[..., ::-1])

    preds = {"Pneumonia": pna_out["cls"].item(), "Pneumothorax": ptx_out["cls"].item()}
    return [info_string, preds, info_overlay, pna_overlay, ptx_overlay]


# --- 3. UI SETUP ---

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device `{device}` ...")

    # Load Models
    cxr_info_model = AutoModel.from_pretrained("ianpan/chest-x-ray-basic", trust_remote_code=True).eval().to(device)
    pna_model = AutoModel.from_pretrained("ianpan/pneumonia-cxr", trust_remote_code=True).eval().to(device)
    ptx_model = AutoModel.from_pretrained("ianpan/pneumothorax-cxr", trust_remote_code=True).eval().to(device)

    
    demo = gr.Interface(
        fn=predict,
        inputs=gr.File(label="Upload X-ray (Standard Image or DICOM)", type="filepath"),
        outputs=[
            gr.Textbox(show_label=False),
            gr.Label(show_label=False, show_heading=False),
            gr.Image(image_mode="RGB", label="Heart & Lungs"),
            gr.Image(image_mode="RGB", label="Pneumonia"),
            gr.Image(image_mode="RGB", label="Pneumothorax")
        ],
        title="Deep Learning for Chest X-ray Images",
        description="Supports **DICOM (.dcm)** and Standard Images (PNG, JPG)",
        cache_examples=False,
    )

    demo.launch(share=True)