from typing import List, Optional
import os
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

from .schemas import DetectionResult, Slot
from .config import settings


_MODEL = None
_DEVICE = None
_IMG_SIZE = 224


def _get_candidate_paths():
    # prefer environment-configured MODEL_PATH, then common training outputs
    paths = []
    if getattr(settings, 'MODEL_PATH', None):
        paths.append(settings.MODEL_PATH)
    paths.extend([
        'training/run_quick/model_best.pth',
        'training/run_quick/model_final.pth',
        'training/run_full/model_best.pth',
        'training/run_full/model_final.pth',
    ])
    return paths


def load_model(path: Optional[str] = None, device: Optional[torch.device] = None):
    global _MODEL, _DEVICE, _IMG_SIZE
    if _MODEL is not None:
        return _MODEL
    if device is None:
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        _DEVICE = device

    candidate = path
    if candidate is None:
        for p in _get_candidate_paths():
            if os.path.isfile(p):
                candidate = p
                break
    if candidate is None or not os.path.isfile(candidate):
        return None

    # only support PyTorch .pth state_dict files for now
    try:
        # lazy import of training model builder to avoid heavy deps at import time
        from training.train import build_model

        model = build_model()
        state = torch.load(candidate, map_location=_DEVICE)
        model.load_state_dict(state)
        model.to(_DEVICE)
        model.eval()
        _MODEL = model
        return _MODEL
    except Exception:
        return None


def _crop_slot_image(frame_bgr: np.ndarray, polygon: List[List[int]], img_size: int):
    # polygon is list of [x,y]
    pts = np.array(polygon, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    # clamp
    h_img, w_img = frame_bgr.shape[:2]
    x2 = min(x + w, w_img)
    y2 = min(y + h, h_img)
    x = max(0, x)
    y = max(0, y)
    crop = frame_bgr[y:y2, x:x2]
    if crop.size == 0:
        return None
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop)
    tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    return tf(pil)


def detect_with_model(frame_bgr: np.ndarray, slots: List[Slot], threshold: float = 0.5, model_path: Optional[str] = None, img_size: int = 224) -> Optional[List[DetectionResult]]:
    """Run the trained classifier on each slot. Returns list of DetectionResult or None if no model loaded."""
    global _MODEL, _DEVICE, _IMG_SIZE
    _IMG_SIZE = img_size
    if _MODEL is None:
        mdl = load_model(model_path)
        if mdl is None:
            return None
    model = _MODEL
    device = _DEVICE or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tensors = []
    slot_ids = []
    for slot in slots:
        t = _crop_slot_image(frame_bgr, slot.polygon, img_size)
        if t is None:
            tensors.append(None)
        else:
            tensors.append(t)
        slot_ids.append(slot.id)

    # batch non-None tensors
    nonnull = [(i, t) for i, t in enumerate(tensors) if t is not None]
    results = []
    if nonnull:
        batch = torch.stack([t for _, t in nonnull], dim=0).to(device)
        with torch.no_grad():
            out = model(batch)
            out = out.squeeze(1).cpu().numpy()
        # map back
        out_idx = 0
        for i, t in enumerate(tensors):
            if t is None:
                # fallback: unknown -> mark unoccupied with low confidence
                results.append(DetectionResult(slot_id=str(slot_ids[i]), occupied=False, confidence=0.0))
            else:
                prob = float(out[out_idx])
                occupied = prob >= threshold
                conf = float(np.clip(prob, 0.0, 1.0))
                results.append(DetectionResult(slot_id=str(slot_ids[i]), occupied=bool(occupied), confidence=conf))
                out_idx += 1
    else:
        # no valid crops, return empty results
        for i in range(len(slots)):
            results.append(DetectionResult(slot_id=str(slot_ids[i]), occupied=False, confidence=0.0))

    return results
