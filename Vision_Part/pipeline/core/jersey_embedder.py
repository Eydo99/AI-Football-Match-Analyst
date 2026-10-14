import torch
import numpy as np
import cv2
from transformers import CLIPProcessor, CLIPModel

class JerseyEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[JerseyEmbedder] Loading CLIP model {model_name} on {self.device}...")
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        except Exception as e:
            print(f"[JerseyEmbedder] Network connection failed ({e}). Falling back to local cache...")
            self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
            self.model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(self.device)
            
        self.model.eval()
        print("[JerseyEmbedder] CLIP model loaded.")

    def _get_crop(self, frame_bgr, bbox_xyxy):
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        h_frame, w_frame = frame_bgr.shape[:2]

        x1 = max(0, min(x1, w_frame))
        y1 = max(0, min(y1, h_frame))
        x2 = max(0, min(x2, w_frame))
        y2 = max(0, min(y2, h_frame))

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        if bbox_w < 10 or bbox_h < 10:
            return None

        jersey_y1 = y1 + int(bbox_h * 0.35)
        jersey_y2 = y1 + int(bbox_h * 0.65)
        margin_x = int(bbox_w * 0.10)
        jersey_x1 = x1 + margin_x
        jersey_x2 = x2 - margin_x

        if jersey_x2 - jersey_x1 < 5 or jersey_y2 - jersey_y1 < 5:
            return None

        crop_bgr = frame_bgr[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
        if crop_bgr.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return crop_rgb

    def embed_batch(self, frame_bgr, bboxes_xyxy):
        crops = []
        valid_indices = []

        for i, bbox in enumerate(bboxes_xyxy):
            crop = self._get_crop(frame_bgr, bbox)
            if crop is not None:
                crops.append(crop)
                valid_indices.append(i)

        results = [None] * len(bboxes_xyxy)

        if not crops:
            return results

        with torch.no_grad():
            inputs = self.processor(images=crops, return_tensors="pt").to(self.device)

            if hasattr(self.model, "get_image_features"):
                image_features = self.model.get_image_features(**inputs)
            else:
                image_features = self.model(**inputs)

            if not isinstance(image_features, torch.Tensor):
                if hasattr(image_features, "image_embeds"):
                    image_features = image_features.image_embeds
                elif hasattr(image_features, "pooler_output"):
                    image_features = image_features.pooler_output
                else:
                    image_features = image_features[0]

            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            embeddings = image_features.cpu().numpy()

        for idx, emb in zip(valid_indices, embeddings):
            results[idx] = emb

        return results

    def embed_text(self, text_prompts):
        with torch.no_grad():
            inputs = self.processor(text=text_prompts, padding=True, return_tensors="pt").to(self.device)
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy()
