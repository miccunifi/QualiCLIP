import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip
from clip.model import CLIP

dependencies = ["torch"]

base_url = "https://github.com/miccunifi/QualiCLIP/releases/download/weights"


class QualiCLIP(nn.Module):
    """
    QualiCLIP model for No-Reference Image Quality Assessment (NR-IQA). The model takes in input an image with CLIP
    normalization and returns the predicted quality score. The predicted quality scores are in the range [0, 1], where
    higher is better.
    """
    def __init__(self):

        super().__init__()
        self.clip_model: CLIP = clip.load("RN50", jit=False)[0].float()

        self.logit_scale = 100.     # Value used in CLIP's paper

        self.prompts = ['Good photo.', 'Bad photo.',
                        'Sharp image.', 'Blurry image.',
                        'Sharp edges.', 'Blurry edges.',
                        'High-resolution image.', 'Low-resolution image.',
                        'Noise-free image.', 'Noisy image.'
                        'High-quality image.', 'Low-quality image.',
                        'Good picture.', 'Bad picture.']
        self.prompts_features = None

        self._load_checkpoint(torch.hub.load_state_dict_from_url(f"{base_url}/qualiclip.pth", progress=True, map_location="cpu"))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        device = img.device
        img_features = self.clip_model.encode_image(img, pos_embedding=self.pos_embedding)
        img_features = F.normalize(img_features, dim=-1)
        prompts_features = self.get_prompts_features(device)

        output = self.logit_scale * img_features @ prompts_features.T

        logits_per_image = output.reshape(output.shape[0], -1, 2)

        similarity_score = logits_per_image.softmax(dim=-1)
        quality_score = similarity_score[..., 0].mean(dim=1)

        return quality_score

    def get_prompts_features(self, device: str) -> torch.Tensor:
        if self.prompts_features is None:
            tokenized_prompts = clip.tokenize(self.prompts).to(device)
            prompts_features = self.clip_model.encode_text(tokenized_prompts)
            prompts_features = F.normalize(prompts_features, dim=-1)
            self.prompts_features = nn.Parameter(prompts_features, requires_grad=False)

        return self.prompts_features

    def _load_checkpoint(self, state_dict: dict) -> None:
        clip_model_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("clip_model."):
                # Remove the prefix
                clip_model_state_dict[k.replace("clip_model.", "")] = v
        self.clip_model.load_state_dict(clip_model_state_dict)
        self.prompts_features = state_dict["prompts_features"]
