import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlipDualEncoderv2(nn.Module):
    def __init__(self, embedding_dim=256, device=device, weights_path=None):
        super().__init__()
        self.device = device

        # ResNet50 as visual encoder
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet50.fc = nn.Identity()
        self.visual_encoder = resnet50.to(self.device)
        self.image_projection = nn.Linear(2048, embedding_dim).to(self.device)

        # LaBSE as text encoder
        self.text_encoder = SentenceTransformer('sentence-transformers/LaBSE').to(self.device)
        self.text_projection = nn.Linear(768, embedding_dim).to(self.device)

        if weights_path:
            self.load_weights(weights_path)

    def load_weights(self, path):
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.load_state_dict(state_dict)
            print(f"✅ Loaded weights from: {path}")
        except Exception as e:
            print(f"❌ Failed to load weights: {e}")

    def forward(self, images=None, texts=None):
        image_embedding = None
        text_embedding = None

        if images is not None:
            image_encoded = self.visual_encoder(images)
            image_embedding = self.image_projection(image_encoded)
            image_embedding = F.normalize(image_embedding, p=2, dim=-1)

        if texts is not None:
            with torch.no_grad():
                text_features = self.text_encoder.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=False
                )
            text_features = text_features.to(self.device)
            text_embedding = self.text_projection(text_features)
            text_embedding = F.normalize(text_embedding, p=2, dim=-1)

        return image_embedding, text_embedding

