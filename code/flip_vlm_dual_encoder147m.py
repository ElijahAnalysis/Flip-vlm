import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlipDualEncoderV3(nn.Module):
    def __init__(self, embedding_dimension=256, weights_path=None):
        super().__init__()

        # EfficientNetB3 as visual encoder
        efficientnetb3 = models.efficientnet_b3(pretrained=True)
        efficientnetb3.classifier = nn.Identity()
        self.visual_encoder = efficientnetb3
        self.image_projection = nn.Linear(in_features=1536, out_features=embedding_dimension)

        # DistilLabSE as text encoder
        self.text_encoder = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
        self.text_encoder = self.text_encoder.to(device)
        self.text_projection = nn.Linear(in_features=512, out_features=embedding_dimension)

        self.device = device
        self.to(self.device)

        # Load weights if provided
        if weights_path is not None:
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.load_state_dict(state_dict)
                print(f"✅ Loaded weights from: {weights_path}")
            except Exception as e:
                print(f"❌ Failed to load weights: {e}")

    def forward(self, images=None, texts=None):
        image_embedding = None
        text_embedding = None

        if images is not None:
            image_features = self.visual_encoder(images.to(self.device))
            image_embedding = self.image_projection(image_features)
            image_embedding = F.normalize(image_embedding, p=2, dim=-1)
            print("✅ Image embedding:", image_embedding.shape)

        if texts is not None:
            try:
                text_features = self.text_encoder.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=False
                )
                if isinstance(text_features, torch.Tensor):
                    text_features = text_features.to(self.device)
                text_embedding = self.text_projection(text_features)
                text_embedding = F.normalize(text_embedding, p=2, dim=-1)
                print("✅ Text embedding:", text_embedding.shape)
            except Exception as e:
                print(f"❌ Text encoding failed: {e}")
                text_embedding = None

        return image_embedding, text_embedding
