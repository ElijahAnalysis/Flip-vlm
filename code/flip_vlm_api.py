

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from PIL import Image
from flip_vlm_dual_encoder495m import FlipDualEncoderv2
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import io
import os
import ast

#### application
app = FastAPI()
app.mount(
    "/images",
    StaticFiles(directory=r"C:\Users\User\Desktop\flip_vlm\flip_data_vlm"),
    name="images"
)



#### dual encoder path
weights_file = r"C:\Users\User\Desktop\flip_vlm\models\flip_dual_encoder28k.pt"
flip_dual_encoder = FlipDualEncoderv2(weights_path=weights_file)
flip_dual_encoder.eval()


#### database
flip_database = pd.read_csv(r"C:\Users\User\Desktop\flip_vlm\flip_data_vlm\all_products_combined_embedded495m.csv.gz")

def fix_list_string(s):
    try:
        return ast.literal_eval(s)
    except:
        # Try inserting commas between numbers
        s_fixed = ",".join(s.strip("[]").split())
        return ast.literal_eval(f"[{s_fixed}]")

valid_embeddings = (
    flip_database['image_embedding']
    .dropna()
    .apply(fix_list_string)
)

image_embs_array = np.array(valid_embeddings.tolist())
flip_database = flip_database.loc[valid_embeddings.index].reset_index(drop=True)



#### FUNCTIONS

def get_images(df):
    base_path = r"C:\Users\User\Desktop\flip_vlm\flip_data_vlm"

    def to_url(full_path):
        # Get the relative path under the base folder
        rel_path = os.path.relpath(full_path, base_path)
        # Convert Windows backslashes to forward slashes for URLs
        return "/images/" + rel_path.replace("\\", "/")

    return [{"image_url": to_url(row["windows_image_path"])} for _, row in df.iterrows()]


def topk_similar(similarity_matrix, k = 10):

    top_k_indices = similarity_matrix.argsort()[-k:][::-1]
    top_results = flip_database.iloc[top_k_indices][['title', 'windows_image_path']].copy()
    top_results['similarity'] = similarity_matrix[top_k_indices]


    image_urls = get_images(top_results)
    
    return image_urls


#### PROTOCOLS

@app.post("/text_search")
async def text_search( text_query : str | None = None ):

    # Get embeddings 
    _, text_embedding = flip_dual_encoder(images=None, texts=[text_query])
    

    # Compute cosine_similarity
    text_image_cosine_similarity = cosine_similarity(text_embedding.detach().cpu().numpy(), image_embs_array).flatten()

    top_k_similar_image_urls = topk_similar( similarity_matrix = text_image_cosine_similarity, k = 10 ) 
    return {"images" : top_k_similar_image_urls}


@app.post("/image_search")
async def image_search( image : UploadFile=File(...) ):

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
        ])

    content = await image.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    
    img_tensor = transform(image).unsqueeze(0).to(flip_dual_encoder.device)
    image_embedding, _ = flip_dual_encoder(images=img_tensor, texts=None)

    image_cosine_similarity = cosine_similarity(image_embedding.detach().cpu().numpy(), image_embs_array).flatten()

    top_k_similar_image_urls = topk_similar( similarity_matrix = image_cosine_similarity, k = 10 ) 
    return {"images" : top_k_similar_image_urls}
    



