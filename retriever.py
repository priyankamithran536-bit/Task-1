import faiss
import numpy as np
import torch
from transformers import AutoProcessor, AutoModel
from load import load_image

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
).to(device)

index = faiss.read_index("vector_store/index.faiss")
metadata = np.load("vector_store/metadata.npy", allow_pickle=True)

def embed_query(query_text=None, image_path=None):
    if query_text:
        inputs = processor(text=[query_text], return_tensors="pt").to(device)
    
    
    elif image_path:
        image = load_image(image_path)
        
        messages = [{"role": "user","content": [{"type": "image"},{"type": "text", "text": query_text or "Describe this image"}]}]

        text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

        inputs = processor(text=[text],images=[image],return_tensors="pt").to(device)
    

    else:
        raise ValueError("Error")

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def search(query_text=None, image_path=None, top_k=3):
    query_emb = embed_query(query_text, image_path)

    D, I = index.search(query_emb.astype("float32"), top_k)

    results = []
    for idx in I[0]:
        results.append(metadata[idx])

    return results