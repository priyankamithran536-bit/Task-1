import torch
import faiss
import numpy as np
from transformers import AutoProcessor, AutoModel
from load import load_documents, load_image
from PIL import Image

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

processor = AutoProcessor.from_pretrained(MODEL_NAME)
device = "cpu"

model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
).to(device)

def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def get_image_embedding(image_path):
    image = load_image(image_path)

    image = Image.open(image_path).convert("RGB")

    messages = [{"role": "user","content": [{"type": "image"},{"type": "text", "text": "Describe this image"}]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor( text=[text],images=[image],return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def build_index(data_folder="data"):
    docs = load_documents(data_folder)

    embeddings = []
    metadata = []

    for doc in docs:
        if doc["type"] == "text":
            emb = get_text_embedding(doc["content"])
        elif doc["type"] == "image":
            emb = get_image_embedding(doc["content"])

        embeddings.append(emb)
        metadata.append(doc)

    embeddings = np.vstack(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "vector_store/index.faiss")

    np.save("vector_store/metadata.npy", metadata, allow_pickle=True)

    print("Index built")

build_index()