import os
from typing import List

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pypdf import PdfReader


MATERIALS_DIR = os.path.join(os.path.dirname(__file__), "materials")
COLLECTION_NAME = "txt_vector_collection"
EMBEDDING_MODEL = "text-embedding-ada-002"


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url="http://localhost:6333")


def list_material_files() -> List[str]:
    if not os.path.exists(MATERIALS_DIR):
        return []
    return [
        f for f in os.listdir(MATERIALS_DIR)
        if os.path.isfile(os.path.join(MATERIALS_DIR, f))
    ]


def read_material_content(filename: str):
    path = os.path.join(MATERIALS_DIR, filename)
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        text = []
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
        return "\n".join(text)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def init_qdrant_with_materials():
    qdrant_client = get_qdrant_client()
    openai_client = get_openai_client()

    try:
        qdrant_client.get_collection(COLLECTION_NAME)
    except Exception:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )

    files = list_material_files()
    if not files:
        print("No material files found in", MATERIALS_DIR)
        return

    points = []

    def get_embedding(text: str, model: str = EMBEDDING_MODEL):
        text = text.replace("\n", " ")
        return openai_client.embeddings.create(input=[text], model=model).data[0].embedding

    for idx, filename in enumerate(files):
        content = read_material_content(filename)
        embedding = get_embedding(content)

        points.append(
            models.PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "filename": filename,
                    "page_content": content,
                },
            )
        )

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Loaded {len(points)} material docs into Qdrant.")