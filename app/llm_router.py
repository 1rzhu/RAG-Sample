import os
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel
from qdrant_client import QdrantClient

MATERIALS_DIR = os.path.join(os.path.dirname(__file__), "materials")
COLLECTION_NAME = "txt_vector_collection"
EMBEDDING_MODEL = "text-embedding-ada-002"

router = APIRouter()


class PromptRequest(BaseModel):
    prompt: str


class PromptResponse(BaseModel):
    response: ChatCompletionMessage


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url="http://localhost:6333")


@router.post("", response_model=PromptResponse)
async def llm(request: PromptRequest):
    def get_embedding(text: str, model: str=EMBEDDING_MODEL):
        text = text.replace("\n", " ")
        return openai_client.embeddings.create(input=[text], model=model).data[0].embedding

    openai_client = get_openai_client()
    qdrant_client = get_qdrant_client()

    embedding = get_embedding(request.prompt)

    try:
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=3,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant search error: {e}")

    # Extract context from search results
    context_list = [point.payload.get("page_content", "") for point in search_result]
    context = "\n\n".join(
        [f"... doc {i + 1} ...\n{doc}" for i, doc in enumerate(context_list)]
    )

    formatted_message = f"""Please answer the question using the provided context.
    <Context>
    {context}
    <Context/>

    Now answer the user question:
    <user question>
    {request.prompt}
    <user question/>"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": formatted_message},
    ]

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )
        return PromptResponse(response=completion.choices[0].message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion error: {e}")
