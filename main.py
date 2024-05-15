from transformers import AutoTokenizer
import transformers 
import torch

from typing import Union
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel

# Modle Import
model = "PY007/TinyLlama-1.1B-Chat-v0.1"
print(f"{model} setting precessing...")
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("Done.")

# Run API Server
app = FastAPI()
router = APIRouter()

class Content(BaseModel):
    prompt: str = None

# Use Instead Of Real DataBase
sequence_settings = {}

@app.get("/ping")
def read_root():
    return {"message": "Pong"}

# LLM Question Path
@app.post("/question")
async def generate_response(prompt: Content):

    formatted_prompt = f"Human: answering follow comment question is that {prompt} \n Assistant: "

    # Error Handling
    try:
        # Load LLM Setting Values
        result = pipeline(
            formatted_prompt,
            do_sample=True, # 확률적 샘플링 시 고려되는 상위 N개의 후보
            top_k=50,  # 확률적 샘플링 시 고려되는 상위 N%의 후보
            top_p=0.7, # 답변 횟수
            num_return_sequences=1, # 반복 토큰 방지. 1이상이면 대게 많이 줄어듦
            repetition_penalty=2.0, # 한 번에 생성할 최대 토큰 수
            max_new_tokens=100,
        )
        if result and len(result) > 0:
            # Parse The Result
            generated_text = result[0]['generated_text'].split("\n Assistant:")[1]
            return {generated_text}
        else:
            return {"error": "No text was generated."}
    except Exception as e:
        return {"error": str(e)}

app.include_router(router, prefix="/api")