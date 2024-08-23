from fastapi import FastAPI, Request, APIRouter
from pydantic import BaseModel
from huggingface_hub.hf_api import HfFolder
import huggingface_hub
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import os
from dotenv import load_dotenv
# accelerate bitsandbytes

load_dotenv()
my_secret_key = os.environ.get('HF_TOKEN')
HfFolder.save_token(my_secret_key) # Pass the actual token value
huggingface_hub.login(my_secret_key)

app = FastAPI(
    title="airon",
    description="패키지 관리자",
    version="0.0.1"
)

router = APIRouter(
    prefix="/airon",
    responses={404: {"description": "Not found"}}
)

# 모델과 토크나이저 로드
model_id = "meta-llama/Meta-Llama-3-8B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
)

def get_response(prompt):
  response = text_generator(prompt)
  gen_text = response[0]['generated_text']
  return gen_text

# 요청으로 받을 데이터의 구조 정의
class TextRequest(BaseModel):
    text: str

@app.post("/generate")
async def generate_text(request: TextRequest):
    input_text = request.text
    
    # prompt = "chatGPT는 인간이 개발한것중 가장"
    llama_response = get_response(input_text)
    return llama_response


# 서버 실행: uvicorn을 통해 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
