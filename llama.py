from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# 모델과 토크나이저 불러오기 (LLaMA-7B 예시)
model_name = "huggingface/llama-7b"  # 예시 모델, 실제 경로로 변경 필요

# 토크나이저 로드
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# 모델 로드
model = LlamaForCausalLM.from_pretrained(model_name)

# 입력 텍스트 예시
input_text = "Hello, how are you?"

# 입력 텍스트를 토큰화
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 모델을 GPU로 이동 (가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델 추론 실행
with torch.no_grad():
    output = model.generate(input_ids=input_ids.to(device), max_length=50)

# 출력 결과 디코딩
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
