import os
import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing")

client = openai.OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

class ProblemRequest(BaseModel):
    problem_statement: str
    input_format: str = ""
    output_format: str = ""
    constraints: str = ""
    sample_input: str = ""
    sample_output: str = ""

def clean_json_string(s: str) -> str:
    s = re.sub(r',\s*}', '}', s)
    s = re.sub(r',\s*]', ']', s)
    return s

@app.post("/api/solve")
async def solve(req: ProblemRequest):
    system_prompt = """You are an AI that provides only the final answer with no extra text.
Output a JSON with a single key "answer". The answer format depends on the question type:
- Multiple choice (A, B, C, D): just the letter (e.g., "A")
- True/False statements (a, b, c, d each true/false): list them like "a) Đúng, b) Sai, c) Đúng, d) Đúng"
- Short text: direct answer (e.g., "8 bit", "Sách")
- Code: Python code as string if problem requires programming.
Never include explanations."""
    user_prompt = f"Problem: {req.problem_statement}"

    try:
        # Thử dùng response_format JSON
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        # Fallback: gọi không có response_format
        try:
            resp2 = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            text = resp2.choices[0].message.content.strip()
            # Cố gắng parse JSON
            try:
                return json.loads(text)
            except:
                # Nếu không phải JSON, thử trích xuất answer bằng regex
                match = re.search(r'"?answer"?\s*[:=]\s*"([^"]+)"', text, re.IGNORECASE)
                if match:
                    return {"answer": match.group(1)}
                return {"answer": text}
        except Exception as fallback_error:
            raise HTTPException(500, detail=str(fallback_error))

# Handler cho Vercel
from mangum import Mangum
handler = Mangum(app)
