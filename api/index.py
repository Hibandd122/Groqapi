import os, json, re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing")
client = openai.OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

class ProblemRequest(BaseModel):
    problem_statement: str
    input_format: str = ""
    output_format: str = ""
    constraints: str = ""
    sample_input: str = ""
    sample_output: str = ""

@app.post("/api/solve")
async def solve(req: ProblemRequest):
    system_prompt = """You are an AI that provides only the final answer with no extra text.
Output a JSON with key "answer". The answer can be:
- Multiple choice: just the letter (A, B, C, D)
- True/False: "Đúng" or "Sai"
- Short text: direct answer (e.g., "8 bit", "Sách")
- Code: Python code as string if problem requires programming.
Never include explanations."""
    user_prompt = f"Problem: {req.problem_statement}"
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
            temperature=0.0,
            response_format={"type":"json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        # fallback nếu lỗi JSON
        try:
            # thử gọi lại không dùng response_format
            resp2 = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
                temperature=0.0
            )
            text = resp2.choices[0].message.content.strip()
            # cố gắng parse
            return {"answer": text}
        except:
            raise HTTPException(500, detail=str(e))

from mangum import Mangum
handler = Mangum(app)
