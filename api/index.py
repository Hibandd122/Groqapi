import os
import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Groq Answer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

client = openai.OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

class ProblemRequest(BaseModel):
    problem_statement: str
    input_format: str = ""
    output_format: str = ""
    constraints: str = ""
    sample_input: str = ""
    sample_output: str = ""

def fix_json_string(json_str: str) -> str:
    json_str = re.sub(r'//.*', '', json_str)
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    return json_str

@app.post("/api/solve")
async def solve_problem(request: ProblemRequest):
    try:
        system_prompt = """You are an AI that provides only the final answer in JSON format with a single key "answer".

Instructions:
- If the problem is a multiple-choice question with options labeled A, B, C, D, output ONLY the letter of the correct option (e.g., "A").
- If the problem is a true/false question without options, output "Đúng" or "Sai" (in Vietnamese).
- If the problem asks for a specific answer (e.g., "What is X?"), output the exact answer as a string.
- If the problem requires code, output the complete Python code as a string.

Do NOT include explanations. Output only valid JSON.

Examples:
{"answer": "A"}
{"answer": "Đúng"}
{"answer": "Hà Nội"}
{"answer": "print('Hello')"}
"""

        user_prompt = f"Problem: {request.problem_statement}"

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except openai.BadRequestError as e:
            error_body = e.body if hasattr(e, 'body') else str(e)
            try:
                error_data = json.loads(error_body) if isinstance(error_body, str) else error_body
                failed_gen = error_data.get('error', {}).get('failed_generation')
                if failed_gen:
                    fixed = fix_json_string(failed_gen)
                    return json.loads(fixed)
            except:
                pass
            raise HTTPException(status_code=400, detail="JSON generation failed")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from mangum import Mangum
handler = Mangum(app)
