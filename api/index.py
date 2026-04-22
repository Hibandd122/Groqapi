import os
import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Groq CP Solver")

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
    """Cố gắng sửa các lỗi JSON phổ biến như thiếu dấu phẩy, escape thừa."""
    # Xóa comment nếu có
    json_str = re.sub(r'//.*', '', json_str)
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    # Thay thế escape không cần thiết
    json_str = json_str.replace('\\n', '\n').replace('\\"', '"').replace("'''", '"""')
    return json_str

@app.post("/api/solve")
async def solve_problem(request: ProblemRequest):
    try:
        system_prompt = """You are a world-class competitive programmer. Your output must be strictly valid JSON with exactly these keys: "analysis", "pseudocode", "solution_code", "complexity".

Rules for JSON:
- Use double quotes for all keys and string values.
- Escape double quotes inside strings with backslash: \\".
- Do NOT include trailing commas.
- The "solution_code" must be a single string with proper Python code (use triple quotes inside the string if needed).
- Ensure the JSON is parseable by standard JSON parsers.

Example output:
{
  "analysis": "The problem asks...",
  "pseudocode": "Step 1: ...",
  "solution_code": "def solve():\\n    return 8",
  "complexity": "O(1) time, O(1) space"
}"""

        user_prompt = f"""Problem: {request.problem_statement}
Input format: {request.input_format or "stdin"}
Output format: {request.output_format or "stdout"}
Constraints: {request.constraints or "none"}
Sample: {request.sample_input or "none"} => {request.sample_output or "none"}

Provide the solution in valid JSON as specified."""

        # Thử gọi API với response_format JSON
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
            # Lỗi 400 do JSON invalid - lấy failed_generation từ error
            error_body = e.body if hasattr(e, 'body') else str(e)
            try:
                error_data = json.loads(error_body) if isinstance(error_body, str) else error_body
                failed_gen = error_data.get('error', {}).get('failed_generation')
                if failed_gen:
                    # Cố gắng sửa và parse lại
                    fixed = fix_json_string(failed_gen)
                    try:
                        return json.loads(fixed)
                    except:
                        # Nếu vẫn fail, trả về raw nhưng đã sửa
                        return {"raw_response": failed_gen, "note": "JSON could not be parsed after fix attempt"}
            except:
                pass
            raise HTTPException(status_code=400, detail="Groq failed to generate valid JSON. Please try again.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Handler cho Vercel
from mangum import Mangum
handler = Mangum(app)
