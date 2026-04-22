import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Groq Competitive Programming API")

# Cho phép CORS để userscript có thể gọi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Phải được đặt trong environment Vercel
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

@app.post("/api/solve")
async def solve_problem(request: ProblemRequest):
    """Giải bài toán tin học với prompt đã được tối ưu."""
    try:
        system_prompt = """You are a world-class competitive programmer with extensive experience in ACM/ICPC and Codeforces. Your task is to analyze and solve algorithmic problems with precision, efficiency, and clarity.

        You will follow a structured approach:
        1. **Problem Analysis**: Understand the problem, identify edge cases, and restate the problem in your own words.
        2. **Algorithm Design**: Choose the most suitable algorithms and data structures based on the constraints.
        3. **Step-by-Step Reasoning (Chain-of-Thought)**: Provide a detailed, step-by-step reasoning process for your solution. Use at least 3 reasoning steps. In your reasoning, explore alternative answers and be aware of your limitations.
        4. **Pseudocode**: Write clear, language-agnostic pseudocode that captures the logic.
        5. **Solution Code**: Provide the complete Python 3 solution code. The code must be clean, well-commented, and follow PEP 8 guidelines.
        6. **Complexity Analysis**: State the time and space complexity of your solution.

        Output Format: You must ALWAYS respond in valid JSON format with the following keys: 'analysis', 'pseudocode', 'solution_code', 'complexity'."""

        user_prompt = f"""### Problem Statement:
        {request.problem_statement}

        ### Input Format:
        {request.input_format if request.input_format else "Standard input (stdin)"}

        ### Output Format:
        {request.output_format if request.output_format else "Standard output (stdout)"}

        ### Constraints:
        {request.constraints if request.constraints else "None provided."}

        ### Sample Input:
        {request.sample_input if request.sample_input else "None provided."}

        ### Sample Output:
        {request.sample_output if request.sample_output else "None provided."}

        Please solve the problem and provide the output in the specified JSON format."""

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
    except json.JSONDecodeError:
        return {"raw_response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Handler cho Vercel (cần export app)
# FastAPI tự tương thích với ASGI, Vercel sẽ dùng mangum hoặc asgi handler.
# Chúng ta thêm dòng này để chạy trên Vercel:
from mangum import Mangum
handler = Mangum(app)
