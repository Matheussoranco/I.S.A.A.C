from dataclasses import dataclass
from typing import Tuple
import json
import re
from isaac.llm.provider import get_llm

@dataclass
class JudgeResult:
    score: float
    justification: str

class LLMJudge:
    """
    LLM-based judge that evaluates a model's response against a task and a rubric.
    Returns a score (0-1) and a detailed justification.
    """
    def __init__(self, model_name: str = None):
        self.llm = get_llm()
        self.model_name = model_name or self.llm.model_name

    def judge(self, task: str, response: str, rubric: str) -> JudgeResult:
        prompt = f'''
You are an expert objective judge. Your task is to evaluate a response to a specific task based on a provided rubric.

### Task
{task}

### Response to Evaluate
{response}

### Rubric
{rubric}

### Instructions
1. Carefully compare the response against the rubric.
2. Provide a justification for your score, citing specific parts of the response.
3. Assign a final score between 0.0 and 1.0 (where 1.0 is perfect and 0.0 is completely incorrect or irrelevant).

Your output must be in the following JSON format:
{{
  "justification": "...",
  "score": 0.85
}}
'''
        raw_output = self.llm.complete(prompt)
        
        # Extract JSON from potential markdown blocks
        match = re.search(r'(\{.*})', raw_output, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return JudgeResult(
                    score=float(data.get("score", 0.0)),
                    justification=data.get("justification", "No justification provided.")
                )
            except (json.JSONDecodeError, ValueError):
                pass
        
        return JudgeResult(score=0.0, justification=f"Judge failed to produce valid JSON. Raw output: {raw_output[:200]}...")
