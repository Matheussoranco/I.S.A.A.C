import json
import re
from dataclasses import dataclass

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

    def __init__(self, model_name: str | None = None):
        from isaac.config.settings import get_settings

        self.llm = get_llm()
        self.model_name = model_name or get_settings().llm.model_name

    def judge(self, task: str, response: str, rubric: str) -> JudgeResult:
        prompt = f"""
You are an expert objective judge. Evaluate the response to the task using the provided rubric.

### Task
{task}

### Response to Evaluate
{response}

### Rubric
{rubric}

### Instructions
1. Carefully compare the response against the rubric.
2. Provide a justification for your score, citing specific parts of the response.
3. Assign a score from 0.0 to 1.0 (1.0 is perfect; 0.0 is incorrect or irrelevant).

Your output must be in the following JSON format:
{{
  "justification": "...",
  "score": 0.85
}}
"""
        llm_response = self.llm.invoke(prompt)
        raw_output = str(getattr(llm_response, "content", llm_response))

        # Extract JSON from potential markdown blocks
        match = re.search(r"(\{.*})", raw_output, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return JudgeResult(
                    score=float(data.get("score", 0.0)),
                    justification=data.get("justification", "No justification provided."),
                )
            except (json.JSONDecodeError, ValueError):
                pass

        return JudgeResult(
            score=0.0,
            justification=f"Judge failed to produce valid JSON. Raw output: {raw_output[:200]}...",
        )
