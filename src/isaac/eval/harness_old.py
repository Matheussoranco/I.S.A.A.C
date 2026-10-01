import json
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

from isaac.eval.judge import LLMJudge
from isaac.llm.provider import get_llm

@dataclass
class EvalTask:
    id: str
    task: str
    rubric: str
    expected: Optional[str] = None
    category: str = "general"
    timeout_seconds: float = 300.0

@dataclass
class EvalResult:
    task_id: str
    response: str
    score: float
    justification: str
    duration_ms: float
    success: bool

@dataclass
class EvalSummary:
    suite_name: str
    accuracy: float
    avg_score: float
    total_tasks: int
    results: List[EvalResult]

class EvalHarness:
    \"\"\"
    Orchestrates the execution of a batch of tasks and uses an LLMJudge to score them.
    \"\"\"
    def __init__(self, judge: LLMJudge, agent_runner: Callable[[str], str]):
        self.judge = judge
        self.agent_runner = agent_runner

    async def run_task(self, task: EvalTask) -> EvalResult:
        start_time = time.perf_counter()
        try:
            # Run the agent on the task
            response = await asyncio.to_thread(self.agent_runner, task.task)
            
            # Judge the response
            judge_res = self.judge.judge(task.task, response, task.rubric)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            return EvalResult(
                task_id=task.id,
                response=response,
                score=judge_res.score,
                justification=judge_res.justification,
                duration_ms=duration_ms,
                success=judge_res.score >= 0.8  # Threshold for 'success'
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return EvalResult(
                task_id=task.id,
                response=f"Error: {str(e)}",
                score=0.0,
                justification=f"Agent failed with exception: {e}",
                duration_ms=duration_ms,
                success=False
            )

    async def run_suite(self, tasks: List[EvalTask], suite_name: str) -> EvalSummary:
        results = []
        for i, task in enumerate(tasks):
            print(f"[{i+1}/{len(tasks)}] Evaluating {task.id}...")
            res = await self.run_task(task)
            results.append(res)
            print(f"  Score: {res.score:.2f} | Success: {res.success}")

        total_score = sum(r.score for r in results)
        success_count = sum(1 for r in results if r.success)
        
        return EvalSummary(
            suite_name=suite_name,
            accuracy=success_count / len(tasks) if tasks else 0,
            avg_score=total_score / len(tasks) if tasks else 0,
            total_tasks=len(tasks),
            results=results
        )

def load_golden_suite(path: str) -> List[EvalTask]:
    \"\"\"Loads tasks from a JSONL file.\"\"\"
    tasks = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            tasks.append(EvalTask(
                id=data.get('id', 'unknown'),
                task=data['task'],
                rubric=data.get('rubric', 'Standard accuracy and completeness.'),
                expected=data.get('expected'),
                category=data.get('category', 'general')
            ))
    return tasks
