"""Behavioural promotion gate — synthesize args, sandbox exec, schema match.

This gate validates that a skill's code behaves correctly by:
1. Synthesizing test arguments for the skill's main function
2. Executing the skill in a sandboxed environment
3. Matching the output against an expected JSON schema

A skill is only promoted to "stable" if it passes the behavioural gate
in addition to the track-record-based curation criteria.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from typing import Any

from isaac.sandbox.executor import CodeExecutor, ExecutionResult
from isaac.sandbox.security import default_policy

logger = logging.getLogger(__name__)


@dataclass
class PromotionGateResult:
    """Result of the behavioural promotion gate."""

    skill_name: str
    passed: bool
    reason: str
    test_cases_run: int
    test_cases_passed: int
    errors: list[str]


def extract_main_function(code: str) -> tuple[str, list[str]] | None:
    """Extract the main function name and its parameters from skill code.

    Looks for a function that looks like the skill entry point:
    - Has a docstring mentioning "skill" or "entry"
    - Or is the only top-level function
    - Or is named "run", "execute", "main", "skill"

    Returns (function_name, [param_names]) or None if not found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        return None

    # Prefer functions with skill-like names
    skill_names = {"run", "execute", "main", "skill", "process", "handle"}
    for fn in functions:
        if fn.name.lower() in skill_names:
            params = [arg.arg for arg in fn.args.args]
            return fn.name, params

    # Look for functions with docstrings mentioning skill/entry
    for fn in functions:
        if (
            fn.body
            and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
        ):
            docstring = fn.body[0].value.value
            if isinstance(docstring, str) and any(
                kw in docstring.lower() for kw in ("skill", "entry", "entrypoint")
            ):
                params = [arg.arg for arg in fn.args.args]
                return fn.name, params

    # Fall back to first function
    fn = functions[0]
    params = [arg.arg for arg in fn.args.args]
    return fn.name, params


def synthesize_test_args(param_names: list[str], code: str) -> list[dict]:
    """Synthesize test arguments for a function based on parameter names and type hints.

    Uses heuristics to generate sensible test values.
    """
    # Look for type hints in the code
    type_hints = {}
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    if arg.annotation:
                        type_hints[arg.arg] = ast.unparse(arg.annotation)
    except Exception:
        pass

    test_cases = []

    # Generate 3 test cases with different value patterns
    for i in range(3):
        args = {}
        for param in param_names:
            hint = type_hints.get(param, "").lower()
            if (
                "int" in hint
                or "number" in hint
                or param in ("count", "num", "n", "limit", "max", "min")
            ):
                args[param] = i + 1
            elif "float" in hint:
                args[param] = float(i + 1) * 1.5
            elif "bool" in hint:
                args[param] = i % 2 == 0
            elif "list" in hint or "sequence" in hint or param.endswith("s"):
                args[param] = [f"item{j}" for j in range(i + 1)]
            elif "dict" in hint or "mapping" in hint:
                args[param] = {"key": f"value{i}"}
            elif "path" in hint or "file" in hint:
                args[param] = f"/tmp/test_{i}.txt"
            else:
                # Default to string
                args[param] = f"test_value_{i}"
        test_cases.append(args)

    return test_cases


def build_test_harness(skill_code: str, func_name: str, test_args: dict) -> str:
    """Build a test harness that calls the skill function with test args."""
    # Serialize test args as JSON for safe embedding
    args_json = json.dumps(test_args)

    harness = f"""
{skill_code}

import json
import sys

if __name__ == "__main__":
    test_args = json.loads({json.dumps(args_json)})
    try:
        result = {func_name}(**test_args)
        # Print result as JSON for capture
        if result is not None:
            print(json.dumps(result, default=str))
        else:
            print("null")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {{e}}", file=sys.stderr)
        sys.exit(1)
"""
    return harness


def validate_output_schema(output: str, expected_schema: dict | None = None) -> tuple[bool, str]:
    """Validate output against expected schema.

    If no schema provided, just check that output is valid JSON.
    """
    if not output.strip():
        return False, "Empty output"

    try:
        parsed = json.loads(output.strip())
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON output: {e}"

    if expected_schema is None:
        # Basic validation: output should be a JSON value (not necessarily object)
        return True, "Valid JSON output"

    # Simple schema validation
    def check_type(value: Any, schema: dict, path: str = "") -> tuple[bool, str]:
        expected_type = schema.get("type")
        if expected_type == "object":
            if not isinstance(value, dict):
                return False, f"{path}: expected object, got {type(value).__name__}"
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            for prop in required:
                if prop not in value:
                    return False, f"{path}: missing required property '{prop}'"
            for prop, prop_schema in properties.items():
                if prop in value:
                    ok, msg = check_type(value[prop], prop_schema, f"{path}.{prop}")
                    if not ok:
                        return False, msg
            return True, "OK"
        elif expected_type == "array":
            if not isinstance(value, list):
                return False, f"{path}: expected array, got {type(value).__name__}"
            items_schema = schema.get("items")
            if items_schema:
                for i, item in enumerate(value):
                    ok, msg = check_type(item, items_schema, f"{path}[{i}]")
                    if not ok:
                        return False, msg
            return True, "OK"
        elif expected_type == "string":
            if not isinstance(value, str):
                return False, f"{path}: expected string, got {type(value).__name__}"
            return True, "OK"
        elif expected_type == "number":
            if not isinstance(value, (int, float)):
                return False, f"{path}: expected number, got {type(value).__name__}"
            return True, "OK"
        elif expected_type == "integer":
            if not isinstance(value, int):
                return False, f"{path}: expected integer, got {type(value).__name__}"
            return True, "OK"
        elif expected_type == "boolean":
            if not isinstance(value, bool):
                return False, f"{path}: expected boolean, got {type(value).__name__}"
            return True, "OK"
        elif expected_type == "null":
            if value is not None:
                return False, f"{path}: expected null, got {type(value).__name__}"
            return True, "OK"
        else:
            # Unknown type, accept anything
            return True, "OK"

    ok, msg = check_type(parsed, expected_schema)
    return ok, msg


def run_promotion_gate(
    skill_name: str,
    skill_code: str,
    expected_schema: dict | None = None,
    max_test_cases: int = 3,
    timeout_seconds: float = 30.0,
) -> PromotionGateResult:
    """Run the behavioural promotion gate on a skill.

    Args:
        skill_name: Name of the skill
        skill_code: Python source code of the skill
        expected_schema: Optional JSON schema for output validation
        max_test_cases: Maximum number of test cases to run
        timeout_seconds: Timeout for each sandbox execution

    Returns:
        PromotionGateResult with pass/fail status and details
    """
    # Extract main function
    extracted = extract_main_function(skill_code)
    if not extracted:
        return PromotionGateResult(
            skill_name=skill_name,
            passed=False,
            reason="Could not identify main function in skill code",
            test_cases_run=0,
            test_cases_passed=0,
            errors=["No suitable function found"],
        )

    func_name, param_names = extracted

    # Synthesize test arguments
    test_cases = synthesize_test_args(param_names, skill_code)
    test_cases = test_cases[:max_test_cases]

    if not test_cases:
        return PromotionGateResult(
            skill_name=skill_name,
            passed=False,
            reason="Could not synthesize test arguments",
            test_cases_run=0,
            test_cases_passed=0,
            errors=["No test cases generated"],
        )

    # Run test cases in sandbox
    policy = default_policy()
    policy = policy.__class__(**{**policy.__dict__, "timeout_seconds": int(timeout_seconds)})
    executor = CodeExecutor(policy=policy)

    passed_count = 0
    errors = []

    for i, test_args in enumerate(test_cases):
        harness = build_test_harness(skill_code, func_name, test_args)

        try:
            result: ExecutionResult = executor.execute(harness)
        except Exception as e:
            errors.append(f"Test case {i}: Sandbox execution failed: {e}")
            continue

        if result.exit_code != 0:
            errors.append(
                f"Test case {i}: Exit code {result.exit_code}, stderr: {result.stderr[:200]}"
            )
            continue

        # Validate output
        output = result.stdout.strip()
        ok, msg = validate_output_schema(output, expected_schema)
        if not ok:
            errors.append(f"Test case {i}: Schema validation failed: {msg}")
            continue

        passed_count += 1

    passed = passed_count > 0 and len(errors) == 0
    reason = (
        f"All {passed_count}/{len(test_cases)} test cases passed"
        if passed
        else f"{passed_count}/{len(test_cases)} passed, {len(errors)} failed"
    )

    executor.close()

    return PromotionGateResult(
        skill_name=skill_name,
        passed=passed,
        reason=reason,
        test_cases_run=len(test_cases),
        test_cases_passed=passed_count,
        errors=errors,
    )


def run_promotion_gate_for_tracked_skill(
    skill_name: str,
    expected_schema: dict | None = None,
) -> PromotionGateResult:
    """Run promotion gate for a skill loaded from procedural memory.

    Args:
        skill_name: Name of the skill in procedural memory
        expected_schema: Optional JSON schema for output validation

    Returns:
        PromotionGateResult
    """
    from isaac.memory.manager import get_memory_manager

    mm = get_memory_manager()
    proc = mm.procedural

    # Try to get skill code from procedural memory
    skill_code = None
    if hasattr(proc, "get_code"):
        skill_code = proc.get_code(skill_name)
    elif hasattr(proc, "get_skill"):
        skill = proc.get_skill(skill_name)
        if skill and hasattr(skill, "code"):
            skill_code = skill.code

    if not skill_code:
        return PromotionGateResult(
            skill_name=skill_name,
            passed=False,
            reason=f"Skill '{skill_name}' not found in procedural memory",
            test_cases_run=0,
            test_cases_passed=0,
            errors=["Skill not found"],
        )

    return run_promotion_gate(skill_name, skill_code, expected_schema)


if __name__ == "__main__":
    # Quick self-test
    test_skill = '''
def run(text: str, count: int = 1) -> dict:
    """Echo skill for testing."""
    return {"echo": text * count, "length": len(text) * count}
'''

    result = run_promotion_gate("test_skill", test_skill)
    print(f"Passed: {result.passed}")
    print(f"Reason: {result.reason}")
    print(f"Test cases: {result.test_cases_passed}/{result.test_cases_run}")
    if result.errors:
        print(f"Errors: {result.errors}")
