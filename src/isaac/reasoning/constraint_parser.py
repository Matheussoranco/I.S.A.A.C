"""Interpret a bounded symbolic expression language without executing Python."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

FUNCTIONS = frozenset({"And", "Or", "Not", "Implies", "If"})
MAX_CONSTRAINTS = 2048
MAX_VARIABLES = 1024


def parse_constraint(text: str) -> ast.expr:
    """Parse an expression, or one name assignment interpreted as an equation."""
    if not isinstance(text, str) or not text.strip() or len(text) > 4096:
        raise ValueError("A constraint must be a nonempty string of at most 4096 characters")
    tree = ast.parse(text, mode="exec")
    if len(tree.body) != 1:
        raise ValueError("Exactly one expression or equation is required")
    statement = tree.body[0]
    if isinstance(statement, ast.Expr):
        expression = statement.value
    elif (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        expression = ast.Compare(
            left=ast.Name(id=statement.targets[0].id, ctx=ast.Load()),
            ops=[ast.Eq()],
            comparators=[statement.value],
        )
    else:
        raise ValueError("Unsupported constraint statement")
    if sum(1 for _ in ast.walk(expression)) > 256:
        raise ValueError("Constraint exceeds the expression size limit")
    return expression


def validate_name(name: str) -> None:
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise ValueError("Invalid variable name")
    if name in FUNCTIONS:
        raise ValueError("Variable name conflicts with a symbolic function")


def evaluate_constraint(tree: ast.expr, variables: dict[str, Any], z3: Any) -> Any:
    """Only explicit literals, symbols, arithmetic and Boolean constructors exist."""
    binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
    }
    comparisons = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }
    constructors = {name: getattr(z3, name) for name in FUNCTIONS}

    def visit(node: ast.expr, depth: int = 0) -> Any:
        if depth > 32:
            raise ValueError("Constraint exceeds the nesting limit")
        child = lambda item: visit(item, depth + 1)  # noqa: E731
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return z3.BoolVal(node.value)
            if isinstance(node.value, int) and node.value.bit_length() <= 256:
                return z3.IntVal(node.value)
            if isinstance(node.value, float) and math.isfinite(node.value):
                return z3.RealVal(str(node.value))
            raise ValueError("Only bounded numeric and Boolean literals are supported")
        if isinstance(node, ast.Name):
            validate_name(node.id)
            if node.id not in variables:
                raise ValueError(f"Undeclared variable: {node.id}")
            return variables[node.id]
        if isinstance(node, ast.UnaryOp):
            value = child(node.operand)
            if isinstance(node.op, ast.Not):
                return z3.Not(value)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
        if isinstance(node, ast.BoolOp):
            values = [child(v) for v in node.values]
            return z3.And(*values) if isinstance(node.op, ast.And) else z3.Or(*values)
        if isinstance(node, ast.BinOp):
            operation = binary.get(type(node.op))
            if operation is not None:
                return operation(child(node.left), child(node.right))
            if isinstance(node.op, ast.Pow):
                if not (
                    isinstance(node.right, ast.Constant)
                    and type(node.right.value) is int
                    and 0 <= node.right.value <= 8
                ):
                    raise ValueError("Exponent must be an integer from 0 to 8")
                return child(node.left) ** node.right.value
        if isinstance(node, ast.Compare):
            left = child(node.left)
            parts = []
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                operation = comparisons.get(type(op))
                if operation is None:
                    raise ValueError("Unsupported comparison")
                right = child(comparator)
                parts.append(operation(left, right))
                left = right
            return z3.And(*parts)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in constructors or node.keywords:
                raise ValueError("Only positional symbolic constructor calls are supported")
            arity = {"Not": 1, "Implies": 2, "If": 3}.get(name)
            if arity is not None and len(node.args) != arity:
                raise ValueError(f"Invalid argument count for {name}")
            return constructors[name](*[child(arg) for arg in node.args])
        raise ValueError(f"Unsupported expression: {type(node).__name__}")

    result = visit(tree)
    if not z3.is_bool(result):
        raise ValueError("A constraint must produce a Boolean expression")
    return result
