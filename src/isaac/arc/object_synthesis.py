"""Object-Level Program Synthesis — the Neurosymbolic Core.

Standard grid-level DSL transformations treat the whole grid as one unit.
Many ARC tasks require reasoning about *individual objects* — their properties,
relationships, and how they change between input and output.

This module provides:
1. **Scene graph construction** — represents a grid as a graph of objects with
   symbolic properties (colour, size, shape, position, relationships)
2. **Object-level rule inference** — infers transformation rules at the object
   level from training pairs (e.g. "the red object moves to where the blue
   object is", "objects are recoloured by their size rank")
3. **Code generation** — synthesises executable Python code that implements
   the inferred object-level rules
4. **Integration with the solver** — feeds the synthesis engine as a strategy
   between beam search and LLM synthesis

This is the heart of neurosymbolic AI: neural networks recognise patterns,
symbolic reasoning structures them into explicit, verifiable programs.

Design philosophy (Chollet):
    Objects are the fundamental units of perception.  A system that reasons
    about objects — rather than raw pixels — is doing the kind of abstraction
    that ARC actually tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from isaac.arc.grid_ops import Grid, GridObject, extract_objects
from isaac.arc.priors import (
    ObjectSignature,
    compute_object_signature,
    full_prior_analysis,
    group_objects_by_colour,
    objects_same_shape,
    relative_position,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Scene graph
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ObjectNode:
    """A node in the scene graph representing one object with full symbolic attributes."""

    obj: GridObject
    sig: ObjectSignature
    rank_by_size: int = 0
    """1 = largest, 2 = second largest, etc."""
    rank_by_position: int = 0
    """1 = top-left, reading order."""
    is_unique_colour: bool = False
    """True if this is the only object with this colour."""
    is_unique_shape: bool = False
    """True if this is the only object with this shape."""
    neighbours: list[int] = field(default_factory=list)
    """IDs of adjacent objects (touching or within 1 cell)."""


@dataclass
class SceneGraph:
    """Symbolic scene graph of a grid."""

    grid_shape: tuple[int, int]
    background: int
    nodes: list[ObjectNode]
    """One node per object, sorted by reading order."""
    colour_groups: dict[int, list[int]]
    """colour → list of object IDs with that colour."""
    shape_groups: list[list[int]]
    """Groups of object IDs that share the same shape."""


def build_scene_graph(grid: Grid) -> SceneGraph:
    """Convert a grid into a symbolic scene graph."""
    from isaac.arc.grid_ops import detect_background

    bg = detect_background(grid)
    objects = extract_objects(grid, bg)

    if not objects:
        return SceneGraph(
            grid_shape=grid.shape,
            background=bg,
            nodes=[],
            colour_groups={},
            shape_groups=[],
        )

    sigs = [compute_object_signature(o) for o in objects]

    # Size ranks
    size_order = sorted(range(len(objects)), key=lambda i: objects[i].size, reverse=True)
    size_rank = [0] * len(objects)
    for rank, idx in enumerate(size_order, 1):
        size_rank[idx] = rank

    # Position ranks (reading order: row then col)
    pos_order = sorted(range(len(objects)), key=lambda i: (objects[i].bbox[0], objects[i].bbox[1]))
    pos_rank = [0] * len(objects)
    for rank, idx in enumerate(pos_order, 1):
        pos_rank[idx] = rank

    # Colour groups
    colour_groups: dict[int, list[int]] = {}
    for i, obj in enumerate(objects):
        colour_groups.setdefault(obj.colour, []).append(i)

    # Shape groups
    shape_groups: list[list[int]] = []
    used = [False] * len(objects)
    for i in range(len(objects)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(objects)):
            if not used[j] and objects_same_shape(sigs[i], sigs[j]):
                group.append(j)
                used[j] = True
        shape_groups.append(group)

    # Neighbour detection
    from isaac.arc.priors import are_adjacent
    neighbours: list[list[int]] = [[] for _ in objects]
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if are_adjacent(objects[i], objects[j], gap=1):
                neighbours[i].append(j)
                neighbours[j].append(i)

    nodes: list[ObjectNode] = []
    for i, (obj, sig) in enumerate(zip(objects, sigs)):
        node = ObjectNode(
            obj=obj,
            sig=sig,
            rank_by_size=size_rank[i],
            rank_by_position=pos_rank[i],
            is_unique_colour=len(colour_groups.get(obj.colour, [])) == 1,
            is_unique_shape=all(len(g) == 1 for g in shape_groups if i in g),
            neighbours=neighbours[i],
        )
        nodes.append(node)

    return SceneGraph(
        grid_shape=grid.shape,
        background=bg,
        nodes=nodes,
        colour_groups=colour_groups,
        shape_groups=shape_groups,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Object-level rule inference
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ObjectRule:
    """A transformation rule expressed at the object level."""

    name: str
    description: str
    confidence: float
    code_template: str
    """Python code template implementing the rule (uses solve() signature)."""
    parameters: dict[str, Any] = field(default_factory=dict)


def _infer_recolour_by_size(
    in_scenes: list[SceneGraph], out_scenes: list[SceneGraph]
) -> ObjectRule | None:
    """Detect: objects recoloured by their size rank."""
    if not in_scenes or not out_scenes:
        return None

    rank_colour_maps: list[dict[int, int]] = []
    for in_sg, out_sg in zip(in_scenes, out_scenes):
        if len(in_sg.nodes) != len(out_sg.nodes):
            return None

        # Match by position
        mapping: dict[int, int] = {}
        for in_node in in_sg.nodes:
            best_out = min(
                out_sg.nodes,
                key=lambda on: abs(on.obj.bbox[0] - in_node.obj.bbox[0]) +
                               abs(on.obj.bbox[1] - in_node.obj.bbox[1]),
                default=None,
            )
            if best_out is None:
                return None
            mapping[in_node.rank_by_size] = best_out.obj.colour

        rank_colour_maps.append(mapping)

    # Consistent across all pairs?
    if not all(m == rank_colour_maps[0] for m in rank_colour_maps):
        return None

    rank_to_colour = rank_colour_maps[0]
    if len(rank_to_colour) < 2:
        return None

    code = f"""def solve(grid: 'np.ndarray') -> 'np.ndarray':
    import numpy as np
    from collections import defaultdict
    bg = int(np.bincount(grid.ravel()).argmax())
    objects = []
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if visited[r, c] or grid[r, c] == bg:
                continue
            colour = int(grid[r, c])
            cells = []
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and grid[nr,nc]==colour:
                        visited[nr,nc] = True
                        stack.append((nr,nc))
            objects.append((colour, len(cells), cells))
    rank_to_colour = {rank_to_colour!r}
    objects_by_size = sorted(enumerate(objects), key=lambda x: x[1][1], reverse=True)
    result = np.full_like(grid, bg)
    for rank, (orig_idx, (colour, size, cells)) in enumerate(objects_by_size, 1):
        new_colour = rank_to_colour.get(rank, colour)
        for rr, cc in cells:
            result[rr, cc] = new_colour
    return result
"""
    return ObjectRule(
        name="recolour_by_size_rank",
        description=f"Recolour objects by size rank: {rank_to_colour}",
        confidence=0.85,
        code_template=code,
        parameters={"rank_to_colour": rank_to_colour},
    )


def _infer_move_to_gravity(
    in_scenes: list[SceneGraph], out_scenes: list[SceneGraph]
) -> list[ObjectRule]:
    """Detect gravity-like movement patterns for individual objects."""
    rules: list[ObjectRule] = []

    for direction in ["down", "up", "left", "right"]:
        consistent = True
        for in_sg, out_sg in zip(in_scenes, out_scenes):
            for in_node, out_node in zip(in_sg.nodes, out_sg.nodes):
                if in_node.obj.colour != out_node.obj.colour:
                    consistent = False
                    break
            if not consistent:
                break

        if consistent:
            code = f"""def solve(grid: 'np.ndarray') -> 'np.ndarray':
    import numpy as np
    bg = int(np.bincount(grid.ravel()).argmax())
    result = np.full_like(grid, bg)
    {'for c in range(grid.shape[1]):' if direction in ("down","up") else 'for r in range(grid.shape[0]):'}
        {'col = grid[:, c]; non_bg = col[col != bg]' if direction in ("down","up") else 'row = grid[r, :]; non_bg = row[row != bg]'}
        if len(non_bg) > 0:
            {'result[-len(non_bg):, c] = non_bg' if direction == "down" else
             'result[:len(non_bg), c] = non_bg' if direction == "up" else
             'result[r, :len(non_bg)] = non_bg' if direction == "left" else
             'result[r, -len(non_bg):] = non_bg'}
    return result
"""
            rules.append(ObjectRule(
                name=f"gravity_{direction}",
                description=f"All non-background cells fall {direction}",
                confidence=0.5,
                code_template=code,
                parameters={"direction": direction},
            ))

    return rules


def _infer_copy_pattern(
    in_scenes: list[SceneGraph], out_scenes: list[SceneGraph]
) -> ObjectRule | None:
    """Detect: a template object is copied to positions of marker objects."""
    if not in_scenes:
        return None

    # Heuristic: if there are 2+ colour types and one colour appears in many
    # positions while another is a pattern, the pattern gets copied
    for sg in in_scenes:
        if len(sg.colour_groups) < 2:
            return None

    # This is a complex pattern — flag for LLM synthesis with object hint
    return None  # Placeholder for future implementation


def _infer_unique_colour_rule(
    in_scenes: list[SceneGraph],
    out_scenes: list[SceneGraph],
    train_pairs: list[tuple[Grid, Grid]],
) -> ObjectRule | None:
    """Detect: the unique-coloured object defines the rule for others."""
    # If each pair has exactly one unique-colour object and it stays as anchor
    unique_colours: list[int] = []
    for sg in in_scenes:
        unique = [n for n in sg.nodes if n.is_unique_colour]
        if len(unique) == 1:
            unique_colours.append(unique[0].obj.colour)

    if len(unique_colours) == len(in_scenes) and len(set(unique_colours)) == 1:
        anchor_colour = unique_colours[0]
        return ObjectRule(
            name="unique_colour_anchor",
            description=f"Colour {anchor_colour} is the unique anchor defining the transformation",
            confidence=0.6,
            code_template="",  # Requires LLM with this hint
            parameters={"anchor_colour": anchor_colour},
        )
    return None


def infer_object_rules(
    train_pairs: list[tuple[Grid, Grid]],
) -> list[ObjectRule]:
    """Infer object-level transformation rules from all training pairs.

    Parameters
    ----------
    train_pairs:
        List of (input_grid, output_grid) numpy array pairs.

    Returns
    -------
    list[ObjectRule]
        Ranked list of inferred object-level rules (highest confidence first).
    """
    if not train_pairs:
        return []

    # Build scene graphs for all pairs
    in_scenes = [build_scene_graph(inp) for inp, _ in train_pairs]
    out_scenes = [build_scene_graph(out) for _, out in train_pairs]

    rules: list[ObjectRule] = []

    # Try each rule inference strategy
    r = _infer_recolour_by_size(in_scenes, out_scenes)
    if r:
        rules.append(r)

    rules.extend(_infer_move_to_gravity(in_scenes, out_scenes))

    r = _infer_unique_colour_rule(in_scenes, out_scenes, train_pairs)
    if r:
        rules.append(r)

    rules.sort(key=lambda r: r.confidence, reverse=True)
    return rules


# ─────────────────────────────────────────────────────────────────────────────
# Code validation and candidate generation
# ─────────────────────────────────────────────────────────────────────────────


def _validate_rule_code(
    code: str,
    train_pairs: list[tuple[Grid, Grid]],
) -> float:
    """Return training accuracy (0–1) for a rule's code template."""
    if not code.strip():
        return 0.0
    namespace: dict[str, Any] = {"np": np, "numpy": np}
    try:
        exec(code, namespace)  # noqa: S102
        solve_fn = namespace.get("solve")
        if solve_fn is None:
            return 0.0
        correct = sum(
            1 for inp, expected in train_pairs
            if _safe_equal(solve_fn(inp.copy()), expected)
        )
        return correct / len(train_pairs)
    except Exception:
        return 0.0


def _safe_equal(a: Any, b: Grid) -> bool:
    try:
        return np.array_equal(np.array(a, dtype=int), b)
    except Exception:
        return False


def synthesise_from_object_rules(
    task: "ArcTask",
) -> tuple[str, float] | None:
    """Try to synthesise a solve() via object-level rule inference.

    Returns (code, train_accuracy) if a rule with accuracy > 0 is found,
    else None.
    """
    from isaac.arc.evaluator import ArcTask  # avoid circular at module level

    train_pairs = [(p.input, p.output) for p in task.train]
    rules = infer_object_rules(train_pairs)

    best_code = ""
    best_acc = 0.0

    for rule in rules:
        if not rule.code_template:
            continue
        acc = _validate_rule_code(rule.code_template, train_pairs)
        logger.debug("Object rule '%s': %.0f%% accuracy", rule.name, acc * 100)
        if acc > best_acc:
            best_code = rule.code_template
            best_acc = acc
        if acc == 1.0:
            logger.info("Object synthesis: solved via rule '%s'", rule.name)
            return best_code, best_acc

    if best_code and best_acc > 0.0:
        return best_code, best_acc
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Scene graph → LLM prompt enrichment
# ─────────────────────────────────────────────────────────────────────────────


def describe_scene_for_prompt(sg: SceneGraph) -> list[str]:
    """Convert a scene graph to structured text observations for LLM prompts."""
    lines: list[str] = []
    lines.append(f"Scene: {sg.grid_shape[0]}x{sg.grid_shape[1]}, "
                 f"background={sg.background}, {len(sg.nodes)} objects")

    for node in sg.nodes:
        sig = node.sig
        desc = (
            f"  Object (colour={node.obj.colour}, size={sig.size}, "
            f"bbox={node.obj.bbox}, rank_size={node.rank_by_size}, "
            f"rank_pos={node.rank_by_position}"
        )
        if node.is_unique_colour:
            desc += ", UNIQUE_COLOUR"
        if sig.is_rectangle:
            desc += ", RECTANGLE"
        if sig.is_line:
            desc += f", LINE({sig.line_direction})"
        if node.neighbours:
            desc += f", neighbours={node.neighbours}"
        desc += ")"
        lines.append(desc)

    if sg.shape_groups:
        multi = [g for g in sg.shape_groups if len(g) > 1]
        if multi:
            lines.append(f"  Same-shape groups: {multi}")

    return lines


def build_object_context_for_llm(
    train_pairs: list[tuple[Grid, Grid]],
) -> str:
    """Build a rich object-level context string for the LLM synthesis prompt."""
    lines: list[str] = ["## Object-level scene analysis"]
    rules = infer_object_rules(train_pairs)

    for i, (inp, out) in enumerate(train_pairs[:3]):
        in_sg = build_scene_graph(inp)
        out_sg = build_scene_graph(out)
        lines.append(f"\n### Pair {i + 1} — Input scene:")
        lines.extend(describe_scene_for_prompt(in_sg))
        lines.append(f"### Pair {i + 1} — Output scene:")
        lines.extend(describe_scene_for_prompt(out_sg))

    if rules:
        lines.append("\n### Inferred object-level rules:")
        for rule in rules[:5]:
            lines.append(
                f"  [{rule.confidence:.0%}] {rule.name}: {rule.description}"
            )

    return "\n".join(lines)
