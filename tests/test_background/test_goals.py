from isaac.background.goals import (
    add_goal,
    complete_goal,
    due_goals_context,
    get_goal,
    list_goals,
    mark_goal_progress,
    pause_goal,
)


def test_goal_crud(tmp_path):
    # Setup
    isaac_home = tmp_path / ".isaac"

    # 1. Add
    goal = add_goal(
        "Test Goal", due="2026-01-01T00:00:00Z", notes="Initial note", isaac_home=isaac_home
    )
    assert goal.text == "Test Goal"
    assert goal.id is not None

    # 2. List
    goals = list_goals(isaac_home=isaac_home)
    assert len(goals) == 1
    assert goals[0].id == goal.id

    # 3. Get
    fetched = get_goal(goal.id, isaac_home=isaac_home)
    assert fetched is not None
    assert fetched.text == "Test Goal"

    # 4. Pause
    assert pause_goal(goal.id, isaac_home=isaac_home) is True
    assert get_goal(goal.id, isaac_home=isaac_home).status == "paused"

    # 5. Complete
    assert complete_goal(goal.id, isaac_home=isaac_home) is True
    assert get_goal(goal.id, isaac_home=isaac_home).status == "completed"


def test_due_goals_context(tmp_path):
    isaac_home = tmp_path / ".isaac"

    # Active goal (no due date) -> should be in context
    add_goal("Active Goal", isaac_home=isaac_home)

    # Completed goal -> should NOT be in context
    g2 = add_goal("Completed Goal", isaac_home=isaac_home)
    complete_goal(g2.id, isaac_home=isaac_home)

    ctx = due_goals_context(isaac_home=isaac_home)
    assert "Active Goal" in ctx
    assert "Completed Goal" not in ctx


def test_progress_marks_accumulate_atomically(tmp_path):
    isaac_home = tmp_path / ".isaac"
    goal = add_goal("Long project", isaac_home=isaac_home)

    for i in range(5):
        assert mark_goal_progress(goal.id, f"step {i}", isaac_home=isaac_home)

    saved = get_goal(goal.id, isaac_home=isaac_home)
    assert saved is not None
    assert saved.progress == 5
    assert saved.notes.count("step ") == 5


def test_goal_prefix_lookup_does_not_treat_wildcards(tmp_path):
    isaac_home = tmp_path / ".isaac"
    first = add_goal("First", isaac_home=isaac_home)
    add_goal("Second", isaac_home=isaac_home)

    assert get_goal("_", isaac_home=isaac_home) is None
    assert get_goal("%", isaac_home=isaac_home) is None
    assert complete_goal("%", isaac_home=isaac_home) is False
    assert get_goal(first.id, isaac_home=isaac_home).status == "active"
