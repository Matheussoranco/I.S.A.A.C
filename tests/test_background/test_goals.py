
import pytest
from pathlib import Path
from isaac.background.goals import add_goal, list_goals, complete_goal, pause_goal, get_goal, due_goals_context

def test_goal_crud(tmp_path):
    # Setup
    isaac_home = tmp_path / ".isaac"
    
    # 1. Add
    goal = add_goal("Test Goal", due="2026-01-01T00:00:00Z", notes="Initial note", isaac_home=isaac_home)
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
