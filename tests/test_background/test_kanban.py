
import pytest
from pathlib import Path
import tempfile
from isaac.background.kanban import KanbanBoard, Task, KanbanError, reset_default_board

@pytest.fixture
def board():
    import shutil
    tmp_dir = Path(tempfile.mkdtemp())
    db_path = tmp_dir / "test_kanban.db"
    board = KanbanBoard(db_path)
    board.init_board()
    yield board
    board.close()
    shutil.rmtree(tmp_dir)

def test_task_lifecycle(board):
    # Create
    t = board.create_task("Test Task", "Test Description", priority=3)
    assert t.title == "Test Task"
    assert t.status == "backlog"
    
    # List
    tasks = board.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].id == t.id
    
    # Assign
    board.assign_task(t.id, "agent-1")
    t_updated = board.get_task(t.id)
    assert t_updated.assignee == "agent-1"
    
    # Claim
    board.claim_task(t.id, "agent-1")
    t_claimed = board.get_task(t.id)
    assert t_claimed.status == "running"
    
    # Heartbeat
    board.heartbeat(t.id, "agent-1")
    
    # Complete
    board.complete_task(t.id, result="Done it!")
    t_done = board.get_task(t.id)
    assert t_done.status == "complete"
    assert len(board.list_comments(t.id)) == 1

def test_dependencies(board):
    t1 = board.create_task("Task 1")
    t2 = board.create_task("Task 2", depends_on=[t1.id])
    
    # t2 should be in backlog (or ready if promoted, but initial is backlog)
    # check if list_ready sees it
    ready = board.list_ready()
    # t1 is ready, t2 is not (depends on t1)
    assert any(task.id == t1.id for task in ready)
    assert not any(task.id == t2.id for task in ready)
    
    # Complete t1
    board.complete_task(t1.id)
    
    # Now t2 should be ready
    ready = board.list_ready()
    assert any(task.id == t2.id for task in ready)

def test_cycle_detection(board):
    t1 = board.create_task("T1")
    t2 = board.create_task("T2", depends_on=[t1.id])
    
    with pytest.raises(KanbanError, match="cycle"):
        board.link_tasks(t1.id, t2.id)

def test_blocking(board):
    t = board.create_task("Block Me")
    board.block_task(t.id, reason="Missing data")
    t_blocked = board.get_task(t.id)
    assert t_blocked.status == "blocked"
    assert t_blocked.blocked_reason == "Missing data"
