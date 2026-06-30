"""Tests for stale requirement-job reconciliation.

Reproduces the bug where a job left in 'running' state by a previous process
caused the Streamlit UI to poll forever (reloading every few seconds), which
wiped the transient "document processed" banners.
"""

from unittest.mock import patch

import core.requirement_jobs as rj


def _job(job_id, status):
    return {"id": job_id, "user_id": 1, "status": status}


def test_reconcile_marks_stale_running_job_as_failed():
    """A 'running' job with no live worker in this process must be failed."""
    jobs = [_job("stale-running", "running"), _job("done", "completed")]

    with patch.object(rj, "db_list_requirement_jobs", return_value=jobs), \
         patch.object(rj, "update_requirement_job") as mock_update, \
         patch.object(rj.RequirementJobManager, "instance") as mock_instance:
        mock_instance.return_value.active_job_ids.return_value = set()

        count = rj.reconcile_stale_jobs(user_id=1)

    assert count == 1
    mock_update.assert_called_once()
    args, kwargs = mock_update.call_args
    assert args[0] == "stale-running"
    assert kwargs["status"] == "failed"


def test_reconcile_marks_stale_queued_job_as_failed():
    jobs = [_job("stale-queued", "queued")]

    with patch.object(rj, "db_list_requirement_jobs", return_value=jobs), \
         patch.object(rj, "update_requirement_job") as mock_update, \
         patch.object(rj.RequirementJobManager, "instance") as mock_instance:
        mock_instance.return_value.active_job_ids.return_value = set()

        count = rj.reconcile_stale_jobs(user_id=1)

    assert count == 1
    assert mock_update.call_args.kwargs["status"] == "failed"


def test_reconcile_preserves_genuinely_active_job():
    """A running job WITH a live worker in this process must not be touched."""
    jobs = [_job("live-running", "running")]

    with patch.object(rj, "db_list_requirement_jobs", return_value=jobs), \
         patch.object(rj, "update_requirement_job") as mock_update, \
         patch.object(rj.RequirementJobManager, "instance") as mock_instance:
        mock_instance.return_value.active_job_ids.return_value = {"live-running"}

        count = rj.reconcile_stale_jobs(user_id=1)

    assert count == 0
    mock_update.assert_not_called()


def test_reconcile_ignores_terminal_jobs():
    jobs = [_job("c", "completed"), _job("f", "failed")]

    with patch.object(rj, "db_list_requirement_jobs", return_value=jobs), \
         patch.object(rj, "update_requirement_job") as mock_update, \
         patch.object(rj.RequirementJobManager, "instance") as mock_instance:
        mock_instance.return_value.active_job_ids.return_value = set()

        count = rj.reconcile_stale_jobs(user_id=1)

    assert count == 0
    mock_update.assert_not_called()


def test_is_job_active_reflects_worker_presence():
    with patch.object(rj.RequirementJobManager, "instance") as mock_instance:
        mock_instance.return_value.active_job_ids.return_value = {"abc"}
        assert rj.is_job_active("abc") is True
        assert rj.is_job_active("xyz") is False
        assert rj.is_job_active(None) is False


def test_reconcile_no_user_returns_zero():
    assert rj.reconcile_stale_jobs(user_id=None) == 0
