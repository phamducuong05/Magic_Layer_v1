from dataclasses import FrozenInstanceError

import pytest


def test_job_lifecycle_keeps_progress_monotonic_and_snapshots_immutable():
    from backend.services.process_jobs import ProcessJobStore

    store = ProcessJobStore()
    created = store.create()

    assert created.status == "queued"
    assert created.stage == "queued"
    assert created.progress == 0
    assert created.message == "Waiting to start"

    store.update(created.job_id, "segmentation", 25, "Finding objects")
    store.update(created.job_id, "keywords", 10, "Older update")
    current = store.get(created.job_id)

    assert current.status == "processing"
    assert current.stage == "segmentation"
    assert current.progress == 25
    assert current.message == "Finding objects"
    with pytest.raises(FrozenInstanceError):
        current.progress = 90


def test_job_can_complete_with_result_or_fail_with_error():
    from backend.services.process_jobs import ProcessJobStore

    store = ProcessJobStore()
    completed = store.create()
    failed = store.create()
    payload = {"layers": [{"keyword": "dog"}]}

    store.complete(completed.job_id, payload)
    store.fail(failed.job_id, "Layer extraction failed.")

    assert store.get(completed.job_id).status == "completed"
    assert store.get(completed.job_id).progress == 100
    assert store.get(completed.job_id).result == payload
    assert store.get(failed.job_id).status == "failed"
    assert store.get(failed.job_id).error == "Layer extraction failed."


def test_unknown_job_raises_key_error():
    from backend.services.process_jobs import ProcessJobStore

    with pytest.raises(KeyError, match="missing"):
        ProcessJobStore().get("missing")


def test_store_evicts_old_terminal_jobs_but_never_active_jobs():
    from backend.services.process_jobs import ProcessJobStore

    store = ProcessJobStore(max_jobs=2)
    oldest = store.create()
    active = store.create()
    store.complete(oldest.job_id, {"layers": []})

    newest = store.create()

    with pytest.raises(KeyError, match=oldest.job_id):
        store.get(oldest.job_id)
    assert store.get(active.job_id).status == "queued"
    assert store.get(newest.job_id).status == "queued"
