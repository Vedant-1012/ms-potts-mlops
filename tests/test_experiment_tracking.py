from ms_potts.utils.experiment_tracking import ExperimentTracker


def test_start_and_end_run():
    tracker = ExperimentTracker(experiment_name="test_experiment")
    run = tracker.start_run(run_name="test_run")

    assert run is not None
    assert tracker.active_run is not None
    # run_id = run.info.run_id

    tracker.end_run()
    assert tracker.active_run is None


def test_log_params_and_metrics(tmp_path):
    tracker = ExperimentTracker(experiment_name="test_experiment")
    tracker.start_run(run_name="test_log")

    tracker.log_params({"lr": 0.001, "epochs": 5})
    tracker.log_metrics({"accuracy": 0.9, "loss": 0.1}, step=1)

    artifact_path = tmp_path / "test.json"
    artifact_path.write_text('{"hello": "world"}')
    tracker.log_artifact(str(artifact_path))

    tracker.end_run()
