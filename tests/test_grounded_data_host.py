from src.evidence.grounded_data_host import build_grounded_data_host_report


def test_grounded_data_host_report_requires_gpu_and_sam3d_assets() -> None:
    report = build_grounded_data_host_report(
        subject_id="workcell_grounding",
        subject_kind="scene_tracks_grounded_host",
        host_capabilities={
            "gpu_available": False,
            "opencv_available": True,
            "sam3d_objects_repo_available": True,
            "sam3d_body_repo_available": True,
            "sam3d_objects_checkpoint_available": True,
            "sam3d_body_checkpoint_available": True,
            "real_sam3d_grounding_ready": False,
        },
    )

    assert report.ready is False
    assert "signal_bool::gpu_available" in report.blocking_preconditions
    assert "signal_bool::real_sam3d_grounding_ready" in report.blocking_preconditions
