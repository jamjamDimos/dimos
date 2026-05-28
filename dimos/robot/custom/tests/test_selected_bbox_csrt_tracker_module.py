# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable, Iterator
import json
from typing import Any

from dimos_lcm.std_msgs import Bool  # type: ignore[import-untyped]
from dimos_lcm.vision_msgs import (  # type: ignore[import-untyped]
    BoundingBox2D,
    Detection2D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
    Point2D,
    Pose2D,
)
import numpy as np
import pytest

from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.std_msgs.Header import Header
from dimos.msgs.vision_msgs.Detection2DArray import Detection2DArray
from dimos.protocol.rpc.spec import RPCSpec
from dimos.robot.custom.modules import selected_bbox_csrt_tracker_module as csrt_module
from dimos.robot.custom.modules.selected_bbox_csrt_tracker_module import (
    SelectedBBoxCsrtTrackerModule,
)


class _NoopRPC(RPCSpec):
    def __init__(
        self,
        *,
        rpc_timeouts: dict[str, float] | None = None,
        default_rpc_timeout: float = 120.0,
    ) -> None:
        self.rpc_timeouts = {} if rpc_timeouts is None else dict(rpc_timeouts)
        self.default_rpc_timeout = default_rpc_timeout

    def serve_module_rpc(self, module: Any, name: str | None = None) -> None:
        pass

    def serve_rpc(self, f: Callable[..., Any], name: str) -> Callable[[], None]:
        return lambda: None

    def call(
        self,
        name: str,
        arguments: tuple[list[Any], dict[str, Any]],
        cb: Callable[[Any], None] | None,
    ) -> Callable[[], None] | None:
        return (lambda: None) if cb is not None else None

    def call_nowait(self, name: str, arguments: tuple[list[Any], dict[str, Any]]) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _FakeTracker:
    def __init__(self, updates: list[tuple[bool, tuple[float, float, float, float]]]) -> None:
        self.updates = list(updates)
        self.init_calls: list[tuple[Any, tuple[float, float, float, float]]] = []
        self.last_bbox: tuple[float, float, float, float] | None = None

    def init(self, frame: Any, bbox: tuple[float, float, float, float]) -> bool:
        self.init_calls.append((frame, bbox))
        self.last_bbox = bbox
        return True

    def update(self, frame: Any) -> tuple[bool, tuple[float, float, float, float]]:
        if self.updates:
            ok, bbox = self.updates.pop(0)
            if ok:
                self.last_bbox = bbox
            return ok, bbox
        assert self.last_bbox is not None
        return True, self.last_bbox


class _FakeTrackerFactory:
    def __init__(self) -> None:
        self.created: list[_FakeTracker] = []
        self._queued_updates: list[list[tuple[bool, tuple[float, float, float, float]]]] = []

    def queue_updates(self, *updates: tuple[bool, tuple[float, float, float, float]]) -> None:
        self._queued_updates.append(list(updates))

    def __call__(self) -> _FakeTracker:
        updates = self._queued_updates.pop(0) if self._queued_updates else []
        tracker = _FakeTracker(updates)
        self.created.append(tracker)
        return tracker


class _FakeOcl:
    def __init__(self) -> None:
        self.set_values: list[bool] = []

    def setUseOpenCL(self, value: bool) -> None:
        self.set_values.append(value)

    def useOpenCL(self) -> bool:
        return bool(self.set_values[-1]) if self.set_values else True


@pytest.fixture()
def module_with_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[SelectedBBoxCsrtTrackerModule, _FakeTrackerFactory]]:
    factory = _FakeTrackerFactory()
    monkeypatch.setattr(csrt_module, "_create_csrt_tracker", factory)
    instance = SelectedBBoxCsrtTrackerModule(
        rpc_transport=_NoopRPC,
        max_lost_frames=2,
        reacquire_max_center_jump_px=80.0,
        action_log_interval_sec=0.0,
        tracker_backend="csrt",
    )
    try:
        yield instance, factory
    finally:
        instance.stop()


def test_configure_opencv_runtime_disables_opencl_and_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ocl = _FakeOcl()
    thread_values: list[int] = []

    monkeypatch.setattr(csrt_module, "_OPENCV_RUNTIME_CONFIGURED", False)
    monkeypatch.setattr(csrt_module.cv2, "ocl", fake_ocl, raising=False)
    monkeypatch.setattr(
        csrt_module.cv2,
        "setNumThreads",
        lambda value: thread_values.append(value),
        raising=False,
    )

    csrt_module._configure_opencv_runtime()

    assert fake_ocl.set_values == [False]
    assert thread_values == [1]


def _make_image(ts: float = 123.0) -> Image:
    return Image.from_numpy(
        np.zeros((100, 100, 3), dtype=np.uint8),
        format=ImageFormat.BGR,
        frame_id="camera_optical",
        ts=ts,
    )


def _make_pattern_image(x: int, y: int, ts: float = 123.0) -> Image:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[y : y + 30, x : x + 30, 0] = 40
    image[y : y + 15, x : x + 30, 1] = 220
    image[y + 15 : y + 30, x : x + 30, 2] = 180
    image[y + 8 : y + 22, x + 8 : x + 22, :] = 255
    return Image.from_numpy(
        image,
        format=ImageFormat.BGR,
        frame_id="camera_optical",
        ts=ts,
    )


def _make_detection(
    detection_id: str,
    class_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Detection2D:
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return Detection2D(
        id=detection_id,
        results_length=1,
        header=Header(123.0, "camera_optical"),
        bbox=BoundingBox2D(
            center=Pose2D(position=Point2D(x=center_x, y=center_y), theta=0.0),
            size_x=x2 - x1,
            size_y=y2 - y1,
        ),
        results=[
            ObjectHypothesisWithPose(hypothesis=ObjectHypothesis(class_id=class_id, score=0.9))
        ],
    )


def _make_array(*detections: Detection2D) -> Detection2DArray:
    return Detection2DArray(
        detections_length=len(detections),
        header=Header(123.0, "camera_optical"),
        detections=list(detections),
    )


def _subscribe_tracked(module: SelectedBBoxCsrtTrackerModule) -> list[Any]:
    received: list[Any] = []
    module.tracked_bbox.subscribe(received.append)
    return received


def _subscribe_status(module: SelectedBBoxCsrtTrackerModule) -> list[Any]:
    received: list[Any] = []
    module.tracking_status.subscribe(received.append)
    return received


def _status_state(msg: Any) -> str:
    return str(json.loads(msg.data)["state"])


def test_default_template_backend_does_not_create_csrt_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_if_called() -> Any:
        raise AssertionError("CSRT factory should not be used by template backend")

    monkeypatch.setattr(csrt_module, "_create_csrt_tracker", _raise_if_called)
    module = SelectedBBoxCsrtTrackerModule(
        rpc_transport=_NoopRPC,
        max_lost_frames=2,
        template_min_score=0.2,
    )
    tracked = _subscribe_tracked(module)
    try:
        module._on_color_image(_make_pattern_image(10, 10))
        module._on_user_selected_bbox(
            _make_array(_make_detection("target", "person", 10.0, 10.0, 40.0, 40.0))
        )
        module._on_color_image(_make_pattern_image(16, 18, ts=124.0))

        assert module.get_tracking_state()["state"] == "locked"
        assert tracked[-1].detections_length == 1
        assert tracked[-1].detections[0].bbox.center.position.x == pytest.approx(31.0)
        assert tracked[-1].detections[0].bbox.center.position.y == pytest.approx(33.0)
    finally:
        module.stop()


def test_selected_bbox_initializes_tracker_and_publishes_tracked_bbox(
    module_with_factory: tuple[SelectedBBoxCsrtTrackerModule, _FakeTrackerFactory],
) -> None:
    module, factory = module_with_factory
    tracked = _subscribe_tracked(module)
    statuses = _subscribe_status(module)

    module._on_color_image(_make_image())
    module._on_user_selected_bbox(
        _make_array(_make_detection("target", "person", 10.0, 10.0, 40.0, 40.0))
    )

    assert module.get_tracking_state()["state"] == "locked"
    assert len(factory.created) == 1
    assert tracked[-1].detections_length == 1
    assert tracked[-1].detections[0].id == "target"
    assert _status_state(statuses[-1]) == "locked"


def test_tracking_success_updates_bbox(
    module_with_factory: tuple[SelectedBBoxCsrtTrackerModule, _FakeTrackerFactory],
) -> None:
    module, factory = module_with_factory
    factory.queue_updates((True, (12.0, 14.0, 34.0, 36.0)))
    tracked = _subscribe_tracked(module)

    module._on_color_image(_make_image())
    module._on_user_selected_bbox(
        _make_array(_make_detection("target", "person", 10.0, 10.0, 40.0, 40.0))
    )
    module._on_color_image(_make_image(ts=124.0))

    detection = tracked[-1].detections[0]
    assert detection.bbox.center.position.x == pytest.approx(29.0)
    assert detection.bbox.center.position.y == pytest.approx(32.0)
    assert detection.bbox.size_x == pytest.approx(34.0)
    assert detection.bbox.size_y == pytest.approx(36.0)


def test_empty_selected_bbox_after_init_does_not_clear_tracker(
    module_with_factory: tuple[SelectedBBoxCsrtTrackerModule, _FakeTrackerFactory],
) -> None:
    module, _ = module_with_factory
    tracked = _subscribe_tracked(module)

    module._on_color_image(_make_image())
    module._on_user_selected_bbox(
        _make_array(_make_detection("target", "person", 10.0, 10.0, 40.0, 40.0))
    )
    published_count = len(tracked)

    module._on_user_selected_bbox(_make_array())

    assert module.get_tracking_state()["state"] == "locked"
    assert len(tracked) == published_count
    assert tracked[-1].detections_length == 1


def test_csrt_failure_enters_searching_not_lost(
    module_with_factory: tuple[SelectedBBoxCsrtTrackerModule, _FakeTrackerFactory],
) -> None:
    module, factory = module_with_factory
    factory.queue_updates((False, (0.0, 0.0, 0.0, 0.0)))
    tracked = _subscribe_tracked(module)
    statuses = _subscribe_status(module)

    module._on_color_image(_make_image())
    module._on_user_selected_bbox(
        _make_array(_make_detection("target", "person", 10.0, 10.0, 40.0, 40.0))
    )
    module._on_color_image(_make_image(ts=124.0))

    assert module.get_tracking_state()["state"] == "searching"
    assert tracked[-1].detections_length == 0
    assert _status_state(statuses[-1]) == "searching"


def test_nearby_yoloe_candidate_reacquires_target(
    module_with_factory: tuple[SelectedBBoxCsrtTrackerModule, _FakeTrackerFactory],
) -> None:
    module, factory = module_with_factory
    factory.queue_updates((False, (0.0, 0.0, 0.0, 0.0)))
    tracked = _subscribe_tracked(module)

    module._on_color_image(_make_image())
    module._on_user_selected_bbox(
        _make_array(_make_detection("target", "person", 10.0, 10.0, 40.0, 40.0))
    )
    module._on_color_image(_make_image(ts=124.0))
    assert module.get_tracking_state()["state"] == "searching"

    module._on_detections(
        _make_array(
            _make_detection("new-id", "person", 12.0, 12.0, 42.0, 42.0),
            _make_detection("far", "person", 80.0, 80.0, 98.0, 98.0),
        )
    )

    state = module.get_tracking_state()
    assert state["state"] == "locked"
    assert state["target_id"] == "target"
    assert state["detector_id"] == "new-id"
    assert len(factory.created) == 2
    assert tracked[-1].detections_length == 1
    assert tracked[-1].detections[0].id == "target"


def test_lost_after_max_lost_frames(
    module_with_factory: tuple[SelectedBBoxCsrtTrackerModule, _FakeTrackerFactory],
) -> None:
    module, factory = module_with_factory
    factory.queue_updates((False, (0.0, 0.0, 0.0, 0.0)))
    tracked = _subscribe_tracked(module)
    statuses = _subscribe_status(module)

    module._on_color_image(_make_image())
    module._on_user_selected_bbox(
        _make_array(_make_detection("target", "person", 10.0, 10.0, 40.0, 40.0))
    )
    module._on_color_image(_make_image(ts=124.0))
    module._on_color_image(_make_image(ts=125.0))
    module._on_color_image(_make_image(ts=126.0))

    assert module.get_tracking_state()["state"] == "lost"
    assert tracked[-1].detections_length == 0
    assert _status_state(statuses[-1]) == "lost"


def test_stop_movement_and_clear_selection_request_clear_tracker(
    module_with_factory: tuple[SelectedBBoxCsrtTrackerModule, _FakeTrackerFactory],
) -> None:
    module, _ = module_with_factory
    tracked = _subscribe_tracked(module)

    module._on_color_image(_make_image())
    module._on_user_selected_bbox(
        _make_array(_make_detection("target", "person", 10.0, 10.0, 40.0, 40.0))
    )
    module._on_stop_movement(Bool(data=True))

    assert module.get_tracking_state()["state"] == "unselected"
    assert tracked[-1].detections_length == 0

    module._on_user_selected_bbox(
        _make_array(_make_detection("target-2", "person", 20.0, 20.0, 50.0, 50.0))
    )
    assert module.get_tracking_state()["state"] == "locked"

    module._on_clear_selection_request(Bool(data=True))

    assert module.get_tracking_state()["state"] == "unselected"
    assert tracked[-1].detections_length == 0
