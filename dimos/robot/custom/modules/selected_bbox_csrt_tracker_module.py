from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Literal

import cv2
from dimos_lcm.std_msgs import Bool, String  # type: ignore[import-untyped]
from dimos_lcm.vision_msgs import (  # type: ignore[import-untyped]
    BoundingBox2D,
    Detection2D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
    Point2D,
    Pose2D,
)
import numpy as np
from reactivex.disposable import Disposable

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.std_msgs.Header import Header
from dimos.msgs.vision_msgs.Detection2DArray import Detection2DArray
from dimos.robot.custom.modules.bbox_selection_module import BBoxSelectionModule
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_DEFAULT_FRAME_ID = "camera_optical"
_OPENCV_RUNTIME_CONFIGURED = False

TrackingState = Literal["unselected", "locked", "searching", "lost"]
BBoxXYXY = tuple[float, float, float, float]
BBoxXYWH = tuple[float, float, float, float]


def _configure_opencv_runtime() -> None:
    """Force OpenCV onto the CPU path before CSRT touches macOS OpenCL runtime."""
    global _OPENCV_RUNTIME_CONFIGURED
    if _OPENCV_RUNTIME_CONFIGURED:
        return

    ocl = getattr(cv2, "ocl", None)
    if ocl is not None:
        set_use_opencl = getattr(ocl, "setUseOpenCL", None)
        if set_use_opencl is not None:
            set_use_opencl(False)

    set_num_threads = getattr(cv2, "setNumThreads", None)
    if set_num_threads is not None:
        set_num_threads(1)

    _OPENCV_RUNTIME_CONFIGURED = True
    use_opencl = None
    if ocl is not None:
        use_opencl_fn = getattr(ocl, "useOpenCL", None)
        if use_opencl_fn is not None:
            use_opencl = bool(use_opencl_fn())
    logger.info(
        "SelectedBBoxCsrtTrackerModule: OpenCV runtime configured "
        f"use_opencl={use_opencl} num_threads=1"
    )


def _create_csrt_tracker() -> Any:
    """Create an OpenCV CSRT tracker across OpenCV package variants."""
    _configure_opencv_runtime()

    factory = getattr(cv2, "TrackerCSRT_create", None)
    if factory is not None:
        return factory()

    legacy = getattr(cv2, "legacy", None)
    legacy_factory = getattr(legacy, "TrackerCSRT_create", None)
    if legacy_factory is not None:
        return legacy_factory()

    raise RuntimeError(
        "OpenCV CSRT tracker is unavailable. Install opencv-contrib-python or a cv2 build with "
        "TrackerCSRT_create."
    )


class SelectedBBoxCsrtTrackerConfig(ModuleConfig):
    tracking_hz: float = 10.0
    max_lost_frames: int = 15
    reacquire_max_center_jump_px: float = 200.0
    action_log_interval_sec: float = 1.0
    tracker_backend: Literal["template", "csrt"] = "template"
    template_search_radius_px: float = 140.0
    template_min_score: float = 0.35


class SelectedBBoxCsrtTrackerModule(Module):
    """Track the user-selected bbox with OpenCV CSRT and YOLOE-assisted reacquisition."""

    config: SelectedBBoxCsrtTrackerConfig

    color_image: In[Image]
    user_selected_bbox: In[Detection2DArray]
    detections: In[Detection2DArray]
    stop_movement: In[Bool]
    clear_selection_request: In[Bool]

    tracked_bbox: Out[Detection2DArray]
    tracking_status: Out[String]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._tracker: Any | None = None
        self._state: TrackingState = "unselected"
        self._latest_image: Image | None = None
        self._latest_image_seq = 0
        self._last_processed_image_seq = 0
        self._latest_detections: Detection2DArray | None = None
        self._last_header: Header | None = None
        self._pending_selected_bbox: Detection2DArray | None = None
        self._target_detector_id: str | None = None
        self._output_target_id: str | None = None
        self._target_class_id: str | None = None
        self._target_score = 0.0
        self._last_bbox_xyxy: BBoxXYXY | None = None
        self._lost_frames = 0
        self._selection_serial = 0
        self._last_action_log_at = 0.0
        self._last_action_log_state: TrackingState | None = None
        self._template_gray: np.ndarray[Any, np.dtype[Any]] | None = None
        self._template_size: tuple[int, int] | None = None

    @rpc
    def start(self) -> None:
        super().start()
        _configure_opencv_runtime()
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_color_image)))
        self.register_disposable(
            Disposable(self.user_selected_bbox.subscribe(self._on_user_selected_bbox))
        )
        self.register_disposable(Disposable(self.detections.subscribe(self._on_detections)))
        self.register_disposable(Disposable(self.stop_movement.subscribe(self._on_stop_movement)))
        self.register_disposable(
            Disposable(self.clear_selection_request.subscribe(self._on_clear_selection_request))
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._tracking_loop,
            name="SelectedBBoxCsrtTrackerModule",
            daemon=True,
        )
        self._thread.start()
        self._publish_status(force=True)
        logger.info(
            "SelectedBBoxCsrtTrackerModule: tracking loop started "
            f"tracking_hz={self.config.tracking_hz} "
            f"max_lost_frames={self.config.max_lost_frames} "
            f"reacquire_max_center_jump_px={self.config.reacquire_max_center_jump_px} "
            f"tracker_backend={self.config.tracker_backend!r}"
        )

    @rpc
    def stop(self) -> None:
        logger.info("SelectedBBoxCsrtTrackerModule: stopping tracking loop")
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(DEFAULT_THREAD_JOIN_TIMEOUT)
        self._clear_tracker(reason="rpc_stop")
        logger.info("SelectedBBoxCsrtTrackerModule: tracking loop stopped")
        super().stop()

    @rpc
    def clear_tracker(self) -> str:
        self._clear_tracker(reason="rpc")
        return "csrt tracker cleared"

    @rpc
    def get_tracking_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "target_id": self._output_target_id,
                "detector_id": self._target_detector_id,
                "target_class_id": self._target_class_id,
                "lost_frames": self._lost_frames,
            }

    def _tracking_loop(self) -> None:
        interval = 1.0 / max(float(self.config.tracking_hz), 0.1)
        while not self._stop_event.wait(interval):
            self._tracking_step()

    def _on_color_image(self, image: Image) -> None:
        with self._lock:
            self._latest_image = image
            self._latest_image_seq += 1
            self._last_header = self._header_from_image(image)

        self._tracking_step()

    def _on_user_selected_bbox(self, selected_bbox: Detection2DArray) -> None:
        detection = self._extract_single_detection(selected_bbox)
        with self._lock:
            self._last_header = selected_bbox.header
            state = self._state
            latest_image = self._latest_image

        if detection is None:
            # BBoxSelectionModule 会在 YOLOE 单帧漏检时发布空 selected_bbox。
            # 一旦 CSRT 已经接管，这个空消息不能再被当作用户取消。
            if state in ("locked", "searching"):
                logger.debug(
                    "SelectedBBoxCsrtTrackerModule: ignoring empty user_selected_bbox "
                    f"while state={state}"
                )
                return
            return

        if latest_image is None:
            with self._lock:
                self._pending_selected_bbox = selected_bbox
            logger.info(
                "SelectedBBoxCsrtTrackerModule: selected bbox received before image; "
                "waiting for color_image"
            )
            return

        self._initialize_from_detection(detection, latest_image, selected_bbox.header)

    def _on_detections(self, detections: Detection2DArray) -> None:
        with self._lock:
            self._latest_detections = detections
            self._last_header = detections.header
            state = self._state

        if state == "searching":
            self._try_reacquire_from_detections()

    def _on_stop_movement(self, msg: Bool) -> None:
        if bool(getattr(msg, "data", False)):
            self._clear_tracker(reason="stop_movement")

    def _on_clear_selection_request(self, msg: Bool) -> None:
        if bool(getattr(msg, "data", False)):
            self._clear_tracker(reason="clear_selection_request")

    def _tracking_step(self) -> None:
        with self._lock:
            pending = self._pending_selected_bbox
            latest_image = self._latest_image
            state = self._state
            image_seq = self._latest_image_seq

        if pending is not None and latest_image is not None:
            detection = self._extract_single_detection(pending)
            if detection is not None:
                self._initialize_from_detection(detection, latest_image, pending.header)
            with self._lock:
                self._pending_selected_bbox = None
            return

        if state == "searching":
            if self._try_reacquire_from_detections():
                return
            with self._lock:
                latest_image = self._latest_image
                image_seq = self._latest_image_seq
                already_processed = image_seq == self._last_processed_image_seq
            if latest_image is not None and not already_processed:
                with self._lock:
                    self._last_processed_image_seq = image_seq
                self._record_tracking_miss(
                    self._header_from_image(latest_image),
                    source="searching",
                )
            return

        with self._lock:
            state = self._state
            latest_image = self._latest_image
            image_seq = self._latest_image_seq
            already_processed = image_seq == self._last_processed_image_seq

        if state != "locked" or latest_image is None or already_processed:
            return

        frame = self._image_to_bgr(latest_image)
        ok, bbox_xyxy = self._update_tracker(frame)
        with self._lock:
            if image_seq == self._latest_image_seq:
                self._last_processed_image_seq = image_seq

        if ok:
            bbox_xyxy = self._clamp_bbox_xyxy(
                bbox_xyxy,
                frame.shape[1],
                frame.shape[0],
            )
            self._publish_tracked_detection(bbox_xyxy, self._header_from_image(latest_image))
            self._set_locked_bbox(bbox_xyxy)
            self._set_state("locked")
            self._maybe_log_action(bbox_xyxy)
            return

        self._record_tracking_miss(self._header_from_image(latest_image), source="csrt_update")

    def _initialize_from_detection(
        self,
        detection: Any,
        image: Image,
        header: Header | None,
        *,
        preserve_output_id: bool = False,
    ) -> bool:
        frame = self._image_to_bgr(image)
        bbox_xyxy = self._clamp_bbox_xyxy(
            BBoxSelectionModule._bbox_corners(detection),
            frame.shape[1],
            frame.shape[0],
        )
        bbox_xywh = self._xyxy_to_xywh(bbox_xyxy)
        if bbox_xywh[2] <= 1.0 or bbox_xywh[3] <= 1.0:
            logger.info(
                "SelectedBBoxCsrtTrackerModule: ignoring invalid selected bbox "
                f"bbox=({bbox_xyxy[0]:.1f}, {bbox_xyxy[1]:.1f}, {bbox_xyxy[2]:.1f}, "
                f"{bbox_xyxy[3]:.1f})"
            )
            return False

        tracker: Any | None = None
        template_gray: np.ndarray[Any, np.dtype[Any]] | None = None
        if self.config.tracker_backend == "csrt":
            tracker = _create_csrt_tracker()
            init_ok = tracker.init(frame, bbox_xywh)
            if init_ok is False:
                logger.info("SelectedBBoxCsrtTrackerModule: CSRT init failed")
                return False
        else:
            template_gray = self._extract_template_gray(frame, bbox_xyxy)
            if template_gray is None:
                logger.info("SelectedBBoxCsrtTrackerModule: template init failed")
                return False
            init_ok = True
        if init_ok is False:
            return False

        detector_id = self._stable_detector_id(detection)
        class_id = self._detection_class_id(detection)
        score, _ = BBoxSelectionModule._best_result(detection)

        with self._lock:
            self._tracker = tracker
            self._template_gray = template_gray
            self._template_size = (
                round(bbox_xywh[2]),
                round(bbox_xywh[3]),
            )
            self._target_detector_id = detector_id
            if not preserve_output_id or self._output_target_id is None:
                self._selection_serial += 1
                self._output_target_id = detector_id or f"csrt-{self._selection_serial}"
            self._target_class_id = class_id
            self._target_score = score
            self._last_bbox_xyxy = bbox_xyxy
            self._lost_frames = 0
            self._last_processed_image_seq = self._latest_image_seq
            self._pending_selected_bbox = None

        self._publish_tracked_detection(bbox_xyxy, header)
        self._set_state("locked")
        logger.info(
            "SelectedBBoxCsrtTrackerModule: tracker initialized "
            f"target_id={self._output_target_id!r} detector_id={detector_id!r} "
            f"class_id={class_id!r} backend={self.config.tracker_backend!r} "
            f"bbox=({bbox_xyxy[0]:.1f}, {bbox_xyxy[1]:.1f}, "
            f"{bbox_xyxy[2]:.1f}, {bbox_xyxy[3]:.1f})"
        )
        return True

    def _record_tracking_miss(self, header: Header | None, *, source: str) -> None:
        with self._lock:
            self._tracker = None
            self._template_gray = None
            self._template_size = None
            self._lost_frames += 1
            lost_frames = self._lost_frames
            max_lost_frames = max(int(self.config.max_lost_frames), 0)

        if lost_frames > max_lost_frames:
            self.tracked_bbox.publish(self._empty_detection_array(header))
            self._set_state("lost", force=True)
            logger.info(
                "SelectedBBoxCsrtTrackerModule: target lost "
                f"target_id={self._output_target_id!r} lost_frames={lost_frames} "
                f"max_lost_frames={max_lost_frames}"
            )
            return

        self.tracked_bbox.publish(self._empty_detection_array(header))
        self._set_state("searching", force=True)
        logger.info(
            "SelectedBBoxCsrtTrackerModule: tracking miss; searching "
            f"target_id={self._output_target_id!r} lost_frames={lost_frames} "
            f"max_lost_frames={max_lost_frames} source={source}"
        )
        self._try_reacquire_from_detections()

    def _try_reacquire_from_detections(self) -> bool:
        with self._lock:
            detections = self._latest_detections
            latest_image = self._latest_image
            last_bbox = self._last_bbox_xyxy
            state = self._state

        if state != "searching" or detections is None or latest_image is None:
            return False

        candidate = self._select_reacquire_candidate(detections, last_bbox)
        if candidate is None:
            return False

        if self._initialize_from_detection(
            candidate,
            latest_image,
            detections.header,
            preserve_output_id=True,
        ):
            logger.info(
                "SelectedBBoxCsrtTrackerModule: reacquired target "
                f"target_id={self._output_target_id!r} "
                f"candidate_detector_id={self._stable_detector_id(candidate)!r}"
            )
            return True

        return False

    def _select_reacquire_candidate(
        self,
        detections: Detection2DArray,
        last_bbox: BBoxXYXY | None,
    ) -> Any | None:
        candidates = list(detections.detections)
        if not candidates:
            return None

        with self._lock:
            target_detector_id = self._target_detector_id
            target_class_id = self._target_class_id

        if target_detector_id is not None:
            id_matches = [
                candidate
                for candidate in candidates
                if self._stable_detector_id(candidate) == target_detector_id
            ]
            if id_matches:
                return self._nearest_candidate_in_search_radius(id_matches, last_bbox)

        if target_class_id is not None:
            class_matches = [
                candidate
                for candidate in candidates
                if self._detection_class_id(candidate) == target_class_id
            ]
            if class_matches:
                candidates = class_matches

        return self._nearest_candidate_in_search_radius(candidates, last_bbox)

    def _nearest_candidate_in_search_radius(
        self,
        candidates: list[Any],
        last_bbox: BBoxXYXY | None,
    ) -> Any | None:
        if not candidates:
            return None
        if last_bbox is None:
            return candidates[0]

        last_center = self._bbox_center(last_bbox)
        ranked = sorted(
            candidates,
            key=lambda candidate: self._center_distance_sq(
                last_center,
                self._detection_center(candidate),
            ),
        )
        best = ranked[0]
        distance = math.sqrt(
            self._center_distance_sq(last_center, self._detection_center(best))
        )
        if distance > float(self.config.reacquire_max_center_jump_px):
            logger.debug(
                "SelectedBBoxCsrtTrackerModule: rejecting reacquire candidate too far "
                f"distance_px={distance:.1f} "
                f"limit_px={self.config.reacquire_max_center_jump_px:.1f}"
            )
            return None

        return best

    def _set_locked_bbox(self, bbox_xyxy: BBoxXYXY) -> None:
        with self._lock:
            self._last_bbox_xyxy = bbox_xyxy
            self._lost_frames = 0

    def _set_state(self, state: TrackingState, *, force: bool = False) -> None:
        with self._lock:
            old_state = self._state
            self._state = state
        self._publish_status(force=force or old_state != state)

    def _clear_tracker(self, reason: str) -> None:
        with self._lock:
            header = self._last_header
            had_target = self._output_target_id is not None or self._state != "unselected"
            self._tracker = None
            self._template_gray = None
            self._template_size = None
            self._state = "unselected"
            self._pending_selected_bbox = None
            self._target_detector_id = None
            self._output_target_id = None
            self._target_class_id = None
            self._target_score = 0.0
            self._last_bbox_xyxy = None
            self._lost_frames = 0
            self._last_action_log_at = 0.0
            self._last_action_log_state = None

        self.tracked_bbox.publish(self._empty_detection_array(header))
        self._publish_status(force=True)
        if had_target:
            logger.info(f"SelectedBBoxCsrtTrackerModule: tracker cleared reason={reason}")

    def _publish_tracked_detection(self, bbox_xyxy: BBoxXYXY, header: Header | None) -> None:
        with self._lock:
            target_id = self._output_target_id or "csrt"
            class_id = self._target_class_id
            score = self._target_score

        detection = self._make_detection(target_id, class_id, score, bbox_xyxy, header)
        self.tracked_bbox.publish(
            Detection2DArray(
                detections_length=1,
                header=self._safe_header(header),
                detections=[detection],
            )
        )

    def _publish_status(self, *, force: bool = False) -> None:
        with self._lock:
            payload = {
                "state": self._state,
                "target_id": self._output_target_id,
                "detector_id": self._target_detector_id,
                "target_class_id": self._target_class_id,
                "lost_frames": self._lost_frames,
            }

        if force or payload["state"] in ("unselected", "lost"):
            self.tracking_status.publish(String(json.dumps(payload, ensure_ascii=True)))

    def _maybe_log_action(self, bbox_xyxy: BBoxXYXY) -> None:
        now = time.monotonic()
        with self._lock:
            state = self._state
            should_log = (
                self._last_action_log_state != state
                or now - self._last_action_log_at
                >= max(float(self.config.action_log_interval_sec), 0.0)
            )
            if not should_log:
                return
            self._last_action_log_state = state
            self._last_action_log_at = now
            target_id = self._output_target_id
            detector_id = self._target_detector_id
            lost_frames = self._lost_frames

        logger.info(
            "SelectedBBoxCsrtTrackerModule: tracking "
            f"state={state} target_id={target_id!r} detector_id={detector_id!r} "
            f"bbox=({bbox_xyxy[0]:.1f}, {bbox_xyxy[1]:.1f}, "
            f"{bbox_xyxy[2]:.1f}, {bbox_xyxy[3]:.1f}) lost_frames={lost_frames}"
        )

    @staticmethod
    def _extract_single_detection(detections: Detection2DArray | None) -> Any | None:
        if detections is None or detections.detections_length == 0 or not detections.detections:
            return None
        return detections.detections[0]

    @staticmethod
    def _image_to_bgr(image: Image) -> np.ndarray[Any, np.dtype[Any]]:
        _configure_opencv_runtime()
        return np.ascontiguousarray(image.to_opencv())

    def _update_tracker(self, frame: np.ndarray[Any, np.dtype[Any]]) -> tuple[bool, BBoxXYXY]:
        if self.config.tracker_backend == "csrt":
            with self._lock:
                tracker = self._tracker
            if tracker is None:
                return False, (0.0, 0.0, 0.0, 0.0)
            ok, bbox_xywh = tracker.update(frame)
            if not ok:
                return False, (0.0, 0.0, 0.0, 0.0)
            return True, self._xywh_to_xyxy(tuple(float(v) for v in bbox_xywh))

        return self._update_template_tracker(frame)

    def _update_template_tracker(
        self,
        frame: np.ndarray[Any, np.dtype[Any]],
    ) -> tuple[bool, BBoxXYXY]:
        with self._lock:
            template_gray = self._template_gray
            template_size = self._template_size
            last_bbox = self._last_bbox_xyxy

        if template_gray is None or template_size is None or last_bbox is None:
            return False, (0.0, 0.0, 0.0, 0.0)

        frame_gray = self._to_gray(frame)
        search = self._search_window(last_bbox, frame_gray.shape[1], frame_gray.shape[0])
        x1, y1, x2, y2 = search
        search_gray = frame_gray[y1:y2, x1:x2]
        template_width, template_height = template_size

        if (
            search_gray.shape[1] < template_width
            or search_gray.shape[0] < template_height
            or template_width <= 1
            or template_height <= 1
        ):
            return False, (0.0, 0.0, 0.0, 0.0)

        result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_score, _, max_loc = cv2.minMaxLoc(result)
        if not math.isfinite(float(max_score)) or max_score < float(self.config.template_min_score):
            logger.info(
                "SelectedBBoxCsrtTrackerModule: template match below threshold "
                f"score={max_score:.3f} threshold={self.config.template_min_score:.3f}"
            )
            return False, (0.0, 0.0, 0.0, 0.0)

        match_x = float(x1 + max_loc[0])
        match_y = float(y1 + max_loc[1])
        return True, (
            match_x,
            match_y,
            match_x + float(template_width),
            match_y + float(template_height),
        )

    def _extract_template_gray(
        self,
        frame: np.ndarray[Any, np.dtype[Any]],
        bbox_xyxy: BBoxXYXY,
    ) -> np.ndarray[Any, np.dtype[Any]] | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self._clamp_bbox_int(bbox_xyxy, width, height)
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            return None
        return np.ascontiguousarray(self._to_gray(frame)[y1:y2, x1:x2])

    @staticmethod
    def _to_gray(frame: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
        if frame.ndim == 2:
            return np.ascontiguousarray(frame)
        return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    def _search_window(self, bbox_xyxy: BBoxXYXY, width: int, height: int) -> tuple[int, int, int, int]:
        radius = max(float(self.config.template_search_radius_px), 0.0)
        x1, y1, x2, y2 = bbox_xyxy
        return self._clamp_bbox_int((x1 - radius, y1 - radius, x2 + radius, y2 + radius), width, height)

    @staticmethod
    def _clamp_bbox_int(bbox_xyxy: BBoxXYXY, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = SelectedBBoxCsrtTrackerModule._clamp_bbox_xyxy(
            bbox_xyxy,
            width,
            height,
        )
        return (
            math.floor(x1),
            math.floor(y1),
            math.ceil(x2),
            math.ceil(y2),
        )

    @staticmethod
    def _header_from_image(image: Image) -> Header:
        return Header(float(image.ts), image.frame_id or _DEFAULT_FRAME_ID)

    @staticmethod
    def _safe_header(header: Header | None) -> Header:
        return header if header is not None else Header(time.time(), _DEFAULT_FRAME_ID)

    @classmethod
    def _empty_detection_array(cls, header: Header | None) -> Detection2DArray:
        return Detection2DArray(
            detections_length=0,
            header=cls._safe_header(header),
            detections=[],
        )

    @classmethod
    def _make_detection(
        cls,
        target_id: str,
        class_id: str | None,
        score: float,
        bbox_xyxy: BBoxXYXY,
        header: Header | None,
    ) -> Detection2D:
        x1, y1, x2, y2 = bbox_xyxy
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        results = (
            [
                ObjectHypothesisWithPose(
                    hypothesis=ObjectHypothesis(class_id=class_id, score=score)
                )
            ]
            if class_id is not None
            else []
        )
        return Detection2D(
            id=target_id,
            results_length=len(results),
            header=cls._safe_header(header),
            bbox=BoundingBox2D(
                center=Pose2D(position=Point2D(x=center_x, y=center_y), theta=0.0),
                size_x=max(x2 - x1, 0.0),
                size_y=max(y2 - y1, 0.0),
            ),
            results=results,
        )

    @staticmethod
    def _stable_detector_id(detection: Any) -> str | None:
        detection_id = str(getattr(detection, "id", "")).strip()
        if not detection_id or detection_id == "-1":
            return None
        return detection_id

    @staticmethod
    def _detection_class_id(detection: Any) -> str | None:
        _, class_id = BBoxSelectionModule._best_result(detection)
        return str(class_id) if class_id is not None else None

    @staticmethod
    def _detection_center(detection: Any) -> tuple[float, float]:
        bbox = BBoxSelectionModule._bbox_corners(detection)
        return SelectedBBoxCsrtTrackerModule._bbox_center(bbox)

    @staticmethod
    def _bbox_center(bbox_xyxy: BBoxXYXY) -> tuple[float, float]:
        return (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0, (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0

    @staticmethod
    def _center_distance_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    @staticmethod
    def _xyxy_to_xywh(bbox_xyxy: BBoxXYXY) -> BBoxXYWH:
        x1, y1, x2, y2 = bbox_xyxy
        return x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)

    @staticmethod
    def _xywh_to_xyxy(bbox_xywh: BBoxXYWH) -> BBoxXYXY:
        x, y, width, height = bbox_xywh
        return x, y, x + width, y + height

    @staticmethod
    def _clamp_bbox_xyxy(bbox_xyxy: BBoxXYXY, width: int, height: int) -> BBoxXYXY:
        x1, y1, x2, y2 = bbox_xyxy
        left = min(max(min(x1, x2), 0.0), float(width))
        right = min(max(max(x1, x2), 0.0), float(width))
        top = min(max(min(y1, y2), 0.0), float(height))
        bottom = min(max(max(y1, y2), 0.0), float(height))
        return left, top, right, bottom


__all__ = [
    "SelectedBBoxCsrtTrackerConfig",
    "SelectedBBoxCsrtTrackerModule",
]
