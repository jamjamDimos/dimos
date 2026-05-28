from __future__ import annotations  # 允许类型注解延迟解析，减少循环导入风险

import json  # 导入 JSON 工具，用于发布结构化行为状态
import math  # 导入数学工具，用于检查有限数值
import threading  # 导入线程工具，用后台循环发布速度命令
import time  # 导入单调时钟工具，用于 dwell / standoff 阶段计时
from typing import Any, Literal  # 导入通用类型和状态字面量类型

from dimos_lcm.sensor_msgs import (
    CameraInfo as DimosLcmCameraInfo,  # type: ignore[import-untyped]  # 导入 LCM CameraInfo 类型，用于 Detection3DPC.from_2d()
)
from dimos_lcm.std_msgs import Bool, String  # type: ignore[import-untyped]  # 导入 LCM 消息类型
from reactivex.disposable import Disposable  # 导入 Disposable，用于注册输入流订阅

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT  # 导入线程停止等待的默认超时时间
from dimos.core.core import rpc  # 导入 rpc 装饰器，让方法可通过 DimOS RPC 调用
from dimos.core.module import Module, ModuleConfig  # 导入模块基类和模块配置基类
from dimos.core.stream import In, Out  # 导入输入输出流类型
from dimos.msgs.geometry_msgs.Transform import Transform  # 导入 TF 变换类型，用于距离估计
from dimos.msgs.geometry_msgs.Twist import Twist  # 导入速度命令消息类型
from dimos.msgs.geometry_msgs.Vector3 import Vector3  # 导入三维向量类型，用于构造 Twist
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo  # 导入相机内参消息类型
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2  # 导入 lidar 点云消息类型
from dimos.msgs.vision_msgs.Detection2DArray import Detection2DArray  # 导入 2D 检测数组消息类型
from dimos.navigation.replanning_a_star.module_spec import (
    ReplanningAStarPlannerSpec,  # 导入规划器 Spec，用于 bbox 任务启动时取消 A* 导航
)
from dimos.perception.detection.type.detection2d.bbox import (
    Detection2DBBox,  # 导入 2D bbox 类型，用于构造 Detection3DPC 输入
)
from dimos.perception.detection.type.detection3d.pointcloud import (
    Detection3DPC,  # 导入 3D 点云检测，用于精确距离估计
)
from dimos.robot.custom.modules.bbox_selection_module import (
    BBoxSelectionModule,  # 导入 bbox 选择模块，复用 _bbox_corners 工具
)
from dimos.utils.logging_config import setup_logger  # 导入日志初始化函数

logger = setup_logger()  # 创建当前文件使用的日志对象

BehaviorState = Literal[
    "idle",
    "approaching",
    "dwelling_near",
    "retreating",
    "standing_off",
    "returning",
    "done",
]  # 定义行为状态机的合法状态

_LINEAR_GAIN = 0.8  # 定义距离误差到线速度的简单比例增益
_ANGULAR_GAIN = 1.0  # 定义横向像素误差到角速度的简单比例增益
_DISTANCE_TOLERANCE_M = 0.05  # 定义靠近完成时使用的距离容差


def _safe_track_id(raw: Any) -> int:  # 安全地把 detection.id 转换为 int，无法转换时返回 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


class BBoxDistanceBehaviorConfig(ModuleConfig):  # 定义 bbox 距离行为模块配置
    command_hz: float = 20.0  # 速度命令发布频率，单位 Hz
    near_distance: float = 0.5  # 靠近/返回阶段目标距离，单位米
    approach_distance: float | None = None  # 兼容旧配置名；设置后会覆盖 near_distance
    dwell_duration_sec: float = 10.0  # 到达近距离后原地停留的时间，单位秒
    standoff_distance: float = 1.5  # 后退和守候阶段要求保持的最小距离，单位米
    standoff_duration_sec: float = 45.0  # 保持远距离等待的时间，单位秒
    max_linear_speed: float = 0.45  # 最大线速度，单位 m/s
    max_angular_speed: float = 0.8  # 最大角速度，单位 rad/s
    tf_time_tolerance: float = 5.0  # TF 查询时间容差，单位秒；和现有 3D 检测模块保持一致
    prefer_latest_tf: bool = True  # 优先使用最新 TF，避免 replay/live 时间戳漂移导致动作卡死
    action_log_interval_sec: float = 1.0  # 动作执行日志最小间隔，避免 command_hz 级别刷屏


class BBoxDistanceBehaviorModule(
    Module
):  # 定义距离行为模块，只消费 selected bbox、lidar 和 camera_info
    config: BBoxDistanceBehaviorConfig  # 声明本模块使用的配置类型
    selected_bbox: In[Detection2DArray]  # 输入选择模块发布的单 bbox Detection2DArray
    lidar: In[PointCloud2]  # 输入 Go2 lidar 点云
    camera_info: In[CameraInfo]  # 输入 Go2 相机内参
    teleop_active: In[Bool]  # 输入遥控信号；有值时中断当前任务，交回键盘控制
    lock_status: In[String]  # 输入目标锁/追踪状态；没有发布者的 blueprint 中此流可以为空
    cmd_vel: Out[Twist]  # 输出 Go2 速度命令（有 MovementManager 时连接到 nav_cmd_vel）
    clear_selection_request: Out[Bool]  # 输出 task 完成/停止后的清除请求，防止旧 bbox 重启 task
    behavior_status: Out[String]  # 输出行为状态，便于日志和调试
    _planner: (
        ReplanningAStarPlannerSpec | None
    )  # 可选：A* 规划器，存在时用于 bbox 任务启动时取消当前导航

    def __init__(self, **kwargs: Any) -> None:  # 定义构造函数，接收框架传入的配置参数
        super().__init__(**kwargs)  # 调用父类构造函数，让 DimOS 初始化模块和流
        self._lock = threading.RLock()  # 创建递归锁，保护传感器缓存和状态机
        self._stop_event = threading.Event()  # 创建后台命令循环停止事件
        self._thread: threading.Thread | None = None  # 保存后台命令循环线程
        self._state: BehaviorState = "idle"  # 初始化行为状态为 idle
        self._active_target_id: str | None = None  # 保存当前自动靠近的目标 id
        self._latest_selected_bbox: Detection2DArray | None = None  # 保存最新 selected bbox
        self._latest_lidar: PointCloud2 | None = None  # 保存最新 lidar 点云
        self._latest_camera_info: CameraInfo | None = None  # 保存最新 camera_info
        self._latest_lock_state: str | None = None  # 保存 tracker 最新状态，用于区分搜索和用户清空
        self._active_near_distance = self._configured_near_distance()  # 保存本次行为使用的近距离
        self._active_dwell_duration_sec = self.config.dwell_duration_sec  # 保存本次近距离停留时间
        self._active_standoff_distance = self.config.standoff_distance  # 保存本次守候距离
        self._active_standoff_duration_sec = self.config.standoff_duration_sec  # 保存本次远距离守候时间
        self._phase_started_at: float | None = None  # 保存当前计时阶段开始时间
        self._last_block_reason: str | None = None  # 记录上一次阻塞原因，避免重复刷日志
        self._last_action_log_at = 0.0  # 记录上次动作日志时间，用于节流
        self._last_action_log_state: BehaviorState | None = None  # 记录上次动作日志状态
        self._planner: ReplanningAStarPlannerSpec | None = (
            None  # 可选规划器引用，由 blueprint 在 build 时注入
        )

    @rpc  # 标记 start() 是框架生命周期 RPC
    def start(self) -> None:  # 定义模块启动逻辑
        super().start()  # 启动父类逻辑，包括 RPC 和自动绑定
        self.register_disposable(
            Disposable(self.selected_bbox.subscribe(self._on_selected_bbox))
        )  # 订阅 selected bbox
        self.register_disposable(
            Disposable(self.lidar.subscribe(self._on_lidar))
        )  # 订阅 lidar 点云
        self.register_disposable(
            Disposable(self.camera_info.subscribe(self._on_camera_info))
        )  # 订阅 camera_info
        self.register_disposable(
            Disposable(self.teleop_active.subscribe(self._on_teleop_active))
        )  # 订阅遥控打断信号
        self.register_disposable(
            Disposable(self.lock_status.subscribe(self._on_lock_status))
        )  # 订阅目标锁/追踪状态，没有发布者时不会收到消息
        self._stop_event.clear()  # 清除停止事件，允许后台线程运行
        self._thread = threading.Thread(  # 创建后台命令发布线程
            target=self._command_loop,  # 指定线程执行固定频率控制循环
            name="BBoxDistanceBehaviorModule",  # 给线程命名，方便日志和调试
            daemon=True,  # 设置守护线程，进程退出时不会被它卡住
        )  # 结束线程参数
        self._thread.start()  # 启动后台命令发布线程
        logger.info(
            "BBoxDistanceBehaviorModule: task loop started "
            f"command_hz={self.config.command_hz} near_distance={self._configured_near_distance()} "
            f"dwell_duration_sec={self.config.dwell_duration_sec} "
            f"standoff_distance={self.config.standoff_distance} "
            f"standoff_duration_sec={self.config.standoff_duration_sec}"
        )
        self._publish_status("idle")  # 发布初始状态

    @rpc  # 标记 stop() 是框架生命周期 RPC
    def stop(self) -> None:  # 定义模块停止逻辑
        logger.info("BBoxDistanceBehaviorModule: stopping task loop")
        self._stop_event.set()  # 通知后台线程退出
        self.cmd_vel.publish(Twist.zero())  # 停止时立即发布零速度
        if self._thread is not None and self._thread.is_alive():  # 如果后台线程存在且仍在运行
            self._thread.join(DEFAULT_THREAD_JOIN_TIMEOUT)  # 等待后台线程退出，但最多等默认超时
        logger.info("BBoxDistanceBehaviorModule: task loop stopped")
        super().stop()  # 调用父类停止逻辑，释放订阅和 transport

    @rpc  # 标记 start_bbox_distance_behavior() 可通过 DimOS RPC 调用
    def start_bbox_distance_behavior(  # 定义行为启动 RPC
        self,  # 传入模块实例
        approach_distance: float | None = None,  # 可选覆盖靠近阶段目标距离
        near_distance: float | None = None,  # 可选覆盖近距离；优先级高于旧的 approach_distance
        dwell_duration_sec: float | None = None,  # 可选覆盖近距离停留时间
        standoff_distance: float | None = None,  # 可选覆盖后退/守候距离
        standoff_duration_sec: float | None = None,  # 可选覆盖远距离守候时间
    ) -> str:  # 返回可读启动结果
        with self._lock:  # 加锁重置状态机参数
            self._active_near_distance = self._resolve_active_near_distance(
                near_distance=near_distance,
                approach_distance=approach_distance,
            )  # 设置本次近距离
            self._active_dwell_duration_sec = self._resolve_duration(
                dwell_duration_sec, self.config.dwell_duration_sec
            )  # 设置本次近距离停留时间
            self._active_standoff_distance = self._resolve_distance(
                standoff_distance, self.config.standoff_distance
            )  # 设置本次远距离守候距离
            self._active_standoff_duration_sec = self._resolve_duration(
                standoff_duration_sec, self.config.standoff_duration_sec
            )  # 设置本次远距离守候时间
            self._state = "approaching"  # 进入靠近阶段
            self._phase_started_at = None  # 靠近阶段无需计时，清空阶段开始时间

        logger.info(
            "BBoxDistanceBehaviorModule: task started by RPC "
            f"near_distance={self._active_near_distance} "
            f"dwell_duration_sec={self._active_dwell_duration_sec} "
            f"standoff_distance={self._active_standoff_distance} "
            f"standoff_duration_sec={self._active_standoff_duration_sec}"
        )

        self._publish_status("approaching")  # 发布状态变化
        return "bbox distance sequence started"  # 返回启动确认

    @rpc  # 标记 stop_bbox_distance_behavior() 可通过 DimOS RPC 调用
    def stop_bbox_distance_behavior(self) -> str:  # 定义行为停止 RPC
        with self._lock:  # 加锁更新状态机
            previous_target_id = self._active_target_id
            self._state = "idle"  # 回到 idle 状态
            self._active_target_id = None  # 清空当前目标 id
            self._phase_started_at = None  # 清空阶段计时
            self._reset_action_log()  # 重置动作日志节流状态

        self.cmd_vel.publish(Twist.zero())  # 行为停止时发布零速度
        self._request_selection_clear(reason="rpc_stop")  # 清掉旧 bbox，避免下一帧自动重启 task
        logger.info(
            f"BBoxDistanceBehaviorModule: task stopped by RPC target_id={previous_target_id!r}"
        )
        self._publish_status("idle")  # 发布 idle 状态
        return "bbox distance behavior stopped"  # 返回停止确认

    def _on_selected_bbox(self, selected_bbox: Detection2DArray) -> None:  # 处理新的 selected bbox
        with self._lock:  # 加锁更新缓存
            self._latest_selected_bbox = selected_bbox  # 保存最新 selected bbox
            detection = self._extract_single_detection(selected_bbox)  # 读取当前选中目标
            current_target_id = self._active_target_id  # 复制当前目标 id
            current_state = self._state  # 复制当前状态
            lock_state = self._latest_lock_state  # 复制目标锁/追踪状态

        if detection is None:  # 如果当前没有选中目标
            if self._should_reset_on_empty_selection(lock_state):  # 无 tracker 或明确清空时重置任务
                with self._lock:  # 加锁重置状态
                    previous_target_id = self._active_target_id
                    self._active_target_id = None  # 清空当前目标 id
                    self._state = "idle"  # 回到 idle
                    self._phase_started_at = None  # 清空阶段计时
                    self._reset_action_log()  # 重置动作日志节流状态

                if previous_target_id is not None or current_state != "idle":
                    self.cmd_vel.publish(Twist.zero())  # 从活动任务退出时立即停止一次
                    logger.info(
                        "BBoxDistanceBehaviorModule: task ended because selection cleared "
                        f"previous_target_id={previous_target_id!r}"
                    )
                    self._publish_status("idle")  # 发布 idle 状态
            return  # 结束处理

        target_id = self._detection_id(detection)  # 读取当前目标 id
        with self._lock:  # 加锁更新当前目标
            self._active_target_id = target_id  # 保存当前目标 id

        if (
            current_state == "done" and current_target_id == target_id
        ):  # 如果已经完成且仍是同一个目标
            self._publish_status("done")  # 保持完成状态
            return  # 不重新启动任务

        if current_state not in ("idle", "done") and current_target_id == target_id:
            return  # 同一目标的完整序列已经在运行，不被后续 bbox 帧重置阶段计时

        with self._lock:  # 加锁切换到靠近状态
            self._active_near_distance = self._configured_near_distance()  # 新目标使用当前配置近距离
            self._active_dwell_duration_sec = self.config.dwell_duration_sec  # 新目标使用当前配置停留时间
            self._active_standoff_distance = self.config.standoff_distance  # 新目标使用当前配置守候距离
            self._active_standoff_duration_sec = self.config.standoff_duration_sec  # 新目标使用当前配置守候时间
            self._state = "approaching"  # 新目标或尚未完成时自动进入靠近阶段
            self._phase_started_at = None  # 靠近阶段无需计时
            self._reset_action_log()  # 新序列重新打印第一条动作日志

        if self._planner is not None:  # 如果存在 A* 规划器，取消当前导航，避免与 bbox 追踪指令竞争
            self._planner.cancel_goal()

        logger.info(
            "BBoxDistanceBehaviorModule: task started from selected bbox "
            f"target_id={target_id!r} state_before={current_state}"
        )

        self._publish_status("approaching")  # 发布自动启动状态

    def _on_lock_status(self, msg: String) -> None:  # 处理目标锁/追踪状态消息
        try:  # 尝试解析 tracker 发布的 JSON 状态
            payload = json.loads(str(getattr(msg, "data", "")))  # 从 LCM String 中读取 data 字段
        except json.JSONDecodeError:  # 如果消息不是合法 JSON
            logger.info("BBoxDistanceBehaviorModule: ignoring malformed lock_status")
            return  # 不改变任务状态

        lock_state = str(payload.get("state", ""))  # 读取 tracker 状态名
        with self._lock:  # 加锁更新状态缓存
            self._latest_lock_state = lock_state  # 保存最新 tracker 状态
            current_state = self._state  # 复制当前行为状态
            previous_target_id = self._active_target_id  # 复制当前目标 id

            if lock_state not in ("unselected", "lost") or current_state == "idle":
                return  # locked/searching 不重置任务；idle 也无需重复处理

            self._state = "idle"  # tracker 明确无目标或丢失时停止任务
            self._active_target_id = None  # 清空目标 id
            self._phase_started_at = None  # 清空阶段计时
            self._reset_action_log()  # 重置动作日志节流状态

        self.cmd_vel.publish(Twist.zero())  # 目标明确丢失时立即停车
        logger.info(
            "BBoxDistanceBehaviorModule: task ended because target tracker is inactive "
            f"lock_state={lock_state!r} previous_target_id={previous_target_id!r}"
        )
        self._publish_status("idle")  # 发布 idle 状态

    def _on_lidar(self, lidar: PointCloud2) -> None:  # 处理新的 lidar 点云
        with self._lock:  # 加锁更新缓存
            self._latest_lidar = lidar  # 保存最新 lidar 点云

    def _on_camera_info(self, camera_info: CameraInfo) -> None:  # 处理新的 camera_info
        with self._lock:  # 加锁更新缓存
            self._latest_camera_info = camera_info  # 保存最新 camera_info

    def _on_teleop_active(
        self, _msg: Any
    ) -> None:  # 收到 MovementManager 的 stop_movement 信号时打断任务
        with self._lock:  # 加锁读取并重置状态
            prev_state = self._state  # 记录打断前状态
            prev_target = self._active_target_id  # 记录打断前目标 id
            if prev_state == "idle":  # 已经空闲，无需操作
                return
            self._state = "idle"  # 回到 idle
            self._active_target_id = None  # 清空目标
            self._phase_started_at = None  # 清空阶段计时
            self._reset_action_log()  # 重置动作日志节流状态
        self.cmd_vel.publish(Twist.zero())  # 立即停止
        self._request_selection_clear(
            reason="interrupt"
        )  # 清掉旧 bbox，避免中断后又被 locked bbox 拉起
        logger.info(
            "BBoxDistanceBehaviorModule: task interrupted by teleop "
            f"target_id={prev_target!r} state_before={prev_state}"
        )
        self._publish_status("idle")  # 发布 idle 状态，让外部知道任务已中断

    def _command_loop(self) -> None:  # 固定频率控制循环
        period_sec = 1.0 / max(self.config.command_hz, 1.0)  # 根据配置计算循环周期，并防止除零
        while not self._stop_event.wait(period_sec):  # 按周期运行，收到停止事件后退出
            twist = self._compute_next_twist()  # 根据当前状态和传感器缓存计算下一条速度命令
            if twist is not None:  # idle 状态会返回 None，表示无需重复发布
                self.cmd_vel.publish(twist)  # 发布速度命令

        self.cmd_vel.publish(Twist.zero())  # 线程退出前再发布一次零速度

    def _compute_next_twist(self) -> Twist | None:  # 计算当前周期应该发布的速度
        with self._lock:  # 加锁读取状态和传感器缓存
            state = self._state  # 复制当前状态
            selected_bbox = self._latest_selected_bbox  # 复制 latest selected bbox
            lidar = self._latest_lidar  # 复制 latest lidar
            camera_info = self._latest_camera_info  # 复制 latest camera_info
            near_distance = self._active_near_distance  # 复制本次近距离
            dwell_duration_sec = self._active_dwell_duration_sec  # 复制本次近距离停留时间
            standoff_distance = self._active_standoff_distance  # 复制本次远距离守候距离
            standoff_duration_sec = self._active_standoff_duration_sec  # 复制本次远距离守候时间
            phase_started_at = self._phase_started_at  # 复制当前阶段开始时间

        if state == "idle":  # 如果行为未启动
            self._set_block_reason(None)
            return None  # 不重复发布速度命令

        if state == "done":  # 如果行为已经完成
            self._set_block_reason(None)
            return Twist.zero()  # 持续发布零速度，确保完成后保持停止

        detection = self._extract_single_detection(
            selected_bbox
        )  # 从 selected_bbox 中取出单个 detection
        if (
            detection is None or lidar is None or camera_info is None
        ):  # 如果 bbox、lidar 或 camera_info 任一缺失
            missing: list[str] = []
            if detection is None:
                missing.append("selected_bbox")
            if lidar is None:
                missing.append("lidar")
            if camera_info is None:
                missing.append("camera_info")
            self._set_block_reason(f"missing_inputs:{','.join(missing)}")
            return Twist.zero()  # 发布零速度并等待数据补齐

        distance = self._estimate_3d_distance(
            detection, lidar, camera_info
        )  # 用 TF + pointcloud 估计 3D 距离
        if distance is None:  # 如果 TF 或点云无法给出有效距离
            self._set_block_reason(f"no_3d_detection:{self._detection_id(detection)!r}")
            return Twist.zero()  # 发布零速度并等待下一帧

        self._set_block_reason(None)

        bbox_center_x = float(detection.bbox.center.position.x)  # 读取 bbox 中心 x 像素坐标
        now = time.monotonic()  # 读取当前单调时间，用于阶段计时

        if state == "approaching":  # 第一阶段：靠近目标直到 near_distance
            if distance <= near_distance + _DISTANCE_TOLERANCE_M:  # 到达近距离阈值
                self._transition_to("dwelling_near", now, distance=distance)  # 进入近距离停留阶段
                return Twist.zero()  # 到达瞬间先停车
            twist = self._make_twist(
                distance, near_distance, bbox_center_x, camera_info
            )  # 目标仍较远时继续靠近
            self._log_action(
                state, now, distance, near_distance, phase_started_at, twist
            )  # 打印靠近动作日志
            return twist

        if state == "dwelling_near":  # 第二阶段：原地停留 dwell_duration_sec
            if self._phase_elapsed(now, phase_started_at) >= dwell_duration_sec:  # 停留时间已满
                self._transition_to("retreating", now, distance=distance)  # 进入后退阶段
            else:
                self._log_action(
                    state, now, distance, near_distance, phase_started_at, Twist.zero()
                )  # 打印近距离停留日志
            return Twist.zero()  # dwell 阶段保持原地停止

        if state == "retreating":  # 第三阶段：后退到至少 standoff_distance
            if distance >= standoff_distance:  # 已经满足最小远距离
                self._transition_to("standing_off", now, distance=distance)  # 进入远距离守候阶段
                return Twist.zero()  # 切换阶段时先停车
            twist = self._make_standoff_twist(
                distance, standoff_distance, bbox_center_x, camera_info
            )  # 目标仍太近时继续后退
            self._log_action(
                state, now, distance, standoff_distance, phase_started_at, twist
            )  # 打印后退动作日志
            return twist

        if state == "standing_off":  # 第四阶段：保持至少 standoff_distance 并等待
            if self._phase_elapsed(now, phase_started_at) >= standoff_duration_sec:  # 守候时间已满
                self._transition_to("returning", now, distance=distance)  # 进入返回目标阶段
                return Twist.zero()  # 切换阶段时先停车
            if distance < standoff_distance:  # 守候期间目标靠近，必须主动后退
                twist = self._make_standoff_twist(
                    distance, standoff_distance, bbox_center_x, camera_info
                )  # 发布只允许后退的守距速度
                self._log_action(
                    state, now, distance, standoff_distance, phase_started_at, twist
                )  # 打印守距后退日志
                return twist
            self._log_action(
                state, now, distance, standoff_distance, phase_started_at, Twist.zero()
            )  # 打印远距离等待日志
            return Twist.zero()  # 距离满足要求时保持原地等待

        if state == "returning":  # 第五阶段：返回目标身边
            if distance <= near_distance + _DISTANCE_TOLERANCE_M:  # 已回到近距离阈值
                completed_target_id = self._detection_id(detection)  # 记录完成目标 id
                with self._lock:  # 加锁切换完成状态
                    self._state = "done"  # 标记行为完成
                    self._active_target_id = completed_target_id  # 保存完成目标 id，避免旧 bbox 重启
                    self._phase_started_at = None  # 完成态无需计时
                    self._reset_action_log()  # 完成后重置动作日志节流状态
                logger.info(
                    "BBoxDistanceBehaviorModule: sequence completed "
                    f"target_id={completed_target_id!r} distance={distance:.3f}"
                )
                self._publish_status("done", distance=distance)  # 发布完成状态
                self._request_selection_clear(
                    reason="completed"
                )  # 完整序列结束后清除选择，回到等待用户输入
                return Twist.zero()  # 完成时发布零速度
            twist = self._make_twist(
                distance, near_distance, bbox_center_x, camera_info
            )  # 目标仍较远时靠近返回
            self._log_action(
                state, now, distance, near_distance, phase_started_at, twist
            )  # 打印返回动作日志
            return twist

        return Twist.zero()  # 未识别状态保守停车

    def _request_selection_clear(self, reason: str) -> None:
        self.clear_selection_request.publish(Bool(data=True))
        logger.info(f"BBoxDistanceBehaviorModule: requested selection clear reason={reason}")

    def _reset_action_log(self) -> None:  # 重置动作日志节流状态
        self._last_action_log_at = 0.0  # 下一次动作循环会立即打印
        self._last_action_log_state = None  # 清空上一次状态，确保新阶段首条日志可见

    def _configured_near_distance(self) -> float:  # 读取配置中的近距离，兼容旧 approach_distance 字段
        if self.config.approach_distance is not None:  # 如果旧配置名被显式设置
            return self._resolve_distance(
                self.config.approach_distance, self.config.near_distance
            )  # 旧配置优先，保证旧 blueprint 行为可控
        return self._resolve_distance(
            self.config.near_distance, 0.5
        )  # 否则使用新的 near_distance 默认值

    def _resolve_active_near_distance(
        self,
        *,
        near_distance: float | None,
        approach_distance: float | None,
    ) -> float:  # 解析 RPC 覆盖参数中的近距离
        if near_distance is not None:  # 新参数优先
            return self._resolve_distance(near_distance, self._configured_near_distance())
        if approach_distance is not None:  # 旧参数作为兼容别名
            return self._resolve_distance(approach_distance, self._configured_near_distance())
        return self._configured_near_distance()  # 没有覆盖时使用配置值

    @staticmethod  # 声明这是不依赖实例状态的工具函数
    def _resolve_distance(value: float | None, default: float) -> float:  # 校验并解析距离参数
        resolved = default if value is None else float(value)  # 使用覆盖值或默认值
        if not math.isfinite(resolved) or resolved <= 0.0:  # 距离必须是正有限数
            raise ValueError(f"distance must be positive and finite, got {resolved!r}")
        return resolved  # 返回合法距离

    @staticmethod  # 声明这是不依赖实例状态的工具函数
    def _resolve_duration(value: float | None, default: float) -> float:  # 校验并解析时间参数
        resolved = default if value is None else float(value)  # 使用覆盖值或默认值
        if not math.isfinite(resolved) or resolved < 0.0:  # 时长必须是非负有限数
            raise ValueError(f"duration must be non-negative and finite, got {resolved!r}")
        return resolved  # 返回合法时长

    @staticmethod  # 声明这是不依赖实例状态的工具函数
    def _phase_elapsed(now: float, phase_started_at: float | None) -> float:  # 计算当前阶段已用时间
        if phase_started_at is None:  # 如果阶段尚未记录开始时间
            return 0.0  # 返回 0，避免误跳阶段
        return max(0.0, now - phase_started_at)  # 返回非负 elapsed 秒数

    @staticmethod  # 声明这是不依赖实例状态的工具函数
    def _should_reset_on_empty_selection(lock_state: str | None) -> bool:  # 判断空 bbox 是否代表清空目标
        return lock_state in (None, "unselected", "lost")  # searching 时保留任务等待 tracker 重找

    def _transition_to(
        self, state: BehaviorState, now: float, **fields: float
    ) -> None:  # 切换状态并记录阶段开始时间
        with self._lock:  # 加锁更新状态机
            old_state = self._state  # 记录旧状态，便于日志
            self._state = state  # 设置新状态
            self._phase_started_at = now  # 记录新阶段开始时间
            self._reset_action_log()  # 新阶段第一条控制命令要打印日志
        logger.info(f"BBoxDistanceBehaviorModule: state {old_state} -> {state}")
        self._publish_status(state, **fields)  # 发布状态变化

    def _log_action(
        self,
        state: BehaviorState,
        now: float,
        distance: float,
        target_distance: float,
        phase_started_at: float | None,
        twist: Twist,
    ) -> None:  # 按阶段节流打印正在执行的动作
        interval_sec = max(float(self.config.action_log_interval_sec), 0.0)  # 读取日志节流间隔
        if (
            self._last_action_log_state == state
            and interval_sec > 0.0
            and now - self._last_action_log_at < interval_sec
        ):  # 同一阶段且还没到日志间隔
            return  # 跳过本周期日志，避免 20Hz 刷屏

        with self._lock:  # 加锁读取当前目标 id
            target_id = self._active_target_id  # 复制目标 id

        elapsed_sec = self._phase_elapsed(now, phase_started_at)  # 计算当前阶段已执行时间
        logger.info(
            "BBoxDistanceBehaviorModule: action "
            f"state={state} target_id={target_id!r} "
            f"distance={distance:.3f} target_distance={target_distance:.3f} "
            f"elapsed_sec={elapsed_sec:.1f} "
            f"linear_x={float(twist.linear.x):.3f} angular_z={float(twist.angular.z):.3f}"
        )  # 输出当前动作、距离和速度，便于现场判断是否在靠近/后退/等待
        self._last_action_log_at = now  # 更新上次日志时间
        self._last_action_log_state = state  # 更新上次日志状态

    def _make_standoff_twist(  # 定义守距阶段速度，只允许后退，不允许主动靠近
        self,
        distance: float,
        standoff_distance: float,
        bbox_center_x: float,
        camera_info: CameraInfo,
    ) -> Twist:
        twist = self._make_twist(
            distance, standoff_distance, bbox_center_x, camera_info
        )  # 复用距离控制和转向控制
        linear_x = min(float(twist.linear.x), 0.0)  # 守距时线速度只允许为负或零，避免主动前进
        return Twist(
            linear=Vector3(linear_x, 0.0, 0.0),
            angular=twist.angular,
        )  # 返回守距速度命令

    def _make_twist(  # 定义根据距离和 bbox 横向位置生成 Twist 的函数
        self,  # 传入模块实例
        distance: float,  # 当前估计距离
        target_distance: float,  # 当前阶段目标距离
        bbox_center_x: float,  # bbox 中心 x 像素坐标
        camera_info: CameraInfo,  # 相机内参
    ) -> Twist:  # 返回速度命令
        distance_error = distance - target_distance  # 计算目标距离误差，正数表示目标太远
        linear_x = self._clamp(
            distance_error * _LINEAR_GAIN,
            -self.config.max_linear_speed,
            self.config.max_linear_speed,
        )  # 计算并限幅线速度
        fx = float(camera_info.K[0]) if len(camera_info.K) >= 9 else 0.0  # 读取相机 fx
        cx = float(camera_info.K[2]) if len(camera_info.K) >= 9 else 0.0  # 读取相机 cx
        angular_z = (
            0.0 if fx <= 0.0 else -((bbox_center_x - cx) / fx) * _ANGULAR_GAIN
        )  # 根据 bbox 横向误差计算转向速度
        angular_z = self._clamp(
            angular_z, -self.config.max_angular_speed, self.config.max_angular_speed
        )  # 对角速度限幅
        return Twist(  # 构造 Twist 命令
            linear=Vector3(linear_x, 0.0, 0.0),  # 设置 x 方向线速度
            angular=Vector3(0.0, 0.0, angular_z),  # 设置 yaw 角速度
        )  # 结束 Twist 构造

    def _estimate_3d_distance(  # 定义基于 TF + 世界坐标系点云的 3D 距离估计函数
        self,  # 传入模块实例
        detection: Any,  # 当前选中的 LCM Detection2D
        lidar: PointCloud2,  # 最新 lidar 点云（Go2 原始点云 frame_id="world"）
        camera_info: CameraInfo,  # 最新相机内参
    ) -> float | None:  # 返回机器人与检测目标之间的 2D 欧氏距离（单位米），失败时返回 None
        # 1. 构造 Detection2DBBox 供 Detection3DPC.from_2d() 使用
        x1, y1, x2, y2 = BBoxSelectionModule._bbox_corners(detection)  # 读取 bbox 的 xyxy 像素范围
        det2d = Detection2DBBox(
            bbox=(x1, y1, x2, y2),  # 传入像素 bbox
            track_id=_safe_track_id(getattr(detection, "id", None)),  # 读取 track id，非整型时用 0
            class_id=0,  # class_id 在此不重要
            confidence=float(getattr(detection, "score", 1.0) or 1.0),  # 读取置信度，缺失时用 1.0
            name=str(getattr(detection, "class_label", "") or ""),  # 读取类名，缺失时用空字符串
            ts=float(lidar.ts or 0.0),  # 用 lidar 时间戳对齐 TF 查询
            image=None,  # from_2d() 不需要 image 字段
        )
        # 2. 从 TF 获取 world→camera_optical 变换
        ts = float(lidar.ts or 0.0)  # 用 lidar 时间戳对齐 TF
        world_to_optical = self._lookup_transform(
            "camera_optical", lidar.frame_id, ts
        )  # 查询 world→camera_optical 变换
        if world_to_optical is None:  # 如果 TF 暂时不可用
            return None  # 等待下一帧
        # 3. 包装 LCM CameraInfo
        lcm_ci = DimosLcmCameraInfo()  # 构造 LCM CameraInfo 对象
        lcm_ci.K = camera_info.K  # 复制内参矩阵
        lcm_ci.width = camera_info.width  # 复制图像宽度
        lcm_ci.height = camera_info.height  # 复制图像高度
        # 4. 投影到 3D（世界坐标系）
        detection_3d = Detection3DPC.from_2d(
            det=det2d,  # 传入 2D bbox
            world_pointcloud=lidar,  # 传入世界坐标系点云
            camera_info=lcm_ci,  # 传入 LCM CameraInfo
            world_to_optical_transform=world_to_optical,  # 传入外参变换
            filters=[],  # 不做额外过滤，保证速度
        )  # 返回带世界坐标系点云的 Detection3DPC，或 None
        if detection_3d is None:  # 如果 bbox 内没有有效点
            return None  # 等待下一帧
        # 5. 获取机器人当前位置（world 坐标系）
        robot_tf = self.tf.get(
            "world", "base_link", time_tolerance=self.config.tf_time_tolerance
        )  # 查询机器人位置
        if robot_tf is None:  # 如果机器人 TF 暂时不可用
            return None  # 等待下一帧
        # 6. 计算机器人与检测目标中心的 2D 欧氏距离
        center = detection_3d.center  # 获取 bbox 点云质心（世界坐标系）
        robot_pos = robot_tf.translation  # 获取机器人在世界坐标系中的位置
        dx = float(center.x) - float(robot_pos.x)  # x 轴方向距离分量
        dy = float(center.y) - float(robot_pos.y)  # y 轴方向距离分量
        distance = math.sqrt(dx * dx + dy * dy)  # 计算 XY 平面欧氏距离
        return (
            distance if math.isfinite(distance) and distance > 0.0 else None
        )  # 只接受有限且正的距离

    def _lookup_transform(
        self, parent_frame: str, child_frame: str, ts: float
    ) -> Transform | None:  # 查询 TF；时间戳查不到时回退到最新 TF
        if self.config.prefer_latest_tf:  # custom 行为默认更重视实时控制可用性
            return self.tf.get(
                parent_frame,
                child_frame,
                time_tolerance=self.config.tf_time_tolerance,
            )  # 使用最新 TF，避免 replay/live 时间戳漂移触发持续 no_3d_detection

        transform = self.tf.get(
            parent_frame,
            child_frame,
            ts,
            time_tolerance=self.config.tf_time_tolerance,
        )  # 优先使用和 lidar 时间戳对齐的 TF
        if transform is not None:  # 如果按时间戳查到了
            return transform  # 返回精确时间附近的 TF

        if ts <= 0.0:  # 如果传入时间戳本身不可用
            return None  # 不做 latest fallback，保持失败语义清晰

        return self.tf.get(
            parent_frame,
            child_frame,
            time_tolerance=self.config.tf_time_tolerance,
        )  # replay/live 时间戳漂移时，回退到最新 TF，避免任务永久卡在 no_3d_detection

    def _set_block_reason(self, reason: str | None) -> None:
        if reason == self._last_block_reason:
            return
        self._last_block_reason = reason
        if reason is not None:
            logger.info(f"BBoxDistanceBehaviorModule: task blocked reason={reason}")

    @staticmethod  # 声明这是不依赖实例状态的工具函数
    def _extract_single_detection(
        selected_bbox: Detection2DArray | None,
    ) -> Any | None:  # 从 selected_bbox 中取单个 detection
        if (
            selected_bbox is None
            or selected_bbox.detections_length == 0
            or not selected_bbox.detections
        ):  # 如果没有选中 bbox
            return None  # 返回空
        return selected_bbox.detections[0]  # 返回第一个 detection，选择模块保证最多一个

    @staticmethod  # 声明这是不依赖实例状态的工具函数
    def _detection_id(detection: Any) -> str:  # 读取 detection.id，并在缺失时回退为空字符串
        detection_id = getattr(detection, "id", "")  # 读取 detection.id，缺失时使用空字符串
        return str(detection_id) if detection_id else ""  # 返回真实 id 或空字符串

    @staticmethod  # 声明这是不依赖实例状态的工具函数
    def _clamp(value: float, lower: float, upper: float) -> float:  # 把数值限制在上下界之间
        return max(lower, min(upper, value))  # 返回限幅后的数值

    def _publish_status(self, state: BehaviorState, **fields: float) -> None:  # 发布行为状态消息
        payload: dict[str, Any] = {"state": state, **fields}  # 构造 JSON 状态载荷
        self.behavior_status.publish(String(json.dumps(payload)))  # 发布 JSON 字符串状态


__all__ = [  # 声明这个文件希望对外暴露的名字
    "BBoxDistanceBehaviorConfig",  # 暴露行为模块配置
    "BBoxDistanceBehaviorModule",  # 暴露行为模块
]  # 结束 __all__ 列表
