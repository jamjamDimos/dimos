# Custom Robot Blueprints

## 目录结构

```text
dimos/robot/custom/
├── modules/                               # 纯业务逻辑，无 blueprint / vis 代码
│   ├── bbox_selection_module.py           # BBoxSelectionModule, BBoxSelectionConfig
│   ├── selected_bbox_csrt_tracker_module.py # SelectedBBoxCsrtTrackerModule, SelectedBBoxCsrtTrackerConfig
│   ├── target_lock_module.py              # TargetLockModule, TargetLockConfig
│   ├── yoloe_tracking_module.py           # YoloeTrackingModule, YoloeTrackingConfig
│   └── go2_startup_self_check_module.py   # Go2StartupSelfCheck, Go2StartupSelfCheckConfig
├── tasks/                                 # 任务实现：每个 task 自带状态机
│   └── bbox_distance_behavior_module.py   # BBoxDistanceBehaviorModule, BBoxDistanceBehaviorConfig
├── visualization/                         # Detection2DArray -> Rerun 2D overlay 适配
│   └── detection2d_overlay.py             # detection_array_to_rerun / detections_overlay /
│                                          # selected_bbox_overlay / yoloe_overlay
├── blueprints/                            # autoconnect 组装 + rerun config + requirements
│   ├── bbox_distance_follow.py            # 最小距离任务蓝图
│   ├── yoloe_target_lock_distance_follow.py # 推荐闭环示例：YOLOE + selection + CSRT tracker + task
│   ├── yoloe_keyboard_teleop.py           # 键盘遥控 + YOLOE（非本文重点）
│   ├── yoloe_tracking_test.py             # 仅检测/跟踪验证
│   └── go2_startup_self_check.py          # 开机自检蓝图
└── tests/
    ├── test_bbox_distance_behavior_module.py
    ├── test_selected_bbox_csrt_tracker_module.py
    └── test_target_lock_module.py
```

依赖方向：`blueprints/` -> `modules/` + `tasks/` + `visualization/`；`tests/` -> `modules/` + `tasks/`。

## 如何最小化测试

建议按下面顺序做最小验证，能最快定位问题在哪一层。

### 1. 先跑 task + tracker 单元测试

```bash
source .venv/bin/activate
pytest dimos/robot/custom/tests/test_selected_bbox_csrt_tracker_module.py dimos/robot/custom/tests/test_bbox_distance_behavior_module.py -q
```

### 2. 再验证 blueprint 自动注册

```bash
source .venv/bin/activate
pytest dimos/robot/test_all_blueprints_generation.py
```

### 3. 最后跑最小闭环蓝图

```bash
.venv/bin/dimos --replay run yoloe-target-lock-distance-follow
```

这个顺序可以把问题快速归类到：
- 单模块逻辑
- 蓝图注册/装配
- 运行时流转

## 当前 blueprint 状态机

### 1) yoloe-target-lock-distance-follow（推荐闭环）

组成：
- `unitree_go2_basic`
- `YoloeTrackingModule.blueprint()`
- `BBoxSelectionModule.blueprint()`
- `SelectedBBoxCsrtTrackerModule.blueprint()`
- `BBoxDistanceBehaviorModule.blueprint(near_distance=0.5, dwell_duration_sec=10.0, standoff_distance=1.5, standoff_duration_sec=45.0)`
- `KeyboardTeleop.blueprint(publish_only_when_active=True)`
- `MovementManager.blueprint()`

键盘控制（pygame 窗口需要焦点）：
- W / S — 前进 / 后退
- A / D — 左转 / 右转
- Q / E — 横移
- Shift — 加速 (2×)  |  Ctrl — 慢速 (0.5×)
- Space — 紧急停止  |  Esc / Q — 退出

CSRT tracker 状态机：

```mermaid
stateDiagram-v2
    [*] --> unselected
    unselected --> locked: user_selected_bbox 有效 + CSRT 初始化成功
    locked --> searching: CSRT update 失败
    searching --> locked: YOLOE nearby candidate 重捕获成功
    searching --> lost: lost_frames > max_lost_frames
    lost --> locked: 用户重新选择
    locked --> unselected: stop_movement / clear_selection_request / RPC stop
    searching --> unselected: stop_movement / clear_selection_request / RPC stop
```

模块消息流：

```mermaid
flowchart LR
    C[color_image] --> Y[YoloeTrackingModule]
    Y --> D[detections]
    D --> S[BBoxSelectionModule]
    S --> USB[user_selected_bbox]
    D --> T[SelectedBBoxCsrtTrackerModule]
    C --> T
    USB --> T
    T --> TB[tracked_bbox]
    T --> TS[tracking_status]
    TB --> B[BBoxDistanceBehaviorModule]
    TS --> B
    L[lidar] --> B
    I[camera_info] --> B
    B -->|nav_cmd_vel| MM[MovementManager]
    KB[KeyboardTeleop] -->|tele_cmd_vel| MM
    MM -->|cmd_vel| V[Go2Connection]
    MM -->|stop_movement = teleop_active| B
```

任务状态机：

```mermaid
stateDiagram-v2
    [*] --> idle
    idle --> approaching: 用户点击 YOLOE bbox
    approaching --> dwelling_near: distance <= 0.5m + tolerance
    dwelling_near --> retreating: dwell 10s
    retreating --> standing_off: distance >= 1.5m
    standing_off --> standing_off: distance < 1.5m / 后退守距
    standing_off --> returning: standoff 45s
    returning --> done: distance <= 0.5m + tolerance
    done --> idle: clear_selection_request
    approaching --> idle: 用户接管 / 目标丢失
    dwelling_near --> idle: 用户接管 / 目标丢失
    retreating --> idle: 用户接管 / 目标丢失
    standing_off --> idle: 用户接管 / 目标丢失
    returning --> idle: 用户接管 / 目标丢失
```

速度优先级（MovementManager）：
- 键盘有输入时：`tele_cmd_vel` 优先，`nav_cmd_vel` 被压制（冷却 1 s）
- 冷却结束后：`nav_cmd_vel`（任务）恢复控制
- 键盘输入同时触发 `stop_movement → teleop_active` → 任务重置为 idle
- 这个 blueprint 会把 `ReplanningAStarPlanner.clicked_point` remap 到 `navigation_clicked_point`，避免 Camera 里的 bbox 点击同时触发 A* 导航。

Visualization 点击闭环：

```mermaid
flowchart LR
    A[dimos-view Camera click] --> B[RerunWebSocketServer.clicked_point]
    B --> C[BBoxSelectionModule._on_clicked_point]
    C --> D[user_selected_bbox]
    D --> E[SelectedBBoxCsrtTrackerModule._on_user_selected_bbox]
    F[detections] --> G[SelectedBBoxCsrtTrackerModule._on_detections]
    E --> H[tracked_bbox]
    G --> H
    H --> K[/color_image/tracked_bbox]
```

### 2) bbox-distance-follow（最小任务链路）

组成：
- `unitree_go2_basic`
- `Detection2DModule.blueprint(camera_info=GO2Connection.camera_info_static, publish_detection_images=False)`
- `BBoxSelectionModule.blueprint()`
- `BBoxDistanceBehaviorModule.blueprint()`

任务状态机：

```text
idle -> approaching -> dwelling_near -> retreating -> standing_off -> returning -> done
```

状态转移规则：
- 选中 bbox（非空）-> `approaching`
- 到达 `near_distance + 0.05m` -> `dwelling_near`
- 停留 `dwell_duration_sec` -> `retreating`
- 后退到 `standoff_distance` 或更远 -> `standing_off`
- `standing_off` 期间如果距离小于 `standoff_distance`，只允许后退守距
- 守候 `standoff_duration_sec` -> `returning`
- 返回到 `near_distance + 0.05m` -> `done`
- 选择清空（空 bbox）-> `idle`

## Task Module

`BBoxDistanceBehaviorModule` 位于 `tasks/`，职责是“执行任务”，不是“做检测或选择”。

职责：
- 输入：`selected_bbox` + `lidar` + `camera_info`
- 输出：`cmd_vel` + `behavior_status`
- 行为目标：点中目标后，持续用 bbox + lidar 估计目标空间距离，执行靠近、停留、后退、守候、返回序列

RPC：
- `start_bbox_distance_behavior(approach_distance=None, near_distance=None, dwell_duration_sec=None, standoff_distance=None, standoff_duration_sec=None) -> str`
- `stop_bbox_distance_behavior() -> str`

默认参数：
- `command_hz = 20.0`
- `near_distance = 0.5`
- `dwell_duration_sec = 10.0`
- `standoff_distance = 1.5`
- `standoff_duration_sec = 45.0`
- `max_linear_speed = 0.45`
- `max_angular_speed = 0.8`
- `approach_distance = None`（旧配置别名；设置后覆盖 `near_distance`）
- `tf_time_tolerance = 5.0`
- `prefer_latest_tf = True`（优先用最新 TF，避免 replay/live 时间戳漂移导致持续 `no_3d_detection`）
- `action_log_interval_sec = 1.0`

点击和深度调参（实机/仿真常用）：
- `BBoxSelectionConfig.click_hit_padding_px`：点击命中 bbox 的边缘扩张像素。
- `BBoxSelectionConfig.click_snap_distance_px`：未命中时吸附到最近 bbox 的最大像素距离。
- `BBoxDistanceBehaviorConfig.tf_time_tolerance`：bbox 投影到点云时允许的 TF 时间容差。
- `BBoxDistanceBehaviorConfig.max_linear_speed`：限制靠近/后退速度。
- `BBoxDistanceBehaviorConfig.max_angular_speed`：限制朝向 bbox 中心的 yaw 速度。

推荐调参顺序：
1. 点不中 bbox：先增大 `click_hit_padding_px`，再增大 `click_snap_distance_px`。
2. 点中了但不前进：检查 `no_3d_detection` 日志、`lidar`、`camera_info`、TF 是否齐全。
3. 后退/返回太快：降低 `max_linear_speed`。
4. bbox 不居中：调整 `max_angular_speed` 或检查相机内参。

边界行为：
- bbox 为空：发布 `Twist.zero()` 并回到 `idle`
- Tracker `searching` 时：保留当前任务，发布 `Twist.zero()` 等待重新锁定
- Tracker `unselected/lost` 时：发布 `Twist.zero()` 并回到 `idle`
- 3D 距离无效：发布 `Twist.zero()` 并等待
- 完整序列结束或 stop：发布 `Twist.zero()`，并请求清除选择

## 排查故障

### 1. 看不到框
- 检查检测流是否有输出（`detections` 是否持续更新）。
- 检查 overlay 绑定是否正确（`/color_image/yoloe_detections`、`/color_image/selected_bbox`、`/color_image/tracked_bbox`）。
- replay 场景下确认模型文件存在。

### 2. 点击后不进入 approaching
- 检查 `clicked_point` 是否到达 `BBoxSelectionModule`。
- 检查 `selected_bbox` 是否非空。
- 检查 `behavior_status` 是否切到 `approaching`。

### 3. 锁定后很快丢失
- 检查 CSRT 是否能在当前图像质量下 update。
- 检查 `SelectedBBoxCsrtTrackerConfig.max_lost_frames` 是否过短。
- 检查 `reacquire_max_center_jump_px` 是否过小，导致 YOLOE nearby candidate 被拒绝。
- 观察 `tracking_status` / 行为模块 `lock_status` 是否进入 `searching/lost`。

### 4. 不前进但没 done
- 检查点云是否正确投影到 bbox。
- 检查 `camera_info` 是否匹配当前相机。
- 检查 bbox 是否过小或位置异常导致深度采样失败。

### 5. 注册/运行异常
- 重新执行：

```bash
source .venv/bin/activate
pytest dimos/robot/test_all_blueprints_generation.py
```

### 6. 建议排查顺序
1. 单测：`test_selected_bbox_csrt_tracker_module.py` + `test_bbox_distance_behavior_module.py`
2. 注册：`test_all_blueprints_generation.py`
3. 运行：`dimos --replay run yoloe-target-lock-distance-follow`
