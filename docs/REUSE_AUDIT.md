# P0 本机复用审计

- 工作区 `~/AstraDroneOpen/AstraDrone_ros2_ws`，新包 `src/shade_challenger`，原包空目录。
- 首次 Git 状态、已有跟踪文件差异、关键配置及用户任务书已保存到 `results/preflight/`。未覆盖用户原有修改。
- `src/YOPO-ROS2/YOPO` 存在原网络、primitive、quintic、ESDF loss、Dataset、Trainer、epoch50 预训练权重。仅 `scripts/baseline.py` 导入这些算法，GAP 训练/部署不导入。
- 原 `Simulator/src/src/dataset_generator.cpp`、CUDA raycaster、随机森林生成器都存在，但未发现可执行数据生成器和原 PNG/PLY 数据集。CMake 锁定 CUDA 12.8 EXACT，本机 toolkit 为 12.3；因此没有声称原 CUDA 森林管线已跑通。
- 可复用 `dependencies/yopo400_dataset/scene_*.npz`：已有 Astra YOPO 接入的八个盒形环境，共 1024 个位姿。GAP 复用环境几何与位姿缓存，重新按照真实相机 FOV 和 MID360 CSV 离线渲染。它不是原始树网格森林数据。
- 原 YOPO baseline 将上述数据转换为 PNG/PLY，实际运行上游 YopoTrainer、YopoNetwork、YOPOLoss 两个 epoch，未改上游源码。所需缺失 Python 包安装到新包 `.deps`，不重建 Conda 环境。
- 已有整机 `astra_yopo/models/my/my.sdf`，包含 `iris400` 本体、电机、PX4 插件、前视 D435、MID360 和真值评价插件。直接复用，不修改几何、惯量、安装位置或已有 PX4 参数。
- 原项目 `simulation/px4_sim_files/px4_iris_sdf`、`simulation/astra_gazebo_models` 也存在多种全机模型；本次优先使用已经接入 ROS2、FAST-LIO 和 MAVROS 的 `my`。此组装模型没有下视相机，未伪称所有传感器齐备。
- D435：原生 640×360、HFOV 1.51843645 rad、20 Hz、0.2–20 m，topic `/camera/depth/image_rect_raw`，body 外参 `[.11,0,.03]`；optical 坐标到 FLU 为 `(z,-x,-y)`。
- MID360：topic `/livox/lidar` 同时提供 PointCloud2、CustomMsg，10 Hz；复用插件 CSV 非重复扫描，20000 samples，部署沿用已有 runner 的 downsample=4。ray 原点在 body `[0,0,.13]`。插件所有点是在帧末实际捕获，CustomMsg `header+offset_time` 表示该时刻。没有把它当作规则旋转扫描。
- FAST-LIO：复用 `astra_yopo/config/mid360_sim.yaml`，IMU 外参及 lidar→IMU 外参保持原值。复用 `lio_bridge.py`，只把 `/yopo/odom` 重映射为 `/gap/odom`；Actor 的角速度单独读取 `/livox/imu`，没有依赖 LIO 里程计中的空角速度字段。
- 复用 `astra_wall_control/run_sim.sh` 和 `/home/li/PX4-Autopilot/build/px4_sitl_default/bin/px4`；独立 rootfs `/tmp/gap_nav_px4`。
- MAVROS 本机特殊话题：`/mavros/mavros/local` 同时承载不同消息类型。控制冲突检查按控制消息类型识别，不能把 Odometry 发布者计入。ROS 输入保持 ENU，MAVROS 完成 NED 转换。
- Gazebo 真值 `/yopo_eval/body` 仅由记录器读取；Actor 不读取世界真值障碍物或位姿。训练代价可读取环境几何，部署神经网络只读取传感器、估计状态和目标。

## 已知模型近似

`config/model_audit.json` 保存了从 SDF 汇总的质量、质心及惯量。训练动力学是带执行器滞后、传输延迟、阻力、推力/角速度约束的刚体近似；执行器响应尚未用 PX4 系统辨识。飞行验证和高速验收必须单独完成。
