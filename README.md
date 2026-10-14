# shade_challenger：GAP-Nav 相机导航

当前按用户要求优先开发 **Depth 相机版本**。算法是本包独立实现的连续 Actor、全状态 MPC 教师和时序/策略双影子；`astra_yopo` 只提供训练环境、整机、传感器、定位与 PX4 启动资源。YOPO 原算法仅由独立基线脚本运行，不参与 GAP 的导航决策。

实际训练、飞行结果与尚未通过的项目见 [实验报告](results/REPORT.md)。这是研发版本；不能将离线预测碰撞率作为 PX4 飞行成功率，也没有完成高速随机森林验收。

## 一键运行相机版

无需先切目录。下面一条命令启动已有 Astra 完整机体、Gazebo 界面、PX4 SITL、FAST-LIO、相机 Actor、控制器和独立安全节点，并在本机 SITL 自动起飞：

```bash
~/AstraDroneOpen/AstraDrone_ros2_ws/src/shade_challenger/scripts/run_depth.sh --auto --gui
```

默认任务：起飞至相对 1.6 m → Actor 导航到相对 `[4,0,0]` 的目标 → 减速悬停 → 降落。水平跟踪速度限制为 1.5 m/s。目标位于现有测试世界首个障碍物后方；旧测试的 3 m 终点离障碍物过近，已经修正。

只启动、不自动解锁：

```bash
~/AstraDroneOpen/AstraDrone_ros2_ws/src/shade_challenger/scripts/run_depth.sh --gui
```

一键无界面自动测试、记录真实深度相机视频、飞行日志和评价 JSON：

```bash
~/AstraDroneOpen/AstraDrone_ros2_ws/src/shade_challenger/scripts/run_depth.sh --test
```

可组合 `--test --gui`。每次测试保存到新的 `results/camera_日期_时间/`，包括 `run.json`（模型 SHA256）、`launch.log`、`events.jsonl`、`actual_depth_stream.avi`、`observed.json`、`analysis.json`。视频是实际 ROS 深度流的伪彩色显示；不是生成画面。评价使用 Gazebo 真值，Actor 使用 FAST-LIO 估计，两者分离。

主动结束当前任务并降落：

```bash
source /opt/ros/foxy/setup.bash
source ~/AstraDroneOpen/AstraDrone_ros2_ws/install/setup.bash
ros2 service call /gap/land std_srvs/srv/Trigger '{}'
```

自动解锁须同时满足模拟时钟、专用 rootfs 下的本机 PX4 SITL 进程、传感器和估计就绪。launch 固定连接 loopback SITL。不要同时启动已有 YOPO 导航控制器；安全节点也会检测控制发布者冲突。

## 一键训练相机版

```bash
~/AstraDroneOpen/AstraDrone_ros2_ws/src/shade_challenger/scripts/train_depth.sh
```

默认流程：4096 个新随机相机位姿 → 100000 条状态/目标样本 → 基础训练 30 epochs → 1% 困难状态的离线 MPC/双影子 → 增强训练 5 epochs → ONNX 导出并数值对照。训练使用现有八个盒形环境，六个训练、两个整张世界留作验证。十万条状态共享 4096 个缓存感知视角，并非十万个独立环境。

脚本不启动 Gazebo，主训练循环不包含渲染或 MPC。输出使用新的时间戳目录，保留旧实验。成功导出后 `results/depth_current` 指向新模型，下一次一键运行会加载它；新训练模型仍须重新做 SITL 验证。

可通过环境变量设置 `GAP_RUN_DIR`、`GAP_PYTHON`、`GAP_VIEWS`、`GAP_STATES`、`GAP_BASE_EPOCHS`、`GAP_DUAL_EPOCHS`、`GAP_SHADOW_FRACTION`、`GAP_MPC_ITERATIONS`、`GAP_INITIAL`。`GAP_INITIAL` 用于明确指定预训练起点；不设置时从随机初始化开始。

本次 `camera_v3` 的实际命令使用已有相机基础权重继续训练 15+5 epochs，完整参数、耗时及限制在实验报告中。重新运行同配置：

```bash
GAP_INITIAL=~/AstraDroneOpen/AstraDrone_ros2_ws/src/shade_challenger/results/augmented/depth_base/best.pt \
GAP_BASE_EPOCHS=15 \
~/AstraDroneOpen/AstraDrone_ros2_ws/src/shade_challenger/scripts/train_depth.sh
```

本机训练 Python 为 `/home/li/miniconda3/envs/torch/bin/python`，ROS2 使用 `/usr/bin/python3` 和 CPU ONNX Runtime。额外基线依赖放在包内 `.deps`；不重建或替换已有 Conda/ROS 环境。

## 配置与算法边界

- `results/depth_current/best.onnx` 是当前相机模型；`run_depth.sh --weights /绝对路径/model.onnx` 可以指定其他模型。`best.pt`、Critic、训练曲线和导出误差留在对应结果目录。
- 相机为已有 D435，原生 640×360、87° 水平视场、20 Hz、0.2–20 m；推理输入为 96×160 的距离/有效性两通道。外参 `[.11,0,.03]`，光学坐标转换到 FLU。
- 输入完整状态 `[p_world(3),v_world(3),q_wxyz(4),omega_body(3)]` 和相对目标。四元数归一化并统一符号。输出四个三次 Bezier 加速度/偏航率控制点，积分得到连续 p/v/a，时域 0.96 s。
- MPC 12 节点、每节点四个刚体动力学子步，保存所有状态、实际推力/力矩、执行器滞后和延迟命令。质量与惯量来自原机体审计；仍为近似模型，尚未完成 PX4 执行器辨识。
- 最多两个风险关键节点继承完整状态与执行器记忆；优化后缀后重新优化原始前缀，避免把未来动作搬到当前。最多两个连续策略挑战，同初态、同扰动、同 1.92 s 评价时域。
- 候选按可行性与代价选择，不平均不同机动。拟合到 Actor 轨迹后再次按全动力学检查，再写入优势监督缓存。未来价值使用继承点重新渲染的相机观测。
- 训练主干是缓存输入、连续轨迹代价、可行性/平滑、优势监督及价值监督。部署不加载 MPC、Critic 或影子。
- 飞行 TRACK 阶段由 Actor 轨迹生成速度参考，带限速及高度边界，PX4 负责内环；起飞、悬停用位置跟踪。MAVROS 完成 ENU/FLU 与 NED/FRD 转换。跟踪器不读取障碍地图或搜索避障路径。
- 独立安全节点监测估计/传感器/推理时效、参考频率、姿态/角速度/速度、OFFBOARD 与发布者冲突。失效执行降落，不为通过实验关闭保护。

## 构建、验证与消融

一键运行脚本在缺少本包安装时自动构建；手动重新构建：

```bash
cd ~/AstraDroneOpen/AstraDrone_ros2_ws
source /opt/ros/foxy/setup.bash
colcon build --packages-select shade_challenger --symlink-install
```

```bash
cd ~/AstraDroneOpen/AstraDrone_ros2_ws
export PYTHONPATH="$PWD/src/shade_challenger"
PY=/home/li/miniconda3/envs/torch/bin/python
PKG="$PWD/src/shade_challenger"
"$PY" -m unittest discover -s "$PKG/tests" -v
"$PY" "$PKG/scripts/evaluate.py" "$PKG/results/camera_v3/dual/best.pt" --data "$PKG/results/camera_v3/samples.pt"
"$PY" "$PKG/scripts/ablate.py" --sensor depth --data "$PKG/results/camera_v3/samples.pt" \
  --base "$PKG/results/camera_v3/base/best.pt" --cache "$PKG/results/camera_v3/shadows/shadows.pt" \
  --output "$PKG/results/camera_ablation" --epochs 5 --batch-size 64
```

消融包含无增强、MPC、Temporal、Policy、Dual，使用相同起点和相同更新次数。离线物理闭环与 PX4 飞行分别记录；消融未匹配原 YOPO 的完整训练墙钟预算，不能据此宣称优于 YOPO。源模型/配置未修改检查由 `scripts/verify_assets.py` 完成；复用明细见 [REUSE_AUDIT](docs/REUSE_AUDIT.md)。

MID360 的已有代码、权重与失败日志保留，后续开发按用户要求暂停；当前入口和验收重点均为相机版。
