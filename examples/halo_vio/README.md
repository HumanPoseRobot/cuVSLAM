# runtime_reduced_v1.37-rc1

原生鱼眼在线 VIO；默认无人体掩模、无额外裁剪，回环和 PGO 关闭。运行前先检查 cam2 相对标定是否需要 180° 补偿；证据不足时停止，不凭文件名猜测。相机按真实同步组/时间戳配对，IMU 默认使用标定 YAML 的时移，额外时移为 0。

```bash
./run_vio.sh --data-root /path/to/recording --output-root /path/to/new-output --audit
./run_vio.sh --data-root /path/to/recording --output-root /path/to/check-output --check-only
```

目录可包含同一段连续录像的多个 pack；标定发生变化时需要拆分。输出目录必须为空。

本版包含已核实的 Cauchy 导数、初始化 LM 对称模型、闭式积分数值范围、旋转正交化、实际鱼眼像素验收、FP64 后验/边缘化、cuNLS 失败传播与工作区初始化修复。输出基于标定噪声的条件视觉有效信息，并正确转换到 IMU 局部坐标。入口启用 `VIO_VISUAL_INFORMATION=1`；不会将其替换为独立导航先验。

源射线参数化和几何不确定度实验默认关闭，启动器清理继承的实验变量。旧发布包保留。

## 已知限制

本版不宣称全部数学问题清零。完整地图—历史位姿相关性、完整惯性初始化联合协方差尚未实现；导航初始化仍有工程先验。room 的既有高度误差仍存在。无真值数据的起终高度是估计变化，不能当成真实误差。200730 起始站立、结束坐姿，合理存在高度下降。

版本源码说明和逐文件修改原因位于 `doc/v1.37/`；运行包中的 `manifest.json`、`release_validation/` 记录实际提交、重建库哈希及完整数据复验。

## 预发布说明

本次为 rc1，存在已记录的 TEST_data 跟踪回归，不自动替换 v1.36。条件几何不确定度存在缓存生命周期缺陷，默认关闭；完整协方差和原有高度问题没有宣称解决。详见 doc/v1.37/release_results/。
