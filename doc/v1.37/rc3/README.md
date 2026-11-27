# runtime_reduced_v1.37-rc3

将2026-11-26已封存且完成room1–room6全程复验的新候选固定为rc3。Git基线是rc2的b188158；默认配置对应6c19a32a核心，无掩模、原生鱼眼、无PGO，新增历史条件恢复默认关闭。

- [逐文件修改与原因](FILE_CHANGES.md)
- [rc2/rc3代码、模型和六组完整精度比较](COMPARISON.md)
- [六组完整轨迹、高度、误差和速度图](comparison.html)：下载HTML即可离线查看，不依赖私有NAS图片路径。
- [指标原始数值](ROOM_RESULTS.json)：SE(3)对齐、不改尺度，包含GT覆盖范围、ATE/RPE/XY/Z误差和峰值。
- [源码SHA256](SOURCE_MANIFEST.json)与[版本范围](VERSION.json)
- [运行和后端重建入口](../../../../examples/halo_vio/rc3/README.md)

rc3的核心变化是完整联合初始化/滑窗后验、相关性与信息归属、先验坐标连续性、FP64加速及前端仿射一致性。它不是在rc2相同因子图上单纯更换CPU/GPU线性求解器。

六组均完整跑完且没有失跟、清图或重力中断；ATE仅room1、room3改善。room2/4/5/6的ATE退化、room5高度误差及room6水平误差仍存在。15次历史恢复诊断来自前一a962核心；该实验不冒充rc3的历史开启全程结果。后续原始非线性历史、条件均值回代和精确无穷远诊断原型没有纳入此提交。

构建边界：提交含精确后端覆盖和按构建哈希找回的前端源码，以及其底层源码上下文。已验证运行核心复用了封存静态依赖；本次不宣称完成全项目从零重建。提交后重建及回归材料记录在共享盘`/nas/shared_halo/baidx/cuVSLAM/cuvslam_runs/rc3_release_20261127/`，原运行包保持不变。
