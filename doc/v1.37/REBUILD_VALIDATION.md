# GitHub 提交重建与完整复验

完整算法验证提交：`f74e2238d55f32620de3e7c44eda6f1a8f3140bf`；核心 SHA256：`b4251e62a9f390aff1f976ae18c03c233b1bfb1946e3c3ad6aecb5742109ca90`。从 GitHub 干净检出，原始 cuNLS 固定归档重新应用补丁；C++ 与 Python 绑定同源构建。

- CTest 22/22 目标通过，发布入口 6/6 通过；Python API 74 项、73 通过、1 按条件跳过。
- 发布默认配置完整跑 19 组：内部无真值 13 组、room1–room6 六组。
- 同一库再跑六组 room 条件几何不确定度开关对照，只有该实验开关不同。默认保持关闭；导航先验不替换。
- 每组每配置一次全程，不宣称多轮稳定性。本轮六组与实验候选的逐位一致性见 release_results/ROOM_RESULTS.json。
- 无真值高度是估计变化，不是误差；重力初始化前属于视觉阶段。表中显式记录重力覆盖、失跟、清图及重力中断。

本提交仅追加文档和结果，算法源文件与上述完整验证提交一致。正式 tag 对应的最终嵌入版本、共享库哈希和两组完整重放验证，记录在共享运行包 manifest.json 与 release_validation/。

运行目录：`/nas/shared/baidx/cuvslam_runs/runtime_reduced_v1.37`。

- release_results/NO_GT_REPORT.md：13 组内部统计；NO_GT_RESULTS.json 含峰值对应距离和真实时间差等。
- release_results/index.html：下拉选择轨迹点图、高度和速度。
- release_results/ROOM_GEOMETRY_REPORT.md：六组条件几何不确定度开关对照。
- release_results/ROOM_GEOMETRY_RESULTS.json：共同有效帧误差、初始化时间、倾角、失跟等完整指标。

## 预发布限制

本次以 runtime_reduced_v1.37-rc1 发布，不标为稳定替代版。162436/180123 在默认关闭几何实验时出现失跟/清图，已与 v1.36 同输入重新核对；详见 release_results/LEGACY_RECHECK_COMPLETE.json 和 NO_GT_REPORT.md。条件几何模块另有独立用例复现的缓存失效缺陷，默认关闭，详见 release_results/GEOMETRY_AUDIT.md；它尚未被证明是实际 room 退化主因。
