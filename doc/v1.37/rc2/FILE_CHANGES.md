# 逐文件变更与原因

- `libs/pnp/geometry_profile.cpp`：取消仅以最后关键帧时间戳判定缓存有效的复用，每调用重新形成历史先验；诊断模式记录同时间戳的先验内容变化。增加显式均值模型入口和同帧影子对照，原条件模型保持默认。PreserveMap仅改变先验中心，不更改信息矩阵；不修改公共地图，不调整验收阈值。影子分支不反馈状态。
- `libs/pnp/geometry_profile.h`：新增内部PriorMean枚举及显式refineWithPriorMean入口；原refine签名及对象字段保持，正常默认调用保持条件模型。
- `libs/pnp/test/geometry_cache_fix_test.cpp`：在保持最后KF时间戳的条件下分别修改历史位姿、地图点和外参，核对持久实例与全新实例位姿及信息一致；测试两个均值模型的重复性、不回写地图；用FP64验证居中高斯先验的QR消元与保留跨观测相关性的完整协方差解一致。
- `libs/pnp/test/CMakeLists.txt`：将新增回归纳入标准PnP/CTest构建。

修复范围：消除跨调用先验残留；不提供图级跨线程原子快照，不补全历史位姿协方差，不改变缺先验点回退，也不扩展到初始化后的滑窗。当前调用者必须序列化地图写入。

诊断环境变量：`VIO_GEOMETRY_MEAN_AUDIT=1`在同一调用中先运行不提交的PreserveMap分支，再运行实际提交的Conditional分支。新接口为内部实验接口；不存在默认开启或发布版替换。
