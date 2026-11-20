# v1.37 逐文件修改及原因

| 文件 | 修改 | 原因 |
|---|---|---|
| `cmake/ext/cunls.cmake` | 注册并幂等应用八个依赖修复补丁 | 保证下载源和源覆盖构建得到相同代码 |
| `cmake/ext/cunls_cauchy_verified.patch` | 修正 Cauchy 一二阶导数及说明 | 代价和导数必须一致 |
| `cmake/ext/cunls_evaluation_status_verified.patch` | 传播因子、损失及包装层失败 | 失败求解不得继续发布或回写 |
| `cmake/ext/cunls_pcg_workspace_verified.patch` | 初始化 PCG 工作区 | 消除未初始化读取 |
| `cmake/ext/cunls_pnp_information_verified.patch` | 收录已使用的 PnP 观测加权和无效投影处理 | 使固定版本依赖能从原始归档重现 |
| `cmake/ext/cunls_sparse_metadata_verified.patch` | 纯值归约并清零覆盖写输出 | 避免元组填充及稀疏乘法未初始化读取 |
| `doc/v1.37/FILE_CHANGES.md` | 新增该版本说明、结果或可视化 | 明确修复范围、未解决限制与可追溯验证 |
| `doc/v1.37/INFORMATION_SOURCE_REFERENCE.json` | 新增该版本说明、结果或可视化 | 明确修复范围、未解决限制与可追溯验证 |
| `doc/v1.37/README.md` | 新增该版本说明、结果或可视化 | 明确修复范围、未解决限制与可追溯验证 |
| `doc/v1.37/ROOM_REPORT.md` | 新增该版本说明、结果或可视化 | 明确修复范围、未解决限制与可追溯验证 |
| `doc/v1.37/ROOM_RESULTS.json` | 新增该版本说明、结果或可视化 | 明确修复范围、未解决限制与可追溯验证 |
| `doc/v1.37/ROOM_TRAJECTORIES.png` | 新增该版本说明、结果或可视化 | 明确修复范围、未解决限制与可追溯验证 |
| `doc/v1.37/ROOM_VERTICAL.png` | 新增该版本说明、结果或可视化 | 明确修复范围、未解决限制与可追溯验证 |
| `doc/v1.37/audit/CORRECTED_MATH_PROOF.jsonl` | 归档对应公式、回归或六组真值对照证据 | 保留来源和实验哈希；最终重建结果另行记录 |
| `doc/v1.37/audit/CTEST_SAFE_FINAL.log` | 归档对应公式、回归或六组真值对照证据 | 保留来源和实验哈希；最终重建结果另行记录 |
| `doc/v1.37/audit/GPU_INITCHECK_CLOSED.log` | 归档对应公式、回归或六组真值对照证据 | 保留来源和实验哈希；最终重建结果另行记录 |
| `doc/v1.37/audit/GPU_RACECHECK_CLOSED.log` | 归档对应公式、回归或六组真值对照证据 | 保留来源和实验哈希；最终重建结果另行记录 |
| `doc/v1.37/audit/MANIFEST.json` | 归档对应公式、回归或六组真值对照证据 | 保留来源和实验哈希；最终重建结果另行记录 |
| `doc/v1.37/audit/ORIGINAL_MATH_PROOF.jsonl` | 归档对应公式、回归或六组真值对照证据 | 保留来源和实验哈希；最终重建结果另行记录 |
| `doc/v1.37/audit/PNP_INITCHECK_CLOSED.log` | 归档对应公式、回归或六组真值对照证据 | 保留来源和实验哈希；最终重建结果另行记录 |
| `doc/v1.37/audit/PNP_MEMCHECK_CLOSED.log` | 归档对应公式、回归或六组真值对照证据 | 保留来源和实验哈希；最终重建结果另行记录 |
| `doc/v1.37/audit/PRODUCTION_LM_AFTER.xml` | 归档对应公式、回归或六组真值对照证据 | 保留来源和实验哈希；最终重建结果另行记录 |
| `doc/v1.37/audit/REBUILD_SOURCE_CHECK.json` | 归档对应公式、回归或六组真值对照证据 | 保留来源和实验哈希；最终重建结果另行记录 |
| `examples/halo_vio/README.md` | 记录使用方法、默认配置、旋转检查与模型限制 | 不将本版描述为全部数学或高度问题已解决 |
| `examples/halo_vio/run.py` | 版本改为 v1.37 并启用条件视觉信息输出 | 启动器使用六组验证通过的固定配置 |
| `examples/halo_vio/test_release.py` | 验证发布配置不继承源射线、几何或先验实验 | 保证使用者环境不会改变发布默认 |
| `libs/common/rotation_utils.h` | double SVD 正交化并保留非有限状态 | 降低旋转正交化误差且不伪造有效输入 |
| `libs/common/test/rotation_utils_test.cpp` | 增加缩放、反射和非有限旋转回归 | 保持原精度阈值验证修复 |
| `libs/imu/imu_preintegration.cpp` | 闭式积分按数值范围选择级数或解析 Gamma | 避免固定 Taylor 阶数在大角增量失效 |
| `libs/imu/inertial_optimization.cpp` | 生产 LM 使用对称二次模型并核对真实运行缩放 | 预测下降量与下三角对称求解一致 |
| `libs/imu/inertial_optimization.h` | 提供测试所需的生产求解访问声明 | 验证真实优化器而非只验证辅助公式 |
| `libs/imu/test/CMakeLists.txt` | 注册初始化和闭式积分数学测试 | 构建门禁覆盖新增修复 |
| `libs/imu/test/closed_form_math_test.cpp` | 独立积分验证位移、速度和偏置雅可比 | 区分公式错误和 float 有限差分误差 |
| `libs/imu/test/init_math_test.cpp` | 核对生产梯度、Hessian、鲁棒损失和 LM 预测 | 防止经验证辅助函数未接入生产 |
| `libs/pipelines/gpu_window/window_backend.cpp` | 实际鱼眼像素验收及 FP64 后验接入 | 前台与地图支持口径一致并避免后验精度回退 |
| `libs/pipelines/gpu_window/window_fp64.cpp` | double 后验及可选源射线坐标处理 | 保留弱约束并避免 XYZ 被重复解释为深度变量 |
| `libs/pipelines/gpu_window/window_fp64.h` | 声明 FP64 后验接口 | 保持评价数据类型明确 |
| `libs/pipelines/gpu_window/window_fp64_gpu.cu` | 可选源射线参数化的 GPU 链式雅可比 | 与 CPU 和物理点坐标保持一致，默认实验关闭 |
| `libs/pipelines/gpu_window/window_math.h` | 源相机固定坐标图的正深度参数化 | 允许独立受控实验，不强锁带噪源像素 |
| `libs/pipelines/gpu_window/window_test.cpp` | 补充 Schur/QR、源射线、FP64 后验与边缘化回归 | 检验线性系统与先验的数值等价性 |
| `libs/pipelines/gpu_window/window_types.h` | 可选源射线数据及快照版本兼容 | 使输入、状态和回放格式有明确语义 |
| `libs/pipelines/gpu_window/window_validation.h` | 验证可选坐标图和有限正深度 | 非法实验输入不能绕过验证 |
| `libs/pipelines/imu_fusion_context.cpp` | 视觉信息转换到 IMU 局部坐标，精化时同步信息，保留导航种子；恢复前校验并显式转换相机 ID | 条件视觉信息不能直接替代独立导航先验 |
| `libs/pipelines/imu_fusion_context.h` | 可选条件几何精化缓存随融合状态管理 | 避免跨地图复用不兼容历史信息 |
| `libs/pnp/CMakeLists.txt` | 编译条件几何精化实现 | 保持模块依赖和构建一致 |
| `libs/pnp/cauchy_parameters.h` | 删除错误梯度兼容归一化并说明有效参数模式 | 已修正公式不再被旧控制分支抵消 |
| `libs/pnp/geometry_profile.cpp` | 历史点先验与跨相机共享潜变量 QR 消元，接受时同步信息 | 避免把相关观测当独立点误差；实验默认关闭 |
| `libs/pnp/geometry_profile.h` | 声明可选精化和信息输出接口 | 明确固定历史位姿条件及拒绝语义 |
| `libs/pnp/multisensor_pose_estimator.cpp` | 输出基于标定噪声的条件视觉信息并保持事务提交 | 信息保留交叉项且不误用求解平均系数 |
| `libs/pnp/pixel_support.h` | 集中实际相机像素支持判断 | 鱼眼边缘不使用针孔残差乘焦距近似 |
| `libs/pnp/test/CMakeLists.txt` | 注册几何、失败传播、稀疏乘法及信息矩阵测试 | 把已确认问题加入可重复回归 |
| `libs/pnp/test/geometry_profile_test.cpp` | 验证消元、实际雅可比、鱼眼像素及信息提交原子性 | 拒绝不能改变均值或信息 |
| `libs/pnp/test/multisensor_pose_estimator_test.cpp` | 验证 Cauchy 导数、生产信息正定性和交叉项 | 防止回到单位阵占位或错误导数 |
| `libs/pnp/test/residual_failure_test.cpp` | 注入五类因子/包装层失败 | 验证真实求解提前停止且状态不变 |
| `libs/pnp/test/sparse_product_test.cpp` | 新分配/NaN 输出与两类稀疏尺寸路径回归 | 确认覆盖写结果正确且无未初始化依赖 |
| `libs/pnp/test/visual_information_test.cpp` | 独立差分验证信息变换、杆臂、非有限和可加性 | 将信息尺度和坐标数学加入门禁 |
| `libs/pnp/visual_information.h` | 局部信息合同变换及相对 IRLS 衰减 | 求逆、杆臂及测量信息尺度有明确数学定义 |

## 完整复验追加

| 文件 | 修改及原因 |
|---|---|
| `doc/v1.37/FILE_CHANGES.md` | 补充本阶段逐文件修改原因，保持提交可审查 |
| `doc/v1.37/README.md` | 记录完整复验结果、限定结论并提供追溯入口 |
| `doc/v1.37/REBUILD_VALIDATION.md` | 记录完整复验结果、限定结论并提供追溯入口 |
| `doc/v1.37/release_results/GEOMETRY_AUDIT.json` | 保留同版本、同输入的机器可读统计和误差口径 |
| `doc/v1.37/release_results/GEOMETRY_AUDIT.md` | 记录完整复验结果、限定结论并提供追溯入口 |
| `doc/v1.37/release_results/LEGACY_RECHECK_COMPLETE.json` | 保留同版本、同输入的机器可读统计和误差口径 |
| `doc/v1.37/release_results/NO_GT_REPORT.md` | 记录完整复验结果、限定结论并提供追溯入口 |
| `doc/v1.37/release_results/NO_GT_RESULTS.json` | 保留同版本、同输入的机器可读统计和误差口径 |
| `doc/v1.37/release_results/ROOM_GEOMETRY_REPORT.md` | 记录完整复验结果、限定结论并提供追溯入口 |
| `doc/v1.37/release_results/ROOM_GEOMETRY_RESULTS.json` | 保留同版本、同输入的机器可读统计和误差口径 |
| `doc/v1.37/release_results/ROOM_GEOMETRY_VERTICAL.png` | 绘制完整轨迹、高度或速度对照，供检查连续性 |
| `doc/v1.37/release_results/ROOM_RESULTS.json` | 保留同版本、同输入的机器可读统计和误差口径 |
| `doc/v1.37/release_results/geometry_audit/CACHE_AUDIT.json` | 保留同版本、同输入的机器可读统计和误差口径 |
| `doc/v1.37/release_results/geometry_audit/CACHE_TEST.log` | 记录完整复验结果、限定结论并提供追溯入口 |
| `doc/v1.37/release_results/geometry_audit/build_audit.py` | 保存独立缓存缺陷复现代码，不注册为已通过的默认 CI 测试 |
| `doc/v1.37/release_results/geometry_audit/cache_test.cpp` | 保存独立缓存缺陷复现代码，不注册为已通过的默认 CI 测试 |
| `doc/v1.37/release_results/index.html` | 提供下拉数据选择和本地结果浏览 |
| `doc/v1.37/release_results/plots/halo851_152938.png` | 绘制完整轨迹、高度或速度对照，供检查连续性 |
| `doc/v1.37/release_results/plots/halo851_154711.png` | 绘制完整轨迹、高度或速度对照，供检查连续性 |
| `doc/v1.37/release_results/plots/halo851_155031.png` | 绘制完整轨迹、高度或速度对照，供检查连续性 |
| `doc/v1.37/release_results/plots/halo851_155625.png` | 绘制完整轨迹、高度或速度对照，供检查连续性 |
| `doc/v1.37/release_results/plots/halo851_155811.png` | 绘制完整轨迹、高度或速度对照，供检查连续性 |
| `doc/v1.37/release_results/plots/halo851_170302.png` | 绘制完整轨迹、高度或速度对照，供检查连续性 |
| `doc/v1.37/release_results/plots/test_162436.png` | 绘制完整轨迹、高度或速度对照，供检查连续性 |
| `doc/v1.37/release_results/plots/test_180123.png` | 绘制完整轨迹、高度或速度对照，供检查连续性 |
| `doc/v1.37/release_results/plots/validation_195610.png` | 绘制完整轨迹、高度或速度对照，供检查连续性 |
| `doc/v1.37/release_results/plots/validation_200730.png` | 绘制完整轨迹、高度或速度对照，供检查连续性 |
| `doc/v1.37/release_results/plots/validation_201148.png` | 绘制完整轨迹、高度或速度对照，供检查连续性 |
| `doc/v1.37/release_results/plots/validation_202341.png` | 绘制完整轨迹、高度或速度对照，供检查连续性 |
| `doc/v1.37/release_results/plots/validation_203017.png` | 绘制完整轨迹、高度或速度对照，供检查连续性 |
| `examples/halo_vio/README.md` | 将 CLI 版本标为 rc1 并说明已确认回归，避免误标稳定版 |
| `examples/halo_vio/run.py` | 将 CLI 版本标为 rc1 并说明已确认回归，避免误标稳定版 |
