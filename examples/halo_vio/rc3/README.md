# runtime_reduced_v1.37-rc3

本入口固定新封存候选的实际核心及配置；历史恢复和PGO默认关闭，无掩模，原生鱼眼。它与父目录旧版HALO兼容启动器的配置不同，rc3复现请使用本目录入口。

```bash
python3 run.py --version
python3 run.py --verify-only
python3 run.py -- tumvi --source /nas/shared_halo/baidx/cuVSLAM/cuvslam_runs/v135_r1_validation_20261119/truth_data/dataset-room6_512_16 --output /nas/shared_halo/baidx/cuVSLAM/cuvslam_runs/candidate_validation/rc3_room6_new --sequence room6 --audit
```

运行库来自已封存的`correctness_candidate_20261126`，核心SHA256为`6c19a32afd35d42fc7bdfd4b9554fb10141afb5bf131e77fa3e4826c6b657970`。入口核对逐文件封存和配置，不复制或覆盖运行库。可用`--package`指定相同封存格式与核心的其他共享路径。输出必须位于项目根目录下、封存包之外，且目录全新。当前cp310绑定要求Linux/CUDA12.1兼容环境与Python3.10。

`--profile observed_history`仅为显式历史条件恢复实验，不代表已有六组rc3历史开启验收。几何有限域、块迹LM及新的无穷远/非线性历史原型不默认启用。

从本提交重建后端（保留原封存运行包）：

```bash
python3 rebuild.py --frozen-package /nas/shared_halo/baidx/cuVSLAM/cuvslam_runs/correctness_candidate_20261126 --output /nas/shared_halo/baidx/cuVSLAM/cuvslam_runs/candidate_builds/rc3_new_build
python3 -B test_sealing.py
```

该重建编译Git源码的三个后端实现，复用哈希校验的静态依赖与原Python绑定，不是全项目从零构建。新构建的完整性、固定图回归和轨迹一致性需另行验收；`BUILD.json`记录新库哈希和实际Git提交。禁止在封存包内构建或覆盖任何已有目录。

逐文件改动原因和六组精度对照见`doc/v1.37/rc3/`。room2/4/5/6仍有ATE退化，rc3用于可追溯比较，不能认定为全面精度升级。
