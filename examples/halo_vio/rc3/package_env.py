from pathlib import Path
import json,os
ROOT=Path('/nas/shared_halo/baidx/cuVSLAM')
def environment(package,profile):
    package=Path(package)
    spec=json.loads((package/'profiles.json').read_text())[profile]
    deps=json.loads((package/'DEPENDENCIES.json').read_text())
    env={k:v for k,v in os.environ.items()if not k.startswith(('VIO_','CUVSLAM_','ROOM_','SLAM_','PGO_','PYTHONPATH'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']}
    env.update(spec['flags'])
    env.update(PYTHONPATH=str(package/'runtime')+':'+deps['shared_python_deps'],LD_LIBRARY_PATH=':'.join([str(package/'runtime/cuvslam'),str(package/'vendor/blas'),deps['shared_support_lib'],deps['cuda_lib']]),VIO_JOINT_BLAS_LIBRARY=str(package/'vendor/blas'/deps['blas_filename']),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VIO_JOINT_BLAS_THREADS='2')
    return env
