from pathlib import Path
import subprocess,json,shlex,os
r=Path('/nas/shared/baidx/cuvslam_runs/geometry_uncertainty_audit_20261120');work=Path('/tmp/geometry_uncertainty_audit_20261120');work.mkdir(exist_ok=True);build=Path('/tmp/runtime_reduced_v137_release_20261120/build')
commands=json.loads((build/'compile_commands.json').read_text());row=next(x for x in commands if x['file'].endswith('/libs/pnp/test/geometry_profile_test.cpp'))
cmd=shlex.split(row['command']);cmd[cmd.index('-o')+1]=str(work/'cache_test.o');cmd[cmd.index('-c')+1]=str(r/'cache_test.cpp');subprocess.run(cmd,cwd=row['directory'],check=True,stdout=(r/'compile.log').open('w'),stderr=subprocess.STDOUT)
link=subprocess.check_output(['ninja','-C',str(build),'-t','commands','bin/pnp_test'],text=True).splitlines()[-1];tokens=shlex.split(link);start=next(i for i,x in enumerate(tokens)if x.endswith('/c++'));tokens=tokens[start:];tokens=tokens[:tokens.index('&&')]if'&&'in tokens else tokens
out=[];inserted=False
for t in tokens:
 if t.endswith('.o'):
  if not inserted:out.append(str(work/'cache_test.o'));inserted=True
 else:out.append(t)
out[out.index('-o')+1]=str(work/'cache_test');subprocess.run(out,cwd=build,check=True,stdout=(r/'link.log').open('w'),stderr=subprocess.STDOUT)
e=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH='/nas/shared/baidx/cuvslam_runs/runtime_reduced_v1.37/lib')
with(r/'CACHE_TEST.log').open('w')as log:p=subprocess.run([str(work/'cache_test'),'--gtest_filter=GeometryCacheAudit.*'],env=e,stdout=log,stderr=subprocess.STDOUT)
print(json.dumps(dict(returncode=p.returncode,expected='0 if current-map cache invalidation is correct; nonzero records a reproducible defect')))
