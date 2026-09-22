#!/usr/bin/env python3
"""One bounded passive USERD capture on an idle, explicitly selected A100."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=Path, default=Path('build-userd'))
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
build, output = a.build.resolve(), a.output.resolve()
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists() or Path(str(output) + '.userd').exists():
    raise SystemExit('Refusing to replace existing evidence')
query = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
    '--query-gpu=uuid,name,driver_version,mig.mode.current,compute_mode', '--format=csv,noheader'], text=True, timeout=10).strip()
uuid, name, driver, mig, mode = [x.strip() for x in query.split(',')]
if driver != '595.58.03' or 'A100' not in name or mig != 'Disabled' or mode != 'Default':
    raise SystemExit('Expected 595.58.03, non-MIG A100, default compute mode')
active = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                                  '--format=csv,noheader'], text=True, timeout=10).strip()
if active:
    raise SystemExit('Selected GPU has an active compute process; no probe launched')
if os.environ.get('LD_PRELOAD'):
    raise SystemExit('Use a clean LD_PRELOAD to preserve the reviewed observer -> bridge -> libc chain')
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = uuid
env['LD_PRELOAD'] = str(build / 'libuserd_observer.so')
command = [str(build / 'context_ping_pong'), '--mode', 'userd-map', '--cpu-a', '0', '--cpu-b', '1',
           '--cpu-control', '2', '--query-timeout-ms', '3000', '--output', str(output)]
# Cache paths are retained locally for exact reproducibility; never copied into
# the deidentified public outputs.
cache = dict(line.split('=', 1) for line in (build / 'CMakeCache.txt').read_text().splitlines()
             if '=' in line and not line.startswith(('#', '//')))
bridge = Path(cache['BENCH_RM_LIBRARY:FILEPATH'])
files = [build / 'context_ping_pong', build / 'libuserd_observer.so', build / 'kernels.cubin', bridge]
manifest = {'gpu_preflight': query, 'command': command, 'timeout_s': 45,
            'CUDA_VISIBLE_DEVICES': uuid, 'LD_PRELOAD': env['LD_PRELOAD'],
            'sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
with Path(str(output) + '.launch.json').open('x') as f:
    json.dump(manifest, f, indent=2); f.write('\n')
start = time.monotonic_ns()
with Path(str(output) + '.stdout').open('x') as out, Path(str(output) + '.stderr').open('x') as err:
    try:
        result = subprocess.run(command, env=env, stdout=out, stderr=err, timeout=45)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
with Path(str(output) + '.exit.json').open('x') as f:
    json.dump({'exit_code': code, 'elapsed_ns': time.monotonic_ns() - start}, f)
print('Probe exit:', code)
print(Path(str(output) + '.stderr').read_text())
raise SystemExit(code)
