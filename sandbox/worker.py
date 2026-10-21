"""Container entry point. Never import or run this worker on the application host.

The isolation boundary is Docker, not Python restrictions. The child may use arbitrary
Python; all its files and processes remain inside the disposable container.
"""
import base64
import json
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys

PROTOCOL_VERSION = 1
MAX_LOG = 64 * 1024
MAX_RESULT = 256 * 1024
MAX_PNG = 2 * 1024 * 1024
MAX_TOTAL_PNG = 4 * 1024 * 1024


def read_regular(path: Path, limit: int) -> bytes:
    # Reject symlinks, devices, FIFOs and oversized files before opening them.
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError(f'Invalid or oversized output: {path.name}')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('Output is not a regular file.')
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f'Oversized output: {path.name}')
    return data


def log_excerpt(path: Path) -> str:
    with path.open('rb') as stream:
        data = stream.read(MAX_LOG)
    # Replacement characters can expand a byte, so bound UTF-8 again afterwards.
    return data.decode('utf-8', errors='replace').encode()[:MAX_LOG].decode('utf-8', errors='ignore')


def main() -> None:
    cpu_seconds = int(sys.argv[1])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    response = dict(protocol_version=PROTOCOL_VERSION, status='error', result=None,
                    stdout='', stderr='', error=None, artifacts=[])
    try:
        with open('/tmp/stdout.log', 'wb') as stdout, open('/tmp/stderr.log', 'wb') as stderr:
            completed = subprocess.run([sys.executable, '-I', '/input/analysis.py'],
                                       stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       cwd='/output', check=False)
        response['stdout'] = log_excerpt(Path('/tmp/stdout.log'))
        response['stderr'] = log_excerpt(Path('/tmp/stderr.log'))
        if completed.returncode:
            response['error'] = f'Python exited with status {completed.returncode}; inspect stderr.'
        else:
            response['result'] = json.loads(read_regular(Path('/output/result.json'), MAX_RESULT))
            entries = []
            with os.scandir('/output') as scan:
                for entry in scan:
                    entries.append(entry.name)
                    if len(entries) > 32:
                        raise ValueError('Too many output files; maximum 32.')
            png_names = sorted(name for name in entries if name.endswith('.png'))
            if len(png_names) > 4:
                raise ValueError('Too many charts; maximum 4.')
            total = 0
            for name in png_names:
                data = read_regular(Path('/output', name), MAX_PNG)
                total += len(data)
                if total > MAX_TOTAL_PNG:
                    raise ValueError('Chart output exceeded 4 MB.')
                response['artifacts'].append(dict(name=name, data_base64=base64.b64encode(data).decode('ascii')))
            response['status'] = 'ok'
    except (OSError, ValueError, RecursionError) as exc:
        response.update(status='error', error=str(exc)[:4000], result=None, artifacts=[])
    # Serialization failures (NaN, recursion) are protocol errors, never findings.
    try:
        encoded = json.dumps(response, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        encoded = json.dumps(dict(protocol_version=PROTOCOL_VERSION, status='error', result=None,
                                  stdout='', stderr='', error='Outputs contain non-JSON values.', artifacts=[]))
    print(encoded, flush=True)


if __name__ == '__main__':
    main()
