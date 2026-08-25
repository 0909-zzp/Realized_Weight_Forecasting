"""启动 factor_bic_full.py 为独立后台进程 (Windows)"""
import subprocess, sys, os
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / 'factor_bic_full.py'
LOGFILE = Path(__file__).resolve().parent.parent / 'factor_bic_full_stdout.log'

# Windows: 独立进程, 脱离父进程
CREATE_FLAGS = 0x00000008  # DETACHED_PROCESS

proc = subprocess.Popen(
    [sys.executable, '-u', str(SCRIPT)],
    stdout=open(str(LOGFILE), 'w', encoding='utf-8'),
    stderr=subprocess.STDOUT,
    creationflags=CREATE_FLAGS,
    cwd=str(Path(__file__).resolve().parents[2]),
    close_fds=True,
)

print(f'后台进程已启动: PID={proc.pid}')
print(f'日志文件: {LOGFILE}')
print(f'查看进度: tail -f "{LOGFILE}"')
print(f'清理checkpoint后重启: del "{Path(SCRIPT).parent.parent / "factor_bic_checkpoint.npz"}"')
