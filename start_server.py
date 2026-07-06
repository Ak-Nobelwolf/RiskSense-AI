import subprocess
import sys
import signal
import time
import os

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8503"],
    cwd=os.path.dirname(os.path.abspath(__file__)),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
)

# Wait a moment for startup
time.sleep(3)
print(f"Server PID: {proc.pid}")
