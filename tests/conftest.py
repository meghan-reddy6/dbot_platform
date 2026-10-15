import atexit
import os
import shutil
import tempfile

# Must run before app modules are imported: settings and DB paths resolve from it.
if "DESKBOT_DATA_DIR" not in os.environ:
    _data_dir = tempfile.mkdtemp(prefix="deskbot-test-")
    os.environ["DESKBOT_DATA_DIR"] = _data_dir
    atexit.register(shutil.rmtree, _data_dir, ignore_errors=True)
