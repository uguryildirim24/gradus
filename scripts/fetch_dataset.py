"""Wait for an Adaption dataset to finish, then download it to data/adaption/<id>/."""
import os, sys, inspect
from pathlib import Path
from adaption import Adaption
os.environ.setdefault("ADAPTION_API_KEY", Path("~/.config/adaption/api_key").expanduser().read_text().strip())
c = Adaption()
did = sys.argv[1]
print(inspect.signature(c.datasets.wait_for_completion)); print(inspect.signature(c.datasets.download))
print("DONE:", c.datasets.wait_for_completion(did))
out = Path("data/adaption") / did; out.mkdir(parents=True, exist_ok=True)
res = c.datasets.download(did)
data = res.read() if hasattr(res, "read") else (res.content if hasattr(res, "content") else res)
if isinstance(data, (bytes, bytearray)):
    (out / "download.bin").write_bytes(data); print("bytes", len(data))
else:
    print("download returned", type(data), str(data)[:2000])
print("STATUS:", c.datasets.get_status(did))
try: print("EVAL:", c.datasets.get_evaluation(did))
except Exception as e: print("eval err", e)
