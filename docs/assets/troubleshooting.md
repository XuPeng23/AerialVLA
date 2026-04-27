# Troubleshooting

## AirSim Communication Latency Bug (simGetImages extremely slow)

During the evaluation phase, if you notice that the simulator runs extremely slow and profiling shows that `simGetImages` takes roughly 4~5 seconds per step, you are likely encountering a severe communication bug caused by `msgpack-rpc-python` dependency conflicts and decoding overhead in the C-S architecture.

We successfully reduced this latency to **~0.2s** by applying the following two steps simultaneously:

### Step 1: Clean Reinstallation of Dependencies
First, uninstall the problematic packages and reinstall them with strict versions, using a modified `msgpack-rpc-python` package provided in our repository.

```bash
# 1. Uninstall current dependencies
pip uninstall msgpack-python msgpack msgpack-rpc-python airsim tornado -y

# 2. Reinstall with strict versions and the fixed RPC library
pip install tornado==4.5.3
# Download the zip file from our repository and install it
pip install msgpack-rpc-python-fix-msgpack-dep.zip
pip install airsim==1.8.1 --no-build-isolation

# 3. Force reinstall msgpack to reset the core
pip uninstall msgpack-python -y
pip install --force-reinstall msgpack==1.1.2
```
*(Note: Replace `path/to/your/` with the actual local path where you downloaded the zip file).*

### Step 2: Modify AirSim Client Encoding (Crucial)
Locate your installed AirSim `client.py` file (typically found at `miniconda3/envs/YOUR_ENV_NAME/lib/python3.X/site-packages/airsim/client.py`).

Find the `__init__` method of the `VehicleClient` class and **remove the UTF-8 encoding parameters**. 

**Change this line:**
```python
self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout = timeout_value, pack_encoding = 'utf-8', unpack_encoding = 'utf-8')
```
**To this:**
```python
self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout = timeout_value)
```

After applying both steps, restart your environment. The communication latency should return to normal.

