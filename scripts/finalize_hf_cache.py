import os
from pathlib import Path

blobs_dir = Path(r"D:\hf_cache\hub\models--Qwen--Qwen2.5-3B-Instruct\blobs")
snap_dir = Path(r"D:\hf_cache\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\aa8e72537993ba99e69dfaafa59ed015b17504d1")

shard1_incomplete = blobs_dir / "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2.incomplete"
shard1_target = blobs_dir / "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2"

shard2_incomplete = blobs_dir / "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d.incomplete"
shard2_target = blobs_dir / "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d"

if shard1_incomplete.exists():
    print(f"Renaming shard1: {shard1_incomplete.name} -> {shard1_target.name}")
    shard1_incomplete.rename(shard1_target)

if shard2_incomplete.exists():
    print(f"Renaming shard2: {shard2_incomplete.name} -> {shard2_target.name}")
    shard2_incomplete.rename(shard2_target)

# Link or copy into snapshot
link1 = snap_dir / "model-00001-of-00002.safetensors"
link2 = snap_dir / "model-00002-of-00002.safetensors"

if not link1.exists() and shard1_target.exists():
    try:
        os.link(str(shard1_target), str(link1))
        print("Created hardlink for shard 1")
    except Exception as e:
        print("Hardlink failed, creating symlink:", e)
        link1.symlink_to(shard1_target)

if not link2.exists() and shard2_target.exists():
    try:
        os.link(str(shard2_target), str(link2))
        print("Created hardlink for shard 2")
    except Exception as e:
        print("Hardlink failed, creating symlink:", e)
        link2.symlink_to(shard2_target)

print("Snapshot directory contents:")
for f in snap_dir.iterdir():
    print(f"  {f.name}: {f.stat().st_size} bytes")
