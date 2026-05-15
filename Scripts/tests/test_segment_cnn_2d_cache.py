from __future__ import annotations

import numpy as np
import pandas as pd

from models import segment_cnn_2d as cnn2d


def test_sharded_scalogram_cache_round_trip(tmp_path):
    cache = cnn2d.ShardedScalogramCache(tmp_path)
    first = np.full(cnn2d.IMAGE_SIZE, 0.25, dtype=np.float32)
    second = np.full(cnn2d.IMAGE_SIZE, 0.75, dtype=np.float32)

    cache.write_shard([10, 20], [first, second])
    reloaded = cnn2d.ShardedScalogramCache(tmp_path)

    assert reloaded.has_segment(10)
    assert reloaded.has_segment(20)
    assert not reloaded.has_segment(30)
    np.testing.assert_array_equal(reloaded.load(10), first)
    np.testing.assert_array_equal(reloaded.load(20), second)


def test_load_or_compute_prefers_shard_then_reads_legacy_cache(tmp_path, monkeypatch):
    sharded = cnn2d.ShardedScalogramCache(tmp_path)
    sharded_arr = np.full(cnn2d.IMAGE_SIZE, 0.5, dtype=np.float32)
    sharded.write_shard([1], [sharded_arr])

    def fail_read_table(*args, **kwargs):
        raise AssertionError("should not read parquet when cache has the segment")

    monkeypatch.setattr(cnn2d.pq, "read_table", fail_read_table)
    np.testing.assert_array_equal(
        cnn2d._load_or_compute_scalogram(1, "unused.parquet", cache_base=tmp_path),
        sharded_arr,
    )

    legacy_arr = np.full(cnn2d.IMAGE_SIZE, 0.125, dtype=np.float32)
    np.save(tmp_path / "seg_2.npy", legacy_arr)
    np.testing.assert_array_equal(
        cnn2d._load_or_compute_scalogram(2, "unused.parquet", cache_base=tmp_path),
        legacy_arr,
    )


def test_prewarm_cache_writes_shards_not_per_segment_files(tmp_path, monkeypatch):
    class FakePool:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, args):
            return [fn(arg) for arg in args]

    class FakeTable:
        def to_pandas(self):
            rows = []
            for seg in [101, 102, 103]:
                rows.append({"segment_idx": seg, "timestamp_ms": 0, "ecg": float(seg)})
                rows.append({"segment_idx": seg, "timestamp_ms": 1, "ecg": float(seg)})
            return pd.DataFrame(rows)

    def fake_read_table(*args, **kwargs):
        return FakeTable()

    def fake_compute(ecg):
        return np.full(cnn2d.IMAGE_SIZE, ecg[0], dtype=np.float32)

    import concurrent.futures

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(cnn2d.pq, "read_table", fake_read_table)
    monkeypatch.setattr(cnn2d, "compute_scalogram", fake_compute)

    cnn2d._prewarm_cache(
        np.array([101, 102, 103], dtype=np.int64),
        "unused.parquet",
        cache_base=tmp_path,
        batch_size=2,
    )

    assert not list(tmp_path.glob("seg_*.npy"))
    reloaded = cnn2d.ShardedScalogramCache(tmp_path)
    for seg in [101, 102, 103]:
        np.testing.assert_array_equal(
            reloaded.load(seg),
            np.full(cnn2d.IMAGE_SIZE, float(seg), dtype=np.float32),
        )


def test_repack_legacy_cache_converts_existing_segment_files_to_shards(tmp_path):
    first = np.full(cnn2d.IMAGE_SIZE, 11, dtype=np.float32)
    second = np.full(cnn2d.IMAGE_SIZE, 12, dtype=np.float32)
    np.save(tmp_path / "seg_11.npy", first)
    np.save(tmp_path / "seg_12.npy", second)

    summary = cnn2d.repack_legacy_cache(
        np.array([11, 12, 13], dtype=np.int64),
        cache_base=tmp_path,
        shard_size=10,
    )

    assert summary == {"repacked": 2, "already_sharded": 0, "missing_legacy": 1}
    sharded = cnn2d.ShardedScalogramCache(tmp_path)
    np.testing.assert_array_equal(sharded.load(11), first)
    np.testing.assert_array_equal(sharded.load(12), second)
    assert not sharded.has_segment(13)
