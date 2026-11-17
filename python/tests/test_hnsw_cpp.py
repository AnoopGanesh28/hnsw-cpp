"""Tests for the hnsw_cpp Python bindings (stdlib unittest + numpy only)."""

import gc
import os
import struct
import tempfile
import threading
import unittest

import numpy as np

import hnsw_cpp

DIM = 32
N = 3000
K = 10
EF = 100
RECALL_THRESHOLD = 0.99  # observed over seeds 0-9 at ef=100: 0.9970 .. 0.9990


def dataset(seed=0, n=N, dim=DIM, nq=200):
    rng = np.random.default_rng(seed)
    return rng.random((n, dim), dtype=np.float32), rng.random((nq, dim), dtype=np.float32)


def exact_sq_dists(queries, base):
    """Independent float64 squared L2 reference, shape (nq, n)."""
    q = queries.astype(np.float64)
    b = base.astype(np.float64)
    return ((q[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)


def exact_knn(queries, base, k):
    return np.argsort(exact_sq_dists(queries, base), axis=1, kind="stable")[:, :k]


def recall(ids, truth):
    return float(np.mean([len(set(a) & set(b)) / truth.shape[1] for a, b in zip(ids, truth)]))


def same(a, b):
    """Bit-identical (ids, dists) pairs."""
    return np.array_equal(a[0], b[0]) and a[1].tobytes() == b[1].tobytes()


def build(seed=0, **kw):
    base, queries = dataset(seed)
    index = hnsw_cpp.Index(DIM, seed=seed, **kw)
    index.add(base)
    return index, base, queries


class TmpDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def path(self, name="index.hnsw"):
        return os.path.join(self.dir, name)


class SearchQuality(unittest.TestCase):
    def test_recall_and_squared_distances(self):
        index, base, queries = build()
        ids, dists = index.search(queries, K, EF)
        self.assertEqual(ids.shape, (len(queries), K))
        self.assertEqual(ids.dtype, np.int64)
        self.assertEqual(dists.dtype, np.float32)
        self.assertGreaterEqual(recall(ids, exact_knn(queries, base, K)), RECALL_THRESHOLD)
        # Distances are squared L2: equal to the float64 reference at returned ids.
        ref = exact_sq_dists(queries, base)
        np.testing.assert_allclose(dists, np.take_along_axis(ref, ids, axis=1), rtol=1e-4)
        self.assertTrue(np.all(np.diff(dists, axis=1) >= 0))

    def test_squared_not_rooted(self):
        index = hnsw_cpp.Index(2)
        index.add(np.array([[0, 0], [3, 4]], dtype=np.float32))
        ids, dists = index.search(np.zeros(2, dtype=np.float32), 2, 10)
        self.assertEqual(ids.tolist(), [[0, 1]])
        self.assertEqual(dists.tolist(), [[0.0, 25.0]])

    def test_properties_and_ids(self):
        index = hnsw_cpp.Index(4, M=8, ef_construction=50, seed=7)
        self.assertEqual((index.dim, index.M, index.ef_construction, index.seed), (4, 8, 50, 7))
        self.assertEqual(index.size, 0)
        x = np.zeros((5, 4), dtype=np.float32)
        ids = index.add(x)
        self.assertEqual(ids.dtype, np.int64)
        self.assertEqual(ids.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(index.add(x[0]).tolist(), [5])  # 1-D = one vector
        self.assertEqual(index.size, 6)
        self.assertEqual(index.add(x[:0]).shape, (0,))
        with self.assertRaises(AttributeError):
            index.size = 3

    def test_one_dimensional_query(self):
        index, base, _ = build()
        ids, dists = index.search(base[5], 3, EF)
        self.assertEqual(ids.shape, (1, 3))
        self.assertEqual(ids[0, 0], 5)
        self.assertEqual(dists[0, 0], 0.0)

    def test_k_larger_than_size_pads(self):
        index = hnsw_cpp.Index(3)
        x = np.arange(15, dtype=np.float32).reshape(5, 3)
        index.add(x)
        for threads in (1, 2):
            ids, dists = index.search(x[:2], 8, 20, threads=threads)
            self.assertEqual(ids.shape, (2, 8))
            self.assertTrue(np.all(ids[:, 5:] == -1))
            self.assertTrue(np.all(dists[:, 5:] == np.inf))
            self.assertTrue(np.all(ids[:, :5] >= 0))
            self.assertTrue(np.all(np.isfinite(dists[:, :5])))

    def test_empty_index_and_empty_queries(self):
        index = hnsw_cpp.Index(DIM)
        q = np.zeros((3, DIM), dtype=np.float32)
        for threads in (1, 2):
            ids, dists = index.search(q, 4, 10, threads=threads)
            self.assertTrue(np.all(ids == -1) and ids.shape == (3, 4))
            self.assertTrue(np.all(dists == np.inf))
        ids, dists = index.search(q[:0], 4, 10)
        self.assertEqual((ids.shape, dists.shape), ((0, 4), (0, 4)))

    def test_threads_give_identical_results(self):
        index, _, queries = build()
        ref = index.search(queries, K, EF, threads=1)
        for threads in (2, 0, 3):
            self.assertTrue(same(ref, index.search(queries, K, EF, threads=threads)), threads)
        # A pool is reused across calls and resized when the request changes.
        for threads in (2, 2, 3, 2):
            self.assertTrue(same(ref, index.search(queries, K, EF, threads=threads)))

    @unittest.skipUnless(os.path.isdir("/proc/self/task"), "needs /proc")
    def test_pool_is_reused_between_calls(self):
        index, _, queries = build()
        index.search(queries[:1], K, EF, threads=1)

        def native_threads():
            return len(os.listdir("/proc/self/task"))

        before = native_threads()
        index.search(queries, K, EF, threads=4)
        during = native_threads()
        self.assertEqual(during, before + 3)  # the caller is pool thread 0
        index.search(queries, K, EF, threads=4)
        self.assertEqual(native_threads(), during)  # kept, not recreated per call


class SaveLoad(TmpDirCase):
    def test_modes_and_validations_are_bit_identical(self):
        index, _, queries = build()
        ref = index.search(queries, K, EF)
        index.save(self.path())
        for mode in ("mmap", "read"):
            for validate in ("deep", "shallow"):
                with hnsw_cpp.load(self.path(), mode=mode, validate=validate) as loaded:
                    self.assertEqual((loaded.size, loaded.dim), (N, DIM))
                    for threads in (1, 2, 0):
                        got = loaded.search(queries, K, EF, threads=threads)
                        self.assertTrue(same(ref, got), (mode, validate, threads))

    def test_defaults_and_pathlike(self):
        import pathlib

        index, _, queries = build()
        index.save(pathlib.Path(self.path()))
        loaded = hnsw_cpp.load(pathlib.Path(self.path()))
        self.assertTrue(same(index.search(queries, K, EF), loaded.search(queries, K, EF)))

    def test_empty_index_roundtrip(self):
        hnsw_cpp.Index(5).save(self.path())
        loaded = hnsw_cpp.load(self.path())
        self.assertEqual((loaded.size, loaded.dim), (0, 5))
        ids, _ = loaded.search(np.zeros(5, dtype=np.float32), 2, 5)
        self.assertEqual(ids.tolist(), [[-1, -1]])

    def test_loaded_outlives_source_index_and_file(self):
        index, _, queries = build()
        ref = index.search(queries, K, EF)
        index.save(self.path())
        loaded = hnsw_cpp.load(self.path())
        del index
        gc.collect()
        os.unlink(self.path())  # the mapping keeps the contents
        self.assertTrue(same(ref, loaded.search(queries, K, EF)))

    def test_close(self):
        index, _, queries = build()
        index.save(self.path())
        loaded = hnsw_cpp.load(self.path())
        self.assertFalse(loaded.closed)
        loaded.close()
        loaded.close()  # idempotent
        self.assertTrue(loaded.closed)
        for call in (lambda: loaded.search(queries, K, EF), lambda: loaded.size, lambda: loaded.dim):
            with self.assertRaisesRegex(ValueError, "closed"):
                call()

    def test_save_overwrite_does_not_disturb_loaded(self):
        a, _, queries = build(0)
        a.save(self.path())
        loaded = hnsw_cpp.load(self.path())
        ref = loaded.search(queries, K, EF)
        b, _, _ = build(1)
        b.save(self.path())
        self.assertTrue(same(ref, loaded.search(queries, K, EF)))

    def test_save_to_missing_directory_is_oserror(self):
        index = hnsw_cpp.Index(3)
        with self.assertRaises(OSError):
            index.save(os.path.join(self.dir, "no", "such", "dir", "i.hnsw"))

    def test_bad_load_arguments(self):
        hnsw_cpp.Index(3).save(self.path())
        with self.assertRaisesRegex(ValueError, "mode must be 'mmap' or 'read'"):
            hnsw_cpp.load(self.path(), mode="bogus")
        with self.assertRaisesRegex(ValueError, "validate must be 'deep' or 'shallow'"):
            hnsw_cpp.load(self.path(), validate="maybe")
        with self.assertRaises(TypeError):
            hnsw_cpp.load(12345)


class CorruptFiles(TmpDirCase):
    def saved_bytes(self):
        index = hnsw_cpp.Index(8, M=4)
        index.add(np.random.default_rng(1).random((50, 8), dtype=np.float32))
        index.save(self.path("good"))
        with open(self.path("good"), "rb") as f:
            return f.read()

    def write(self, data):
        with open(self.path("bad"), "wb") as f:
            f.write(data)
        return self.path("bad")

    def test_missing_file_is_oserror(self):
        with self.assertRaises(FileNotFoundError):
            hnsw_cpp.load(self.path("nope"))

    def test_truncated_and_garbage_are_valueerror(self):
        data = self.saved_bytes()
        cases = {
            "empty": b"",
            "short header": data[:100],
            "truncated body": data[: len(data) // 2],
            "one byte short": data[:-1],
            "garbage": os.urandom(len(data)),
            "bad magic": b"X" + data[1:],
        }
        for name, content in cases.items():
            for mode in ("mmap", "read"):
                for validate in ("deep", "shallow"):
                    with self.subTest(name=name, mode=mode, validate=validate):
                        with self.assertRaises(ValueError) as cm:
                            hnsw_cpp.load(self.write(content), mode=mode, validate=validate)
                        self.assertIn("bad", str(cm.exception))  # message names the file

    def test_header_bitflip_is_valueerror(self):
        data = bytearray(self.saved_bytes())
        data[20] ^= 0xFF  # dim field; the header CRC no longer matches
        with self.assertRaisesRegex(ValueError, "CRC"):
            hnsw_cpp.load(self.write(bytes(data)))

    def test_nan_vector_is_rejected_by_deep_validation_only(self):
        data = bytearray(self.saved_bytes())
        arena = struct.unpack_from("<Q", data, 64 + 16)[0]  # section 1 offset
        stride = struct.unpack_from("<Q", data, 48)[0]
        vector_offset = (4 + 2 * 4 * 4 + 31) // 32 * 32  # M = 4, see layout.h
        self.assertGreater(stride, vector_offset)
        struct.pack_into("<f", data, arena + vector_offset, float("nan"))
        path = self.write(bytes(data))
        with self.assertRaisesRegex(ValueError, "not finite"):
            hnsw_cpp.load(path, validate="deep")
        hnsw_cpp.load(path, validate="shallow").close()  # trusted-file mode skips it


class Validation(unittest.TestCase):
    def setUp(self):
        self.index = hnsw_cpp.Index(4)
        self.index.add(np.zeros((3, 4), dtype=np.float32))
        self.good = np.zeros((2, 4), dtype=np.float32)

    def assertRejected(self, exc, pattern, fn):
        with self.assertRaisesRegex(exc, pattern):
            fn()
        self.assertEqual(self.index.size, 3)  # a rejected add changes nothing

    def test_wrong_type_and_dtype(self):
        fix = "np.ascontiguousarray"
        for fn in (self.index.add, lambda x: self.index.search(x, 1, 1)):
            self.assertRejected(TypeError, "numpy.ndarray.*got list.*" + fix, lambda: fn([[0.0] * 4]))
            self.assertRejected(TypeError, "float32, got float64.*" + fix,
                                lambda: fn(self.good.astype(np.float64)))
            self.assertRejected(TypeError, "float32, got int64", lambda: fn(self.good.astype(np.int64)))
            self.assertRejected(TypeError, "float32", lambda: fn(self.good.astype(">f4")))
            self.assertRejected(TypeError, "got NoneType", lambda: fn(None))

    def test_contiguity_and_alignment(self):
        wide = np.zeros((2, 8), dtype=np.float32)
        for fn in (self.index.add, lambda x: self.index.search(x, 1, 1)):
            self.assertRejected(ValueError, "C-contiguous.*np.ascontiguousarray", lambda: fn(wide[:, ::2]))
            self.assertRejected(ValueError, "C-contiguous", lambda: fn(np.asfortranarray(self.good)))
            unaligned = np.zeros(4 * 4 + 1, dtype=np.uint8)[1:].view(np.float32)
            self.assertFalse(unaligned.flags.aligned)
            self.assertRejected(ValueError, "aligned", lambda: fn(unaligned))

    def test_shape_and_dim(self):
        for fn in (self.index.add, lambda x: self.index.search(x, 1, 1)):
            self.assertRejected(ValueError, "length 3 but the index has dim 4",
                                lambda: fn(np.zeros((2, 3), dtype=np.float32)))
            self.assertRejected(ValueError, "length 5 but the index has dim 4",
                                lambda: fn(np.zeros(5, dtype=np.float32)))
            self.assertRejected(ValueError, "length 0 but the index has dim 4",
                                lambda: fn(np.zeros((0, 0), dtype=np.float32)))
            self.assertRejected(ValueError, "3-D", lambda: fn(np.zeros((1, 2, 4), dtype=np.float32)))
            self.assertRejected(ValueError, "0-D", lambda: fn(np.zeros((), dtype=np.float32)))

    def test_non_finite_vectors(self):
        for bad in (np.nan, np.inf, -np.inf):
            x = self.good.copy()
            x[1, 2] = bad
            self.assertRejected(ValueError, r"vectors\[1\] element 2 is NaN or infinite.*nothing was added",
                                lambda: self.index.add(x))
        # Queries are not restricted to finite values by the bindings' contract; only add is.

    def test_bad_search_arguments(self):
        q = self.good
        self.assertRejected(ValueError, "k must be >= 1", lambda: self.index.search(q, 0, 1))
        self.assertRejected(ValueError, "k must be >= 1", lambda: self.index.search(q, -1, 1))
        self.assertRejected(ValueError, "ef must be >= 1", lambda: self.index.search(q, 1, 0))
        self.assertRejected(ValueError, "threads must be >= 0", lambda: self.index.search(q, 1, 1, threads=-1))
        self.assertRejected(ValueError, "at most 1024", lambda: self.index.search(q, 1, 1, threads=10**6))
        self.assertRejected(TypeError, "incompatible", lambda: self.index.search(q, 1.5, 1))

    def test_bad_constructor_arguments(self):
        for kwargs, pattern in (
            ({"dim": 0}, "dim must be"),
            ({"dim": -1}, "dim must be"),
            ({"dim": 65537}, "dim must be"),
            ({"dim": 4, "M": 1}, "M must be"),
            ({"dim": 4, "M": 1025}, "M must be"),
            ({"dim": 4, "ef_construction": 0}, "ef_construction must be"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, pattern):
                    hnsw_cpp.Index(**kwargs)

    def test_inputs_are_not_modified(self):
        x = np.random.default_rng(3).random((20, 4), dtype=np.float32)
        before = x.copy()
        index = hnsw_cpp.Index(4)
        index.add(x)
        index.search(x, 3, 10, threads=2)
        self.assertTrue(np.array_equal(x, before))

    def test_loaded_index_cannot_be_constructed(self):
        with self.assertRaises(TypeError):
            hnsw_cpp.LoadedIndex()


class FailedAdd(TmpDirCase):
    def test_failed_add_leaves_index_identical_and_usable(self):
        base, queries = dataset(5, n=300)
        ref = hnsw_cpp.Index(DIM, seed=5)
        ref.add(base[:150])
        ref.add(base[150:])
        ref.save(self.path("ref"))

        index = hnsw_cpp.Index(DIM, seed=5)
        index.add(base[:150])
        bad = base[150:].copy()
        bad[100, 7] = np.nan  # a failure in the middle of a batch adds none of it
        with self.assertRaises(ValueError):
            index.add(bad)
        with self.assertRaises(ValueError):
            index.add(np.full(DIM, np.inf, dtype=np.float32))
        with self.assertRaises(TypeError):
            index.add(base[150:].astype(np.float64))
        self.assertEqual(index.size, 150)
        self.assertEqual(index.add(base[150:]).tolist(), list(range(150, 300)))
        index.save(self.path("got"))
        with open(self.path("ref"), "rb") as a, open(self.path("got"), "rb") as b:
            self.assertEqual(a.read(), b.read())
        self.assertTrue(same(ref.search(queries, K, EF), index.search(queries, K, EF)))


class Concurrency(unittest.TestCase):
    def run_threads(self, fns):
        errors = []

        def wrap(fn):
            def run():
                try:
                    fn()
                except BaseException as e:  # reported from the main thread
                    errors.append(e)

            return run

        threads = [threading.Thread(target=wrap(f)) for f in fns]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
            self.assertFalse(t.is_alive(), "thread did not finish (deadlock?)")
        if errors:
            raise errors[0]

    def test_concurrent_searches_equal_sequential(self):
        index, _, queries = build()
        ref = index.search(queries, K, EF)
        results = {}

        def worker(i, threads):
            def run():
                out = []
                for _ in range(10):
                    out.append(index.search(queries, K, EF, threads=threads))
                results[i] = out

            return run

        self.run_threads([worker(i, 1 if i < 6 else 2) for i in range(8)])
        for i, out in results.items():
            for got in out:
                self.assertTrue(same(ref, got), i)

    def test_concurrent_searches_on_loaded_index(self):
        index, _, queries = build()
        ref = index.search(queries, K, EF)
        with tempfile.TemporaryDirectory() as d:
            index.save(os.path.join(d, "i"))
            loaded = hnsw_cpp.load(os.path.join(d, "i"))
            outs = []
            self.run_threads(
                [lambda: outs.append(loaded.search(queries, K, EF)) for _ in range(6)]
            )
        self.assertEqual(len(outs), 6)
        self.assertTrue(all(same(ref, o) for o in outs))

    def test_searches_while_adding(self):
        # Many small adds grow (and reallocate) the index's storage while
        # several threads search it. Without the add/search lock this reads
        # freed memory; a crash or an inconsistent result fails the test.
        total = 12000
        base, queries = dataset(2, n=total, nq=20)
        index = hnsw_cpp.Index(DIM, M=8, ef_construction=40, seed=2)
        index.add(base[:100])
        done = threading.Event()
        searches = [0]

        def adder():
            try:
                for start in range(100, total, 25):
                    index.add(base[start : start + 25])
            finally:
                done.set()

        def searcher():
            while not done.is_set():
                ids, dists = index.search(queries, 5, 30)
                self.assertTrue(np.all(ids < total) and np.all(ids >= -1))
                valid = ids >= 0
                self.assertTrue(np.all(valid[:, 0]))
                # Each result is a real vector and its distance is the true
                # squared distance (the vector with id i is base[i]).
                rows = np.nonzero(valid)
                true = ((queries[rows[0]] - base[ids[valid]]) ** 2).sum(axis=1)
                np.testing.assert_allclose(dists[valid], true, rtol=1e-3, atol=1e-5)
                searches[0] += 1

        self.run_threads([adder] + [searcher] * 4)
        self.assertEqual(index.size, total)
        self.assertGreater(searches[0], 0)

    def test_concurrent_adds_are_exclusive(self):
        total, parts = 8000, 3
        base, _ = dataset(4, n=total, nq=1)
        index = hnsw_cpp.Index(DIM, M=8, ef_construction=40, seed=4)
        got = [[] for _ in range(parts)]

        def adder(part):
            def run():
                for start in range(part * 20, total, 20 * parts):
                    got[part].append((start, index.add(base[start : start + 20])))

            return run

        stop = threading.Event()
        queries = base[:10]

        def searcher():
            while not stop.is_set():
                ids, _ = index.search(queries, 3, 20)
                self.assertTrue(np.all(ids < total))

        s = threading.Thread(target=searcher)
        s.start()
        try:
            self.run_threads([adder(i) for i in range(parts)])
        finally:
            stop.set()
            s.join()
        self.assertEqual(index.size, total)
        all_ids = np.concatenate([ids for part in got for _, ids in part])
        self.assertEqual(np.sort(all_ids).tolist(), list(range(total)))  # unique, no gaps
        # Every id holds exactly the vector that was added under it: returned
        # distances equal the true squared distances to base[row of that id].
        id_to_row = np.empty(total, dtype=np.int64)
        for part in got:
            for start, ids in part:
                id_to_row[ids] = np.arange(start, start + len(ids))
        sample = np.arange(0, total, 37)
        q = np.ascontiguousarray(base[id_to_row[sample]])
        ids, dists = index.search(q, 5, 200)
        true = ((q[:, None, :] - base[id_to_row[ids]]) ** 2).sum(axis=2)
        np.testing.assert_allclose(dists, true, rtol=1e-3, atol=1e-5)
        # HNSW is approximate, but nearly every stored vector finds itself.
        self.assertGreaterEqual(np.mean(dists[:, 0] == 0.0), 0.9)


if __name__ == "__main__":
    unittest.main()
