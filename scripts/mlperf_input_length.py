"""
The mean prompt length of the MLPerf Inference Llama 2 70B dataset, which
the MLPerf ceiling in accel_model/core.py needs (the benchmark's README
states only the output length, 294.45 tokens per sample).

Not run in CI: it needs the 31.7 MB dataset from MLCommons' public bucket,

    curl -sLO https://inference.mlcommons-storage.org/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz
    md5sum open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz   # c247a315b9328365f66d2c2585b35cd4
    gunzip -k open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz
    python3 -I scripts/mlperf_input_length.py open_orca_gpt4_tokenized_llama.sampled_24576.pkl

(the MD5 is the one the bucket's manifest lists:
https://inference.mlcommons-storage.org/metadata/llama-2-70b-open-orca-dataset.md5),
and pandas. The file is a pickled pandas DataFrame; a pickle can run code,
so this loads it with an Unpickler that allows only the pandas and numpy
classes a DataFrame needs and refuses anything else.

Run on 2026-10-16: 24,576 samples, tok_input_length total 5,463,976, mean
222.33, median 152, min 48, max 1,021 (tok_output_length, the dataset's
reference answers: mean 218.89).
"""
from __future__ import annotations

import pickle
import statistics
import sys

ALLOW = {
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.base", "Index"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("builtins", "slice"),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in ALLOW:
            raise pickle.UnpicklingError(f"refused {module}.{name}")
        import numpy

        if module.startswith("numpy.core") and numpy.__version__ >= "2":
            module = module.replace("numpy.core", "numpy._core", 1)
        return super().find_class(module, name)


def main(path: str) -> None:
    with open(path, "rb") as f:
        df = SafeUnpickler(f).load()
    for col in ("tok_input_length", "tok_output_length"):
        v = [int(x) for x in df[col]]
        print(f"{col}: n {len(v)}, total {sum(v)}, mean {statistics.fmean(v):.4f}, "
              f"median {statistics.median(v)}, min {min(v)}, max {max(v)}")


if __name__ == "__main__":
    main(sys.argv[1])
