"""
The sourced data: every hardware figure is a field with a value, a status and
a source, and every source carries the date it was read.

Statuses:
  spec          the vendor states it (product page, datasheet, documentation, paper)
  derived       computed from other sourced figures; the formula is in `ref`
  reported      reported by a third party (named in the source), not the vendor
  illustrative  a model parameter chosen for the picture, labelled as such
  undisclosed   not disclosed in the sources used (value None)

Availability of a part (`avail`):
  product       shipping or generally available (a product page or cloud documentation lists it)
  announced     announced by the vendor, shipping not yet stated or only just starting
  retired       superseded; kept because published results used it
"""
from __future__ import annotations

from typing import Any

ACCESSED = "2026-10-15"


def _s(title: str, url: str, kind: str, accessed: str = ACCESSED) -> dict[str, str]:
    return {"title": title, "url": url, "kind": kind, "accessed": accessed}


ACCESSED_42 = "2026-10-16"  # sources added by brief 42


SOURCES: dict[str, dict[str, str]] = {
    # GPUs
    "h100-page": _s(
        "NVIDIA H100 product page, Product Specifications (H100 SXM column)",
        "https://www.nvidia.com/en-us/data-center/h100/",
        "vendor product page",
    ),
    "hopper-blog": _s(
        "NVIDIA Hopper Architecture In-Depth (NVIDIA Technical Blog)",
        "https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/",
        "vendor technical blog",
    ),
    "gb200-page": _s(
        "NVIDIA GB200 NVL72 product page, Specifications, GB200 Superchip column ('Specification in sparse. Dense is one-half sparse spec shown')",
        "https://www.nvidia.com/en-us/data-center/gb200-nvl72/",
        "vendor product page",
    ),
    "bwu-blog": _s(
        "Inside NVIDIA Blackwell Ultra: The Chip Powering the AI Factory Era (NVIDIA Technical Blog), Table 2",
        "https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/",
        "vendor technical blog",
    ),
    "hgx-page": _s(
        "NVIDIA HGX platform page, HGX B200 specifications ('Specification in Sparse. Dense is 1/2 sparse spec shown')",
        "https://www.nvidia.com/en-us/data-center/hgx/",
        "vendor product page",
    ),
    # systolic designs
    "tpu-arch": _s(
        "Google Cloud TPU documentation: TPU architecture (TensorCores and MXUs)",
        "https://cloud.google.com/tpu/docs/system-architecture-tpu-vm",
        "vendor documentation",
    ),
    "tpu-v6e": _s(
        "Google Cloud TPU documentation: TPU v6e (Trillium), system architecture table",
        "https://cloud.google.com/tpu/docs/v6e",
        "vendor documentation",
    ),
    "tpu-7x": _s(
        "Google Cloud TPU documentation: TPU7x (Ironwood), system architecture table and memory hierarchy",
        "https://cloud.google.com/tpu/docs/tpu7x",
        "vendor documentation",
    ),
    "trn2-arch": _s(
        "AWS Neuron documentation: Trainium2 architecture",
        "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html",
        "vendor documentation",
    ),
    "nc3-arch": _s(
        "AWS Neuron documentation: NeuronCore-v3 architecture (on-chip SRAM, Tensor Engine)",
        "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html",
        "vendor documentation",
    ),
    "nki-trn2": _s(
        "AWS Neuron documentation: NKI, Trainium2 architecture guide (TensorE PE grid)",
        "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html",
        "vendor documentation",
    ),
    # wafer-scale
    "cs3-datasheet": _s(
        "Cerebras, 'The World's Fastest AI Infrastructure': CS-3 and WSE-3 datasheet (PDF; 'FLOPS shown as sparse')",
        "https://cdn.sanity.io/files/e4qjo92p/production/864133620041e5af418074842e42a669c137fa23.pdf",
        "vendor datasheet",
    ),
    "cerebras-wp": _s(
        "Cerebras white paper, 'The Cerebras Wafer-Scale Architecture for Deep Learning' (PDF)",
        "https://cdn.sanity.io/files/e4qjo92p/production/2d7fa58e3b820715664bcf42097e86c05070c161.pdf",
        "vendor white paper",
    ),
    "cs4-datasheet": _s(
        "Cerebras, 'Meet CS-4' datasheet (PDF; 'AI compute shown as sparse FP16')",
        "https://cdn.sanity.io/files/e4qjo92p/production/bb028ae2422089dfbcc9fd2455c021f3625c7a75.pdf",
        "vendor datasheet",
    ),
    "cs4-page": _s(
        "Cerebras CS-4 system page ('First CS-4 shipments begin this quarter')",
        "https://www.cerebras.ai/system",
        "vendor product page",
    ),
    "cerebras-chip": _s(
        "Cerebras WSE-3 Turbo chip page ('How Cerebras Solved the Yield Problem')",
        "https://www.cerebras.ai/chip",
        "vendor product page",
    ),
    "cerebras-inference": _s(
        "Cerebras blog, 'Introducing Cerebras Inference: AI at Instant Speed' (August 2024)",
        "https://www.cerebras.ai/blog/introducing-cerebras-inference-ai-at-instant-speed",
        "vendor blog",
    ),
    # compiler-scheduled
    "groqchip-brief": _s(
        "Groq, GroqChip Processor Product Brief v1.5 (2022; Internet Archive copy of groq.com)",
        "https://web.archive.org/web/20250615145238/https://groq.com/wp-content/uploads/2022/10/GroqChip%E2%84%A2-Processor-Product-Brief-v1.5.pdf",
        "vendor datasheet",
    ),
    "groq-isca22": _s(
        "Abts et al., 'A Software-defined Tensor Streaming Multiprocessor for Large-scale Machine Learning', ISCA 2022 (Groq's copy, Internet Archive)",
        "https://web.archive.org/web/20250612160012/https://groq.com/wp-content/uploads/2024/02/GroqISCAPaper2022_ASoftwareDefinedTensorStreamingMultiprocessorForLargeScaleMachineLearning-1.pdf",
        "paper (ISCA 2022)",
    ),
    "lpx-page": _s(
        "NVIDIA Groq 3 LPX product page (LPU accelerator and rack)",
        "https://www.nvidia.com/en-us/data-center/lpx/",
        "vendor product page",
    ),
    "groq-home": _s(
        "Groq home page, LPX rack figures ('When released, each NVIDIA Groq 3 LPX rack connects 256 next-generation LPU accelerators')",
        "https://groq.com/",
        "vendor product page",
    ),
    "nextplatform-groq": _s(
        "The Next Platform, 'Groq Says It Can Deploy 1 Million AI Inference Chips In Two Years' (2023-11-27): Groq's Llama 2 70B demonstration",
        "https://www.nextplatform.com/2023/11/27/groq-says-it-can-deploy-1-million-ai-inference-chips-in-two-years/",
        "trade press (reports a vendor demonstration)",
    ),
    # benchmarks and models
    "mlperf-h100": _s(
        "MLCommons, MLPerf Inference v4.0 results: NVIDIA DGX H100 (8x H100-SXM-80GB, TensorRT), llama2-70b-99.9, Offline, mlperf_log_summary.txt",
        "https://raw.githubusercontent.com/mlcommons/inference_results_v4.0/main/closed/NVIDIA/results/DGX-H100_H100-SXM-80GBx8_TRT/llama2-70b-99.9/Offline/performance/run_1/mlperf_log_summary.txt",
        "benchmark result log",
    ),
    "mlperf-llama2": _s(
        "MLCommons inference reference implementation, Llama 2 70B README ('Tokens per sample: 294.45')",
        "https://github.com/mlcommons/inference/blob/master/language/llama2-70b/README.md",
        "benchmark documentation",
    ),
    "nv-mlperf-v40": _s(
        "NVIDIA Technical Blog, 'NVIDIA H200 Tensor Core GPUs and NVIDIA TensorRT-LLM Set MLPerf LLM Inference Records' (MLPerf Inference v4.0; FP8 quantization)",
        "https://developer.nvidia.com/blog/nvidia-h200-tensor-core-gpus-and-nvidia-tensorrt-llm-set-mlperf-llm-inference-records/",
        "vendor technical blog",
    ),
    "llama3": _s(
        "Grattafiori et al., 'The Llama 3 Herd of Models', arXiv:2407.21783 (Table 3)",
        "https://arxiv.org/abs/2407.21783",
        "paper",
    ),
    "llama2": _s(
        "Touvron et al., 'Llama 2: Open Foundation and Fine-Tuned Chat Models', arXiv:2307.09288",
        "https://arxiv.org/abs/2307.09288",
        "paper",
    ),
    # the other families (chapter 1's map; chapters 6-10 add their figures)
    "sze-survey": _s(
        "Sze, Chen, Yang and Emer, 'Efficient Processing of Deep Neural Networks: A Tutorial and Survey', arXiv:1703.09039",
        "https://arxiv.org/abs/1703.09039",
        "paper",
    ),
    "sn40l": _s(
        "Prabhakar et al., 'SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts', arXiv:2405.07518",
        "https://arxiv.org/abs/2405.07518",
        "paper",
    ),
    "sambanova-rdu": _s(
        "SambaNova, RDU chip page (SN50: dataflow execution, tiered memory)",
        "https://sambanova.ai/technology/sn40l-rdu-ai-chip",
        "vendor product page",
    ),
    "ipu-dissect": _s(
        "Jia, Tillman, Maggioni and Scarpazza, 'Dissecting the Graphcore IPU Architecture via Microbenchmarking', arXiv:1912.03413",
        "https://arxiv.org/abs/1912.03413",
        "paper",
    ),
    "tt-hardware": _s(
        "Tenstorrent hardware page (Blackhole: Tensix cores, 'big RISC-V' cores, GDDR6, QSFP-DD 800G ports)",
        "https://tenstorrent.com/hardware/blackhole",
        "vendor product page",
    ),
    "amd-aie": _s(
        "AMD AI Engine technology page (Versal adaptive SoCs: 2-D array of VLIW SIMD tiles with local memory)",
        "https://www.amd.com/en/products/adaptive-socs-and-fpgas/technologies/ai-engine.html",
        "vendor product page",
    ),
    "apple-m4": _s(
        "Apple Newsroom, 'Apple introduces M4 chip' (2024-05-07; Neural Engine 38 trillion operations per second)",
        "https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/",
        "vendor press release",
    ),
    "pcm-imc": _s(
        "Le Gallo et al., 'A 64-core mixed-signal in-memory compute chip based on phase-change memory for deep neural network inference', arXiv:2212.02872",
        "https://arxiv.org/abs/2212.02872",
        "paper",
    ),
    # chapters 6-10
    "sambanova-sn50-ds": _s(
        "SambaNova SN50 RDU datasheet (PDF; three-tier memory, 'DDR5 Up to 2 TB')",
        "https://sambanova.ai/hubfs/SambaNova%20SN50%20RDU.pdf",
        "vendor datasheet",
    ),
    "graphcore-home": _s(
        "Graphcore home page (lists no IPU products; graphcore.ai/products returns 404 on the access date)",
        "https://www.graphcore.ai/",
        "vendor home page",
    ),
    "apple-mbp-m4": _s(
        "Apple Newsroom, 'New MacBook Pro features M4 family of chips and Apple Intelligence' (2024-10-30; M4: up to 32GB unified memory, 120GB/s)",
        "https://www.apple.com/newsroom/2024/10/new-macbook-pro-features-m4-family-of-chips-and-apple-intelligence/",
        "vendor press release",
    ),
    "amd-v70": _s(
        "AMD Alveo V70 accelerator card page, Specifications (XDNA, Versal AI Core, AIE-ML tiles)",
        "https://www.amd.com/en/products/accelerators/alveo/v70.html",
        "vendor product page",
    ),
    "amd-v70-part": _s(
        "AMD Alveo V70 part page a-v70-p16g-pq-g ('Discontinued: This product has been discontinued and is no longer offered for sale')",
        "https://www.amd.com/en/products/accelerators/alveo/v70/a-v70-p16g-pq-g.html",
        "vendor product page",
    ),
    "hailo-10h": _s(
        "Hailo-10H M.2 AI acceleration module page (40|20 TOPS INT4|INT8, 2.5W typical, 4|8GB LPDDR4/4X)",
        "https://hailo.ai/products/ai-accelerators/hailo-10h-m-2-ai-acceleration-module/",
        "vendor product page",
    ),
    # added by brief 42 (accessed 2026-10-16)
    "amd-mi355x": _s(
        "AMD Instinct MI355X product page, Specifications (launch date 6/12/2025)",
        "https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html",
        "vendor product page", ACCESSED_42,
    ),
    "amd-v80": _s(
        "AMD Alveo V80 compute accelerator card, part page A-V80-P64G-PQ-G, General Specifications (Versal HBM XCV80; 'Buy')",
        "https://www.amd.com/en/products/accelerators/alveo/v80/a-v80-p64g-pq-g.html",
        "vendor product page", ACCESSED_42,
    ),
    "mlperf-gb200": _s(
        "MLCommons, MLPerf Inference v5.1 results: NVIDIA GB200 NVL72 (4x GB200-186GB, one tray, TensorRT), llama2-70b-99.9, Offline, mlperf_log_summary.txt (weights FP4 per the submission's measurements JSON)",
        "https://raw.githubusercontent.com/mlcommons/inference_results_v5.1/main/closed/NVIDIA/results/GB200-NVL72_GB200-186GB_aarch64x4_TRT/llama2-70b-99.9/Offline/performance/run_1/mlperf_log_summary.txt",
        "benchmark result log", ACCESSED_42,
    ),
    "mlperf-gb200-meas": _s(
        "MLCommons, MLPerf Inference v5.1: the same submission's measurements JSON ('weight_data_types': 'fp4')",
        "https://raw.githubusercontent.com/mlcommons/inference_results_v5.1/main/closed/NVIDIA/measurements/GB200-NVL72_GB200-186GB_aarch64x4_TRT/llama2-70b-99.9/Offline/GB200-NVL72_GB200-186GB_aarch64x4_TRT_Offline.json",
        "benchmark submission metadata", ACCESSED_42,
    ),
    "mlperf-dataset": _s(
        "MLCommons, the MLPerf Llama 2 70B dataset (the 24,576-sample file and its MD5, as the bucket's manifest lists them): prompt lengths averaged by the site's scripts/mlperf_input_length.py",
        "https://inference.mlcommons-storage.org/metadata/llama-2-70b-open-orca-dataset.md5",
        "benchmark dataset", ACCESSED_42,
    ),
    "flightllm": _s(
        "Zeng et al., 'FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs', FPGA 2024, arXiv:2401.03868",
        "https://arxiv.org/abs/2401.03868",
        "peer-reviewed paper", ACCESSED_42,
    ),
    "chen-spatial": _s(
        "Chen et al., 'Understanding the Potential of FPGA-Based Spatial Acceleration for Large Language Model Inference', arXiv:2312.15159",
        "https://arxiv.org/abs/2312.15159",
        "paper", ACCESSED_42,
    ),
    "finn": _s(
        "Umuroglu et al., 'FINN: A Framework for Fast, Scalable Binarized Neural Network Inference', FPGA 2017, arXiv:1612.07119",
        "https://arxiv.org/abs/1612.07119",
        "peer-reviewed paper", ACCESSED_42,
    ),
    "hls4ml": _s(
        "Duarte et al., 'Fast inference of deep neural networks in FPGAs for particle physics' (hls4ml), JINST 2018, arXiv:1804.06913",
        "https://arxiv.org/abs/1804.06913",
        "peer-reviewed paper", ACCESSED_42,
    ),
}


def _f(v: Any, st: str, src: str, ref: str) -> dict[str, Any]:
    """One field: value, status, source id, where in the source."""
    return {"v": v, "st": st, "src": src, "ref": ref}


GiB = 1024**3

# ---------------------------------------------------------------------------
# The families (chapter 1's map). Positions on the axes are categories, each
# with a one-line basis and a source. Order = the order the map steps through.
# ---------------------------------------------------------------------------

# Memory placement of the weights during inference, from far to near the maths:
#   0 = off-chip DRAM (HBM, GDDR, LPDDR) behind on-chip buffers
#   1 = tiered: large on-chip SRAM plus off-chip memory, both used
#   2 = on-chip SRAM only (the model is split over as many chips as it takes)
#   3 = in the memory array itself (compute where the weight is stored)
# Scheduling (who decides when data moves):
#   0 = hardware at run time (warps, caches)
#   1 = mixed (compiler-planned, hardware-triggered dataflow or per-core control)
#   2 = the compiler, ahead of time (a static schedule)
# Granularity: few big engines (0) ... many small cores (2).
PLACEMENT_LABELS = ("off-chip DRAM", "tiered SRAM + DRAM", "on-chip SRAM", "in the memory array")
SCHEDULE_LABELS = ("hardware, at run time", "mixed", "compiler, ahead of time")
GRAIN_LABELS = ("a few large engines", "tens to hundreds of cores", "hundreds of thousands of cores")

FAMILIES: list[dict[str, Any]] = [
    {
        "id": "gpu", "name": "GPUs", "chapter": 2, "example": "NVIDIA H100, B200", "also": ["hgx-page"],
        "placement": _f(0, "spec", "hopper-blog", "weights in HBM; a 50 MB L2 cache 'reduces trips to HBM3'"),
        "schedule": _f(0, "spec", "hopper-blog", "warps scheduled by the SMs at run time; L2 is a hardware-managed cache"),
        "grain": _f(1, "spec", "hopper-blog", "132 SMs per H100 SXM5, each with tensor cores"),
    },
    {
        "id": "systolic", "name": "Systolic designs", "chapter": 3, "example": "Google TPU, AWS Trainium",
        "placement": _f(0, "spec", "tpu-7x", "weights in HBM; VMEM is 'a high-speed scratchpad' the compiler and kernels manage"),
        "schedule": _f(2, "spec", "tpu-arch", "programs reach the TensorCores through the XLA compiler (the documentation's XLA compiler and flags); VMEM is managed by the compiler and custom kernels"),
        "grain": _f(0, "spec", "tpu-arch", "a few 256 x 256 or 128 x 128 MXUs per TensorCore"),
    },
    {
        "id": "wafer", "name": "Wafer-scale", "chapter": 4, "example": "Cerebras WSE-3",
        "placement": _f(2, "spec", "cerebras-inference", "'we split them at layer boundaries and map them to multiple CS-3 systems'; 44 GB of on-chip SRAM per wafer"),
        "schedule": _f(1, "spec", "cerebras-wp", "static routes on the fabric; computation 'triggered by the data' in each core"),
        "grain": _f(2, "spec", "cerebras-wp", "900,000 cores on one wafer"),
    },
    {
        "id": "lpu", "name": "Compiler-scheduled", "chapter": 5, "example": "Groq LPU",
        "placement": _f(2, "spec", "groqchip-brief", "230 MB SRAM per chip, 'without the need for external memory'"),
        "schedule": _f(2, "spec", "groq-isca22", "'statically scheduled, deterministic execution'; even chip-to-chip transfers are scheduled"),
        "grain": _f(0, "spec", "groq-isca22", "'instruction execution [partitioned] into functional slices': a few wide slices on one chip"),
    },
    {
        "id": "dataflow", "name": "Reconfigurable dataflow", "chapter": 6, "example": "SambaNova RDU, Graphcore IPU", "also": ["sambanova-rdu"],
        "placement": _f(1, "spec", "sn40l", "SN40L: 'three-tier memory system with on-chip distributed SRAM, on-package HBM, and off-package DDR DRAM'"),
        "schedule": _f(2, "spec", "sn40l", "dataflow: operations fused into one pipeline on the chip instead of a kernel per operation (the paper's 'fused operations')"),
        "grain": _f(1, "spec", "ipu-dissect", "IPU (GC2): '1,216 tiles', each a core with its own memory"),
    },
    {
        "id": "mesh", "name": "Meshes of small cores", "chapter": 7, "example": "Tenstorrent Blackhole",
        "placement": _f(1, "spec", "tt-hardware", "Tensix cores with local SRAM, plus GDDR6 per chip"),
        "schedule": _f(1, "spec", "tt-hardware", "small RISC-V cores in every Tensix core run the data movement"),
        "grain": _f(1, "spec", "tt-hardware", "120 Tensix cores per Blackhole chip"),
    },
    {
        "id": "fpga-npu", "name": "FPGAs and edge NPUs", "chapter": 8, "example": "AMD Versal AI Engines, Apple Neural Engine", "also": ["apple-m4"],
        "placement": _f(1, "spec", "amd-aie", "each AI Engine tile has local data memory; weights also stream from DRAM"),
        "schedule": _f(2, "spec", "amd-aie", "'DMA engines for scheduled data movement'"),
        "grain": _f(1, "spec", "amd-aie", "'10s to 100s of AI Engines in a single device'"),
    },
    {
        "id": "imc", "name": "In-memory and analogue", "chapter": 9, "example": "phase-change crossbars (research)", "also": ["sze-survey"],
        "placement": _f(3, "spec", "pcm-imc", "weights stored as device conductances; the multiply happens in the array"),
        "schedule": _f(2, "spec", "pcm-imc", "weights programmed into the devices' conductances before inference; the network is mapped onto the cores"),
        "grain": _f(1, "spec", "pcm-imc", "64 in-memory compute cores on one chip"),
    },
]

# ---------------------------------------------------------------------------
# Parts: one or two representative chips per family (chapters 2-5).
# Bandwidths are per direction unless the field name says bidir. Peak rates
# are dense unless the field name says sparse.
# ---------------------------------------------------------------------------

PARTS: dict[str, dict[str, Any]] = {
    "h100": {
        "id": "h100", "name": "NVIDIA H100 SXM", "family": "gpu", "layout": "tensor",
        "avail": _f("product", "spec", "h100-page", "product page with specifications"),
        "fields": {
            "peak_bf16": _f(989.5e12, "derived", "h100-page", "BF16 Tensor Core 1,979 teraFLOPS is with sparsity; dense is half"),
            "peak_fp8": _f(1979e12, "derived", "h100-page", "FP8 Tensor Core 3,958 teraFLOPS is with sparsity; dense is half"),
            "peak_int8": _f(1979e12, "derived", "h100-page", "INT8 Tensor Core 3,958 TOPS is with sparsity; dense is half"),
            "sram_bytes": _f(50e6, "spec", "hopper-blog", "50 MB L2 cache (H100 SXM5)"),
            "sram_bw": _f(None, "undisclosed", "hopper-blog", "L2 bandwidth not given in the sources used"),
            "off_bytes": _f(80e9, "spec", "h100-page", "GPU Memory 80GB"),
            "off_bw": _f(3.35e12, "spec", "h100-page", "GPU Memory Bandwidth 3.35TB/s"),
            "c2c_bidir": _f(900e9, "spec", "h100-page", "NVIDIA NVLink 900GB/s"),
            "hop_s": _f(None, "undisclosed", "h100-page", "NVLink latency not given in the sources used"),
            "power_w": _f(700, "spec", "h100-page", "Max Thermal Design Power: up to 700W (configurable)"),
            "units": _f(132, "spec", "hopper-blog", "132 SMs per GPU (H100 SXM5)"),
        },
    },
    "b200": {
        "id": "b200", "name": "NVIDIA B200 (as in GB200 NVL72)", "family": "gpu", "layout": "tensor",
        "avail": _f("product", "spec", "gb200-page", "GB200 NVL72 product page with specifications"),
        "fields": {
            "peak_bf16": _f(2.5e15, "derived", "gb200-page", "GB200 Superchip FP16/BF16 10 PFLOPS sparse for 2 GPUs: dense per GPU = 10 / 2 / 2"),
            "peak_fp8": _f(5e15, "spec", "bwu-blog", "Table 2, Blackwell: FP8 dense 5 PetaFLOPS (GB200 Superchip: 20 PFLOPS sparse for 2 GPUs, the same)"),
            "peak_int8": _f(5e15, "derived", "gb200-page", "GB200 Superchip INT8 20 POPS sparse for 2 GPUs: dense per GPU = 20 / 2 / 2"),
            "peak_fp4": _f(10e15, "spec", "bwu-blog", "Table 2, Blackwell: NVFP4 dense 10 PetaFLOPS (GB200 Superchip: 20 PFLOPS dense for 2 GPUs)"),
            "sram_bytes": _f(None, "undisclosed", "bwu-blog", "L2 size not given in the sources used"),
            "sram_bw": _f(None, "undisclosed", "bwu-blog", "not given in the sources used"),
            "off_bytes": _f(186e9, "derived", "gb200-page", "GB200 Superchip: 372 GB HBM3E for 2 GPUs, so 186 GB per GPU (Blackwell's maximum is 192 GB, bwu-blog Table 2)"),
            "off_bw": _f(8e12, "derived", "gb200-page", "GB200 Superchip: 16 TB/s for 2 GPUs, so 8 TB/s per GPU (bwu-blog Table 2: 8 TB/s)"),
            "c2c_bidir": _f(1.8e12, "derived", "gb200-page", "GB200 Superchip NVLink 3.6 TB/s for 2 GPUs: 1.8 TB/s per GPU"),
            "hop_s": _f(None, "undisclosed", "gb200-page", "NVLink latency not given in the sources used"),
            "power_w": _f(1200, "spec", "bwu-blog", "Table 2, Blackwell: Max power (TGP) up to 1,200W: the GPU's board power, as the H100's 700 W TDP is"),
            "units": _f(None, "undisclosed", "bwu-blog", "SM count not given in the sources used"),
        },
    },
    "mi355x": {
        "id": "mi355x", "name": "AMD Instinct MI355X", "family": "gpu", "layout": "tensor",
        "avail": _f("product", "spec", "amd-mi355x", "product page with specifications; launch date 6/12/2025"),
        "fields": {
            "peak_bf16": _f(2.5e15, "spec", "amd-mi355x", "Peak bfloat16 (BF16) Matrix Performance 2.5 PFLOPs (dense; 5 PFLOPs with structured sparsity)"),
            "peak_fp8": _f(5e15, "spec", "amd-mi355x", "Peak OCP-FP8 Matrix Performance (E5M2, E4M3) 5 PFLOPs (dense)"),
            "peak_int8": _f(5e15, "spec", "amd-mi355x", "Peak INT8 Matrix Performance 5 POPs (dense)"),
            "peak_fp4": _f(10.1e15, "spec", "amd-mi355x", "Peak Microscaling Four-bit Precision Matrix (MXFP4) Performance 10.1 PFLOPs"),
            "sram_bytes": _f(256e6, "spec", "amd-mi355x", "Last Level Cache (LLC) 256 MB (a cache in front of HBM; the model reads weights from HBM)"),
            "sram_bw": _f(None, "undisclosed", "amd-mi355x", "LLC bandwidth not given on the page"),
            "off_bytes": _f(288e9, "spec", "amd-mi355x", "Dedicated Memory Size 288 GB, HBM3E"),
            "off_bw": _f(8e12, "spec", "amd-mi355x", "Peak Memory Bandwidth 8 TB/s"),
            "c2c_bidir": _f(None, "undisclosed", "amd-mi355x", "7 Infinity Fabric links at 153 GB/s each; the page does not say whether that is per direction"),
            "hop_s": _f(None, "undisclosed", "amd-mi355x", "link latency not given"),
            "power_w": _f(1400, "spec", "amd-mi355x", "Typical Board Power (TBP) 1400W"),
            "units": _f(256, "spec", "amd-mi355x", "Compute Units 256 (1,024 matrix cores)"),
        },
    },
    "tpu-v6e": {
        "id": "tpu-v6e", "name": "Google TPU v6e (Trillium)", "family": "systolic", "layout": "tensor",
        "avail": _f("product", "spec", "tpu-v6e", "Cloud TPU documentation lists v6e configurations"),
        "fields": {
            "peak_bf16": _f(918e12, "spec", "tpu-v6e", "Peak compute per chip (bf16) 918 TFLOPs"),
            "peak_fp8": _f(918e12, "spec", "tpu-7x", "comparison table: Peak compute per chip (FP8) 918 TFLOPs for v6e"),
            "peak_int8": _f(1836e12, "spec", "tpu-v6e", "Peak compute per chip (Int8) 1836 TOPs"),
            "sram_bytes": _f(None, "undisclosed", "tpu-v6e", "on-chip memory size not given in the sources used"),
            "sram_bw": _f(None, "undisclosed", "tpu-v6e", "not given in the sources used"),
            "off_bytes": _f(32e9, "spec", "tpu-v6e", "HBM capacity per chip 32 GB"),
            "off_bw": _f(1.638e12, "spec", "tpu-v6e", "HBM bandwidth per chip 1638 GBps"),
            "c2c_bidir": _f(800e9, "spec", "tpu-v6e", "Bidirectional inter-chip interconnect (ICI) bandwidth (per chip) 800 GBps"),
            "hop_s": _f(None, "undisclosed", "tpu-v6e", "ICI latency not given in the sources used"),
            "power_w": _f(None, "undisclosed", "tpu-v6e", "chip power not given in the sources used"),
            "units": _f(2, "spec", "tpu-v6e", "one TensorCore with 2 MXUs"),
            "array_rows": _f(256, "spec", "tpu-arch", "An MXU is composed of 256 x 256 (TPU v6e and TPU7x) multiply-accumulators in a systolic array"),
            "array_cols": _f(256, "spec", "tpu-arch", "256 x 256"),
        },
    },
    "tpu-7x": {
        "id": "tpu-7x", "name": "Google TPU7x (Ironwood)", "family": "systolic", "layout": "tensor",
        "avail": _f("product", "spec", "tpu-7x", "Cloud TPU documentation lists TPU7x configurations"),
        "fields": {
            "peak_bf16": _f(2307e12, "spec", "tpu-7x", "Peak compute per chip (BF16) 2307 TFLOPs"),
            "peak_fp8": _f(4614e12, "spec", "tpu-7x", "Peak compute per chip (FP8) 4614 TFLOPs"),
            "peak_int8": _f(None, "undisclosed", "tpu-7x", "INT8 rate not given in the sources used"),
            "sram_bytes": _f(None, "undisclosed", "tpu-7x", "VMEM size 'is a tunable parameter'; total not given"),
            "sram_bw": _f(None, "undisclosed", "tpu-7x", "'significantly higher bandwidth to the MXU than HBM'; no figure"),
            "off_bytes": _f(192 * GiB, "spec", "tpu-7x", "HBM capacity per chip 192 GiB (the prose says '192 GB'; the table's GiB is used)"),
            "off_bw": _f(7.38e12, "spec", "tpu-7x", "HBM bandwidth per chip 7380 GBps"),
            "c2c_bidir": _f(1.2e12, "spec", "tpu-7x", "Bidirectional ICI bandwidth per chip 1200 GBps"),
            "hop_s": _f(None, "undisclosed", "tpu-7x", "ICI latency not given in the sources used"),
            "power_w": _f(None, "undisclosed", "tpu-7x", "chip power not given in the sources used"),
            "units": _f(2, "spec", "tpu-7x", "two TensorCores per chip (one per chiplet)"),
            "array_rows": _f(256, "spec", "tpu-arch", "MXU of 256 x 256 multiply-accumulators (TPU v6e and TPU7x)"),
            "array_cols": _f(256, "spec", "tpu-arch", "256 x 256"),
        },
    },
    "trn2": {
        "id": "trn2", "name": "AWS Trainium2", "family": "systolic", "layout": "tensor",
        "avail": _f("product", "spec", "trn2-arch", "Neuron documentation for Trn2 instances"),
        "fields": {
            "peak_bf16": _f(667e12, "spec", "trn2-arch", "667 BF16/FP16/TF32 TFLOPS"),
            "peak_fp8": _f(1299e12, "spec", "trn2-arch", "1,299 FP8 TFLOPS"),
            "peak_int8": _f(None, "undisclosed", "trn2-arch", "INT8 rate not given in the sources used"),
            "sram_bytes": _f(224e6, "derived", "nc3-arch", "8 NeuronCore-v3 x 28 MB on-chip SRAM each"),
            "sram_bw": _f(None, "undisclosed", "nc3-arch", "not given in the sources used"),
            "off_bytes": _f(96 * GiB, "spec", "trn2-arch", "96 GiB of device memory"),
            "off_bw": _f(2.9e12, "spec", "trn2-arch", "2.9 TB/sec of bandwidth"),
            "c2c_bidir": _f(1.28e12, "spec", "trn2-arch", "NeuronLink-v3 provides 1.28 TB/sec bandwidth per chip (direction not stated; taken as both directions)"),
            "hop_s": _f(None, "undisclosed", "trn2-arch", "NeuronLink latency not given in the sources used"),
            "power_w": _f(None, "undisclosed", "trn2-arch", "chip power not given in the sources used"),
            "units": _f(8, "spec", "trn2-arch", "eight NeuronCore-v3 per chip"),
            "array_rows": _f(128, "spec", "nki-trn2", "'a grid of 128x128 processing elements'"),
            "array_cols": _f(128, "spec", "nki-trn2", "128x128"),
        },
    },
    "wse-3": {
        "id": "wse-3", "name": "Cerebras WSE-3 (CS-3)", "family": "wafer", "layout": "pipeline",
        "avail": _f("product", "spec", "cs3-datasheet", "CS-3 datasheet; Cerebras Inference ran on CS-3"),
        "fields": {
            "peak_fp16_sparse": _f(125e15, "spec", "cs3-datasheet", "125 petaFLOPS FP16, '* FLOPS shown as sparse'"),
            "peak_bf16": _f(None, "undisclosed", "cerebras-wp", "dense rate = 900,000 cores x 8-wide FP16 SIMD x clock; the clock is not given"),
            "peak_fp8": _f(None, "undisclosed", "cs3-datasheet", "no FP8 figure"),
            "peak_int8": _f(None, "undisclosed", "cs3-datasheet", "no INT8 figure"),
            "sram_bytes": _f(44e9, "spec", "cs3-datasheet", "44 GB of on-chip memory"),
            "sram_bw": _f(21e15, "spec", "cs3-datasheet", "21 PB/s memory bandwidth"),
            "off_bytes": _f(0, "spec", "cerebras-wp", "no DRAM on the wafer; weights for training stream from an external MemoryX unit"),
            "off_bw": _f(0, "spec", "cerebras-wp", "no DRAM on the wafer"),
            "io_bps": _f(1.2e12, "spec", "cs3-datasheet", "I/O Bandwidth 1.2 Tb/s"),
            "c2c_bidir": _f(None, "undisclosed", "cs3-datasheet", "wafer-to-wafer bandwidth for inference not given; system I/O is 1.2 Tb/s"),
            "hop_s": _f(None, "undisclosed", "cs4-page", "CS-3 wafer-to-wafer latency not given (CS-4 is 'less than half the latency')"),
            "power_w": _f(27e3, "spec", "cs3-datasheet", "System Power 27 kW max: a whole CS-3 system, not the wafer alone"),
            "units": _f(900000, "spec", "cs3-datasheet", "900,000 cores"),
            "area_mm2": _f(46225, "spec", "cs3-datasheet", "46,225 mm2 silicon"),
            "transistors": _f(4e12, "spec", "cs3-datasheet", "4 trillion transistors"),
            "redundancy": _f("spare cores and routes; defects shut down and routed round", "spec", "cerebras-chip", "'redundant compute cores, redundant routing, and a fail-in-place architecture that shuts flaws down and routes around them'"),
            "dies": _f(84, "spec", "cerebras-wp", "'The final chip contains a total of 84 dies'"),
            "core_um2": _f(38000, "spec", "cerebras-wp", "'approximately 38,000 um2 of silicon area' per core"),
            "core_sram": _f(48e3, "spec", "cerebras-wp", "'48 kB of local SRAM dedicated to the core'"),
            "fabric_bps": _f(214e15, "spec", "cs3-datasheet", "214 Pb/s fabric bandwidth"),
        },
    },
    "wse-3t": {
        "id": "wse-3t", "name": "Cerebras WSE-3 Turbo (CS-4)", "family": "wafer", "layout": "pipeline",
        "avail": _f("announced", "spec", "cs4-page", "'First CS-4 shipments begin this quarter' (read 2026-10-15)"),
        "fields": {
            "peak_fp16_sparse": _f(250e15, "spec", "cs4-datasheet", "AI Compute 250 PFLOPS per wafer, '*AI compute shown as sparse FP16'"),
            "peak_bf16": _f(None, "undisclosed", "cs4-datasheet", "dense rate not given"),
            "peak_fp8": _f(None, "undisclosed", "cs4-datasheet", "no FP8 figure"),
            "peak_int8": _f(None, "undisclosed", "cs4-datasheet", "no INT8 figure"),
            "sram_bytes": _f(44e9, "spec", "cs4-datasheet", "900,000 cores; 44 GB SRAM"),
            "sram_bw": _f(43.2e15, "spec", "cs4-datasheet", "Memory Bandwidth 43.2 PB/s per wafer"),
            "off_bytes": _f(0, "spec", "cs4-datasheet", "no DRAM on the wafer"),
            "off_bw": _f(0, "spec", "cs4-datasheet", "no DRAM on the wafer"),
            "io_bps": _f(2.4e12, "spec", "cs4-datasheet", "I/O Bandwidth 2.4 Tbit/s per wafer"),
            "c2c_bidir": _f(None, "undisclosed", "cs4-datasheet", "wafer-link bandwidth per direction not given; I/O is 2.4 Tbit/s per wafer"),
            "hop_s": _f(2e-6, "spec", "cs4-datasheet", "I/O latency 2 microseconds ('wafer-to-wafer latency as low as 2 microseconds')"),
            "power_w": _f(None, "undisclosed", "cs4-datasheet", "power not given in the sources used"),
            "units": _f(900000, "spec", "cs4-datasheet", "900,000 cores"),
            "area_mm2": _f(46225, "spec", "cs4-datasheet", "46,225 mm2"),
            "dies": _f(84, "spec", "cerebras-wp", "84 dies (WSE-3)"),
            "core_um2": _f(38000, "spec", "cerebras-wp", "~38,000 um2 per core (WSE-3)"),
            "core_sram": _f(48e3, "spec", "cerebras-wp", "48 kB per core (WSE-3)"),
            "fabric_bps": _f(53.5e15 * 8, "derived", "cs4-datasheet", "On-Chip Fabric Bandwidth 53.5 PB/s, in bits"),
        },
    },
    "groqchip": {
        "id": "groqchip", "name": "GroqChip (first-generation LPU)", "family": "lpu", "layout": "pipeline",
        "avail": _f("retired", "spec", "groqchip-brief", "brief v1.5 (2022): 'In production'; superseded by the Groq 3 LPU"),
        "fields": {
            "peak_bf16": _f(None, "undisclosed", "groqchip-brief", "FP16 188 TFLOPs is given, BF16 is not"),
            "peak_fp16": _f(188e12, "spec", "groqchip-brief", "Up to 750 TOPs, 188 TFLOPs (INT8, FP16 @900 MHz)"),
            "peak_fp8": _f(None, "undisclosed", "groqchip-brief", "no FP8"),
            "peak_int8": _f(750e12, "spec", "groqchip-brief", "Up to 750 TOPs (INT8 @900 MHz)"),
            "sram_bytes": _f(230e6, "spec", "groqchip-brief", "230 MB SRAM per chip"),
            "sram_bw": _f(80e12, "spec", "groqchip-brief", "Up to 80 TB/s on-die memory bandwidth"),
            "off_bytes": _f(0, "spec", "groqchip-brief", "'without the need for external memory'"),
            "off_bw": _f(0, "spec", "groqchip-brief", "no external memory"),
            "c2c_links": _f(16, "spec", "groqchip-brief", "16 integrated RealScale chip-to-chip interconnects"),
            "link_bps": _f(100e9, "spec", "groq-isca22", "links run at 25 Gbps per lane, 'a combined bandwidth of 100 Gbps across the four lanes in a link'"),
            "c2c_bidir": _f(3.2e12 / 8, "derived", "groq-isca22", "16 links x 100 Gb/s per direction = 1.6 Tb/s = 200 GB/s each way, 400 GB/s both ways"),
            "hop_s": _f(722e-9, "spec", "groq-isca22", "'a pipelined network latency of 722 ns per hop'"),
            "node_chips": _f(8, "spec", "groq-isca22", "a 4U node houses eight TSPs, fully connected by 7 local links each"),
            "power_w": _f(215, "spec", "groqchip-brief", "Power Max: 300W; TDP: 215W; Average: 185W (TDP used)"),
            "clock_hz": _f(900e6, "spec", "groqchip-brief", "@900 MHz"),
            "process": _f("14 nm", "spec", "groqchip-brief", "Process Node 14nm"),
        },
    },
    "groq3": {
        "id": "groq3", "name": "NVIDIA Groq 3 LPU (LPX rack)", "family": "lpu", "layout": "pipeline",
        "avail": _f("announced", "spec", "groq-home", "'When released, each NVIDIA Groq 3 LPX rack connects 256 next-generation LPU accelerators'"),
        "fields": {
            "peak_bf16": _f(None, "undisclosed", "lpx-page", "not given in the sources used"),
            "peak_fp8": _f(315e15 / 256, "derived", "groq-home", "315 PFLOPS of FP8 inference compute per rack of 256 LPUs (dense or sparse not stated)"),
            "peak_int8": _f(None, "undisclosed", "lpx-page", "not given in the sources used"),
            "sram_bytes": _f(500e6, "spec", "lpx-page", "Each LPU accelerator delivers 500 megabytes (MB) of SRAM"),
            "sram_bw": _f(150e12, "spec", "lpx-page", "150 terabytes per second (TB/s) of SRAM bandwidth"),
            "off_bytes": _f(0, "spec", "lpx-page", "SRAM on the LPU; the rack adds 12 TB of DDR5 'for large models and workloads'"),
            "off_bw": _f(0, "spec", "lpx-page", "no memory on the LPU besides SRAM"),
            "c2c_bidir": _f(2.5e12, "spec", "lpx-page", "2.5 TB/s scale-up bandwidth (direction not stated; taken as both directions)"),
            "hop_s": _f(None, "undisclosed", "lpx-page", "chip-to-chip latency not given in the sources used"),
            "power_w": _f(None, "undisclosed", "lpx-page", "power not given in the sources used"),
            "rack_chips": _f(256, "spec", "lpx-page", "256 LPU chips per rack"),
        },
    },
}

# Chapters 6-8: the dataflow, mesh and edge parts the model runs.
PARTS.update({
    "sn40l": {
        "id": "sn40l", "name": "SambaNova SN40L RDU", "family": "dataflow", "layout": "tensor",
        "avail": _f("product", "spec", "sambanova-rdu", "the chip page lists the SN40 beside the SN50: 'the SN40 is well-suited for existing inference deployments'"),
        "fields": {
            "peak_bf16": _f(638e12, "spec", "sn40l", "Table II: Compute Capability 638 BFLOAT16 (BF16) TFLOPs"),
            "peak_fp8": _f(None, "undisclosed", "sn40l", "no FP8 figure in the paper"),
            "peak_int8": _f(None, "undisclosed", "sn40l", "no INT8 figure in the paper"),
            "sram_bytes": _f(520e6, "spec", "sn40l", "Table II: SRAM Capacity 520 MB (the text says 520 MiB)"),
            "sram_bw": _f(None, "undisclosed", "sn40l", "the PMUs 'in aggregate provide hundreds of TBps of on-chip memory bandwidth'; no figure"),
            "off_bytes": _f(64e9, "spec", "sn40l", "Table II: HBM Capacity 64 GB"),
            "off_bw": _f(1.8e12, "spec", "sn40l", "Table II: HBM Bandwidth 1.8 TB/s"),
            "ddr_bytes": _f(1.5e12, "spec", "sn40l", "Table II: DDR Capacity 1.5 TB (the chip page now lists 'Up to 512 GB' DDR5 for the SN40)"),
            "ddr_bw": _f(200e9, "spec", "sn40l", "Table II: DDR Bandwidth 200 GB/s"),
            "c2c_bidir": _f(None, "undisclosed", "sn40l", "socket-to-socket bandwidth not given in the sources used"),
            "hop_s": _f(None, "undisclosed", "sn40l", "socket-to-socket latency not given in the sources used"),
            "power_w": _f(None, "undisclosed", "sambanova-rdu", "chip power not given in the sources used"),
            "units": _f(1040, "spec", "sn40l", "Table II: PCU Count 1040 (and 1040 PMUs)"),
            "process": _f("5 nm", "spec", "sn40l", "Table II: 5nm, two dies per socket"),
        },
    },
    "gc2": {
        "id": "gc2", "name": "Graphcore GC2 IPU (first generation)", "family": "dataflow", "layout": "pipeline",
        "avail": _f("retired", "spec", "graphcore-home", "Graphcore's site lists no IPU products on the access date (its products page returns 404); the GC2 is the IPU the 2019 paper measured"),
        "fields": {
            "peak_bf16": _f(None, "undisclosed", "ipu-dissect", "no BF16 figure"),
            "peak_fp16": _f(124.5e12, "spec", "ipu-dissect", "'124.5 TFlops/s in mixed precision per chip' (FP16 products, FP32 accumulation)"),
            "peak_fp8": _f(None, "undisclosed", "ipu-dissect", "no FP8"),
            "peak_int8": _f(None, "undisclosed", "ipu-dissect", "no INT8 figure"),
            "sram_bytes": _f(304 * 2**20, "spec", "ipu-dissect", "'Each tile contains 256 KiB of memory, totaling 304 MiB per device'"),
            "sram_bw": _f(45e12, "spec", "ipu-dissect", "'much higher bandwidth (45 TB/s, aggregate)'"),
            "off_bytes": _f(0, "spec", "ipu-dissect", "'the IPU offers no memory other than the distributed memories local to each tile'"),
            "off_bw": _f(0, "spec", "ipu-dissect", "no memory on the device besides the tiles' SRAM"),
            "c2c_bidir": _f(None, "undisclosed", "ipu-dissect", "not used here"),
            "hop_s": _f(None, "undisclosed", "ipu-dissect", "IPU-to-IPU latency not used here"),
            "power_w": _f(150, "reported", "ipu-dissect", "the paper's 'nominal power figures (250 W for the GPU, 150 W for the IPU)'"),
            "units": _f(1216, "spec", "ipu-dissect", "'Each IPU contains 1,216 processing elements called tiles'"),
            "tile_sram": _f(256 * 1024, "spec", "ipu-dissect", "256 KiB per tile"),
        },
    },
    "blackhole": {
        "id": "blackhole", "name": "Tenstorrent Blackhole (p150a card)", "family": "mesh", "layout": "tensor",
        "avail": _f("product", "spec", "tt-hardware", "p150a 'In stock: Ships in 2 weeks'"),
        "fields": {
            "peak_bf16": _f(None, "undisclosed", "tt-hardware", "no BF16 figure on the page"),
            "peak_fp8": _f(664e12, "spec", "tt-hardware", "TeraFLOPS (BLOCKFP8) 664 (a block floating-point format; dense or sparse not stated)"),
            "peak_int8": _f(None, "undisclosed", "tt-hardware", "no INT8 figure"),
            "sram_bytes": _f(180e6, "spec", "tt-hardware", "SRAM 180MB"),
            "sram_bw": _f(None, "undisclosed", "tt-hardware", "not given"),
            "off_bytes": _f(32e9, "spec", "tt-hardware", "Memory 32 GB GDDR6"),
            "off_bw": _f(512e9, "spec", "tt-hardware", "Memory Bandwidth 512 GB/sec"),
            "c2c_bidir": _f(800e9, "derived", "tt-hardware", "4x QSFP-DD 800G ports: 3.2 Tb/s = 400 GB/s each way, 800 GB/s both ways"),
            "hop_s": _f(None, "undisclosed", "tt-hardware", "Ethernet latency not given"),
            "power_w": _f(300, "spec", "tt-hardware", "TBP (Total Board Power) 300W: the card"),
            "units": _f(120, "spec", "tt-hardware", "Tensix Cores 120"),
            "riscv": _f(16, "spec", "tt-hardware", "'Big RISC-V' Cores 16"),
            "clock_hz": _f(1.35e9, "spec", "tt-hardware", "AI Clock 1.35 GHz"),
        },
    },
    "m4": {
        "id": "m4", "name": "Apple M4 (Neural Engine)", "family": "fpga-npu", "layout": "tensor",
        "avail": _f("product", "spec", "apple-mbp-m4", "MacBook Pro with M4 (October 2024)"),
        "fields": {
            "peak_bf16": _f(None, "undisclosed", "apple-m4", "no floating-point figure"),
            "peak_fp8": _f(None, "undisclosed", "apple-m4", "no FP8 figure"),
            "peak_int8": _f(38e12, "spec", "apple-m4", "'38 trillion operations per second'; Apple does not state the format, so the model treats it as the 8-bit rate"),
            "sram_bytes": _f(None, "undisclosed", "apple-m4", "on-chip memory not given"),
            "sram_bw": _f(None, "undisclosed", "apple-m4", "not given"),
            "off_bytes": _f(32e9, "spec", "apple-mbp-m4", "'starts with 16GB of faster unified memory with support for up to 32GB' (the largest M4 configuration is used)"),
            "off_bw": _f(120e9, "spec", "apple-mbp-m4", "'120GB/s of memory bandwidth' (unified memory, shared by the CPU, GPU and Neural Engine)"),
            "c2c_bidir": _f(None, "undisclosed", "apple-m4", "a single device"),
            "hop_s": _f(None, "undisclosed", "apple-m4", "a single device"),
            "power_w": _f(None, "undisclosed", "apple-m4", "Neural Engine power not given"),
            "units": _f(16, "spec", "apple-m4", "a 16-core Neural Engine"),
        },
    },
})

PART_ORDER = ("h100", "b200", "mi355x", "tpu-v6e", "tpu-7x", "trn2", "wse-3", "wse-3t", "groqchip", "groq3", "sn40l", "gc2", "blackhole", "m4")

# Parts quoted on the site but not run through placement or decode: a
# capacity or bandwidth the model needs is not disclosed, or (SN50) the
# memory bandwidth is not. Same field structure as PARTS.
OTHER_PARTS: dict[str, dict[str, Any]] = {
    "sn50": {
        "id": "sn50", "name": "SambaNova SN50 RDU", "family": "dataflow", "also": ["sambanova-sn50-ds"],
        "avail": _f("announced", "spec", "sambanova-rdu", "the chip page lists specifications and a datasheet; it does not state that the SN50 is shipping"),
        "fields": {
            "peak_bf16": _f(1600e12, "spec", "sambanova-rdu", "Compute 1600 TFLOPS at BF16"),
            "peak_fp8": _f(3200e12, "spec", "sambanova-rdu", "3200 TFLOPS at FP8"),
            "sram_bytes": _f(432e6, "spec", "sambanova-rdu", "On-chip SRAM 432 MB"),
            "off_bytes": _f(64e9, "spec", "sambanova-rdu", "HBM2E 64 GB"),
            "off_bw": _f(None, "undisclosed", "sambanova-rdu", "HBM bandwidth not given"),
            "ddr_bytes": _f(512e9, "spec", "sambanova-rdu", "DDR5 Up to 512 GB (the datasheet says 'Up to 2 TB')"),
            "scale_up": _f(256, "spec", "sambanova-rdu", "Scale-up: Up to 256 RDUs"),
        },
    },
    "v70": {
        "id": "v70", "name": "AMD Alveo V70 (Versal AI Core)", "family": "fpga-npu",
        "avail": _f("retired", "spec", "amd-v70-part", "'This product has been discontinued and is no longer offered for sale'"),
        "fields": {
            "peak_int8": _f(404e12, "spec", "amd-v70", "TOPS (INT8) 404"),
            "peak_bf16": _f(202e12, "spec", "amd-v70", "TOPS (BF16) 202"),
            "sram_bw": _f(47.6e12, "spec", "amd-v70", "47.6 TB/s for internal memory"),
            "off_bytes": _f(None, "undisclosed", "amd-v70", "DDR capacity not given on the page"),
            "off_bw": _f(76.8e9, "spec", "amd-v70", "76.8 GB/s for external DDR"),
            "power_w": _f(75, "spec", "amd-v70", "Power (TDP) 75W"),
        },
    },
    "v80": {
        "id": "v80", "name": "AMD Alveo V80 (Versal HBM)", "family": "fpga-npu",
        "avail": _f("product", "spec", "amd-v80", "part page with General Specifications and 'Buy'"),
        "fields": {
            "peak_int8": _f(None, "undisclosed", "amd-v80", "no TOPS figure: an FPGA's arithmetic rate depends on the design (10,848 DSP slices)"),
            "peak_bf16": _f(None, "undisclosed", "amd-v80", "no FLOPS figure"),
            "sram_bytes": _f((132e6 + 541e6) / 8, "derived", "amd-v80", "132 Mb Block RAM + 541 Mb UltraRAM = 673 Mb, about 84 MB"),
            "off_bytes": _f(32e9, "spec", "amd-v80", "HBM Capacity 32 GB HBM2e DRAM (2x 16 GB stacks)"),
            "off_bw": _f(820e9, "spec", "amd-v80", "820 GB/s peak bandwidth (the overview page's chart says 810 GB/s)"),
            "luts": _f(2.6e6, "spec", "amd-v80", "2.6M Look-up tables (LUTs)"),
            "dsps": _f(10848, "spec", "amd-v80", "10,848 DSP slices"),
            "power_w": _f(190, "spec", "amd-v80", "Up to 190W TDP"),
        },
    },
    "hailo10h": {
        "id": "hailo10h", "name": "Hailo-10H (M.2 module)", "family": "fpga-npu",
        "avail": _f("product", "spec", "hailo-10h", "M.2 module with a 'How to Buy' section"),
        "fields": {
            "peak_int4": _f(40e12, "spec", "hailo-10h", "40|20 TOPS (INT4|8)"),
            "peak_int8": _f(20e12, "spec", "hailo-10h", "40|20 TOPS (INT4|8)"),
            "off_bytes": _f(8e9, "spec", "hailo-10h", "ON-Module DDR 4|8GB LPDDR4/4X (the larger is used)"),
            "off_bw": _f(None, "undisclosed", "hailo-10h", "memory bandwidth not given"),
            "power_w": _f(2.5, "spec", "hailo-10h", "Consumes 2.5W (typical)"),
        },
    },
}
OTHER_ORDER = ("sn50", "v70", "v80", "hailo10h")

# Chapter 9: one analogue in-memory compute chip (a research chip, not a product).
IMC: dict[str, dict[str, Any]] = {
    "hermes": {
        "id": "hermes", "name": "IBM HERMES project chip (PCM, research)", "family": "imc",
        "avail": _f("research", "spec", "pcm-imc", "a research chip described in a paper (Le Gallo et al.); not a product"),
        "fields": {
            "cores": _f(64, "spec", "pcm-imc", "'64 256x256 AIMC cores'"),
            "rows": _f(256, "spec", "pcm-imc", "each core 'a PCM crossbar array capable of storing a 256 x 256 weight matrix'"),
            "cols": _f(256, "spec", "pcm-imc", "256 x 256"),
            "weights": _f(4194304, "spec", "pcm-imc", "'up to 4,194,304 weights can be stored on-chip'"),
            "devices_per_weight": _f(4, "spec", "pcm-imc", "'using only 4 PCM devices to encode a weight'"),
            "lat_1p": _f(133e-9, "spec", "pcm-imc", "Table 1: MVM latency 133 ns, 1-phase read (low precision)"),
            "lat_4p": _f(520e-9, "spec", "pcm-imc", "Table 1: MVM latency 520 ns, 4-phase read (high precision)"),
            "e_1p": _f(0.86e-6, "spec", "pcm-imc", "Table 1: energy 0.86 uJ, 1-phase, all 64 cores at 100% utilisation"),
            "e_4p": _f(3.38e-6, "spec", "pcm-imc", "Table 1: energy 3.38 uJ, 4-phase, all 64 cores"),
            "tops_1p": _f(63.1e12, "spec", "pcm-imc", "peak MVM throughput 63.1 TOPS (1-phase)"),
            "tops_4p": _f(16.1e12, "spec", "pcm-imc", "16.1 TOPS (4-phase)"),
            "topsw_1p": _f(9.76e12, "spec", "pcm-imc", "9.76 TOPS/W (1-phase)"),
            "topsw_4p": _f(2.48e12, "spec", "pcm-imc", "2.48 TOPS/W (4-phase)"),
            "io_bits": _f(8, "spec", "pcm-imc", "'8-bit input/output matrix-vector multiplications'"),
            "die_mm2": _f(144, "derived", "pcm-imc", "the chip is 12 mm x 12 mm"),
            "process": _f("14 nm", "spec", "pcm-imc", "'14-nm CMOS' with back-end-integrated PCM"),
        },
    },
}

# ---------------------------------------------------------------------------
# The running workload: Llama decoder models (layer shapes from the papers).
# ---------------------------------------------------------------------------

MODELS: dict[str, dict[str, Any]] = {
    "llama3-8b": {
        "id": "llama3-8b", "name": "Llama 3.1 8B", "src": "llama3",
        "layers": 32, "d": 4096, "ffn": 14336, "heads": 32, "kv_heads": 8, "vocab": 128000,
    },
    "llama3-70b": {
        "id": "llama3-70b", "name": "Llama 3.1 70B", "src": "llama3",
        "layers": 80, "d": 8192, "ffn": 28672, "heads": 64, "kv_heads": 8, "vocab": 128000,
    },
    "llama2-70b": {
        # Llama 2's paper gives no layer table; the 70B layer shape is the
        # Llama 3 70B shape (80 x 8,192, FFN 28,672, 8 KV heads), with Llama
        # 2's 32k vocabulary. The count comes to 68.98 B.
        "id": "llama2-70b", "name": "Llama 2 70B", "src": "llama2",
        "layers": 80, "d": 8192, "ffn": 28672, "heads": 64, "kv_heads": 8, "vocab": 32000,
    },
}
MODEL_ORDER = ("llama3-8b", "llama3-70b", "llama2-70b")

# Bytes per weight in each format (the format's width; no block scales).
FORMATS: dict[str, float] = {"bf16": 2.0, "fp8": 1.0, "int8": 1.0}

# ---------------------------------------------------------------------------
# Published figures the model is checked against (validation targets; never inputs).
# ---------------------------------------------------------------------------

PUBLISHED: dict[str, dict[str, Any]] = {
    "mlperf-h100": {**_f(21805.8, "spec", "mlperf-h100", "Offline 'Tokens per second: 21805.8' (8x H100 SXM, Llama 2 70B, 99.9% accuracy target; NVIDIA's blog: TensorRT-LLM with FP8 quantization)"), "also": ["nv-mlperf-v40"]},
    "mlperf-tokens-per-sample": _f(294.45, "spec", "mlperf-llama2", "Tokens per sample: 294.45 (output tokens)"),
    "mlperf-input-mean": _f(5463976 / 24576, "derived", "mlperf-dataset", "mean of tok_input_length over the 24,576 samples: 5,463,976 tokens / 24,576 = 222.33 (scripts/mlperf_input_length.py; median 152, min 48, max 1,021)"),
    "mlperf-gb200": {**_f(51736.9, "spec", "mlperf-gb200", "Offline 'Tokens per second: 51736.9' (4x GB200 in one tray, Llama 2 70B, 99.9% accuracy target, FP4 weights)"), "also": ["mlperf-gb200-meas"]},
    "mlperf-gb200-chips": _f(4, "spec", "mlperf-gb200", "system GB200-NVL72_GB200-186GB_aarch64x4: 4 GPUs"),
    "cerebras-70b-systems": _f(4, "spec", "cerebras-inference", "'70B models fit on as few as four systems' (16-bit weights)"),
    "cerebras-20b-one": _f(1, "spec", "cerebras-inference", "'20B models fit on a single CS-3'"),
    "cerebras-8b-tps": _f(1800, "spec", "cerebras-inference", "'1,800 tokens per second for Llama3.1 8B' (16-bit weights)"),
    "cerebras-70b-tps": _f(450, "spec", "cerebras-inference", "'450 tokens per second for Llama3.1 70B' (16-bit weights)"),
    "groq-70b-chips": _f(576, "reported", "nextplatform-groq", "'Groq linked together 576 of its GroqChips' for Llama 2 70B, INT8"),
    "groq-70b-tps": _f(300, "reported", "nextplatform-groq", "'the Groq setup with 576 chips was able to push above 300 tokens per second'"),
    "sn40l-sockets": _f(16, "spec", "sn40l", "Table IV: 'measured on 16 SN40L sockets'"),
    "sn40l-8b-tps": _f(1042, "spec", "sn40l", "Table IV: Llama 3.1 Instruct 8B, 1042 output tokens/second/user (BF16 weights, 8K sequence length)"),
    "sn40l-70b-tps": _f(457, "spec", "sn40l", "Table IV: Llama 3.1 Instruct 70B, 457 output tokens/second/user ('Speculative decoding is employed on 70B and 405B models')"),
}

# Quoted claims: shown on the site, attributed, never used as model inputs.
CLAIMS: dict[str, dict[str, str]] = {
    "cs4-30x": {
        "text": "CS-4 'delivers up to 30x faster inference compared to GPU systems'",
        "src": "cs4-page",
    },
    "cs4-1000": {
        "text": "CS-4 'delivers more than 1,000 tokens per second on models exceeding 50 trillion parameters' (the page; the datasheet says 10 trillion)",
        "src": "cs4-page",
    },
    "cerebras-20x": {
        "text": "Cerebras Inference 'runs Llama3.1 20x faster than GPU solutions at 1/3 the power of DGX solutions'",
        "src": "cerebras-inference",
    },
    "lpx-1000": {
        "text": "an LPX rack: '1,000 tokens/sec/user'",
        "src": "groq-home",
    },
    "cerebras-sparse-10x": {
        "text": "Cerebras cores reach 'as high as 10x more utilization than GPUs on unstructured, sparse compute'",
        "src": "cerebras-wp",
    },
    "sn40l-85": {
        "text": "SN40L: 'Dataflow enables overlapping weight loads with computation to achieve over 85% of HBM bandwidth' (the SN40L paper; it says GPU implementations 'rarely exceed 50%')",
        "src": "sn40l",
    },
    "flightllm": {
        "text": "FlightLLM on an Alveo U280: '6.0x higher energy efficiency and 1.8x better cost efficiency against commercial GPUs (e.g., NVIDIA V100S) on modern LLMs (e.g., LLaMA2-7B)' at batch 1; on a Versal VHK158, '1.2x higher throughput' than an A100",
        "src": "flightllm",
    },
    "chen-spatial": {
        "text": "a spatial (one unit per operator) GPT-2 design on an Alveo U280: 'a 1.9x speedup and a 5.7x improvement in energy efficiency compared to the NVIDIA A100 GPU in the decode stage', with a library of HLS kernels",
        "src": "chen-spatial",
    },
    "finn": {
        "text": "FINN builds FPGA accelerators for binarized networks as 'a flexible heterogeneous streaming architecture', 'with per-layer compute resources being tailored to user-provided throughput requirements': up to 12.3 million MNIST classifications per second at 0.31 us latency, under 25 W",
        "src": "finn",
    },
    "hls4ml": {
        "text": "hls4ml, 'a package based on High-Level Synthesis', builds neural networks into FPGAs for particle-physics triggers; the example jet classifier fits 'with a latency on the scale of 100 ns'",
        "src": "hls4ml",
    },
    "m4-npu": {
        "text": "M4's Neural Engine is 'faster than the neural processing unit of any AI PC today' (Apple, May 2024)",
        "src": "apple-m4",
    },
}

# Illustrative parameters (labelled on the site).
ILLUSTRATIVE: dict[str, dict[str, Any]] = {
    "d0_per_cm2": _f(0.1, "illustrative", "cerebras-wp", "defect density for chapter 4's yield picture: a round number for a mature process, not a TSMC figure (TSMC does not publish one for N5)"),
    "gpu_die_mm2": _f(814, "illustrative", "hopper-blog", "a reticle-sized die for comparison (H100's GH100 is 814 mm2 per NVIDIA's Hopper blog)"),
    "spare_frac": _f(0.015, "illustrative", "cerebras-wp", "share of cores held as spares; Cerebras says it has 'redundant compute cores' but not how many"),
}
