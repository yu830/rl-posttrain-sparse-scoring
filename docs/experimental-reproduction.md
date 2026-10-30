# Experimental Worker reproduction / 实验版 Worker 复现

The commands invoke the real pinned verl `TeacherFSDPWorker` directly on one GPU. Inputs are deterministic synthetic text and candidate IDs; this is not Ray dispatch or RL training. 以下命令直接调用固定 verl 的真实单卡 Worker；输入为确定性合成文本和候选 ID，不代表 Ray 调度或完整 RL 训练。

## Environment / 环境

Measured: WSL2/Ubuntu, RTX 5070 Laptop (8151 MiB reported), Python 3.12.3, PyTorch 2.14.1+cu130, Transformers 5.19.0, tensordict 0.10.0, Ray 2.59.0, PEFT 0.21.2, Hydra 1.3.8, Accelerate 1.15.0, torchdata 0.11.0 and codetiming 1.4.0. These are observed versions, not a complete reproducible environment lock. The earlier broader environment is in `requirements-g2.lock`; Worker setup and its additional dependencies are in the [integration guide](../integrations/verl_6920/README.md) / [中文集成说明](../integrations/verl_6920/README.zh-CN.md).

Use the pinned Qwen2.5-0.5B-Instruct snapshot `7ae557604adf67be50417f59c2c2f167def9a775` and a separately patched verl checkout at `87cd055e872f9fdd7b976fcd797a0934b6dba265`. The probe checks the model/tokenizer hashes and verl commit. Set the paths below; `WORKER_DEPS` may be empty when dependencies are installed in the project-local environment. Keep outputs outside the source checkout on persistent storage.

使用固定模型快照和独立打补丁的 verl checkout。先设置以下路径；若依赖已在项目本地环境中，`WORKER_DEPS` 可留空。输出应放在仓库外的持久存储。

```bash
export SCORING_DIR=/absolute/path/to/rl-posttrain-sparse-scoring
export VERL_DIR=/absolute/path/to/patched-verl
export MODEL_DIR=/absolute/path/to/Qwen2.5-0.5B-Instruct
export WORKER_DEPS=/absolute/path/to/optional-worker-dependencies
export OUT_DIR=/absolute/path/to/a/new/persistent-output-directory
cd "$SCORING_DIR"
mkdir -p "$OUT_DIR"
export PYTHONPATH="$SCORING_DIR:$WORKER_DEPS${PYTHONPATH:+:$PYTHONPATH}"
```

The measured runs prechecked host available memory ≥4 GiB; free GPU memory ≥5120 MiB for long Dense or ≥4096 MiB for short/Chunked; GPU utilization ≥75% stopped a run. Use `nvidia-smi` before each process and do not interrupt others. These individual commands do **not** implement that external resource gate. Preserve OOM, sampler failures and logs.

实测每个进程前检查主机可用内存至少 4 GiB、长 Dense 设备空闲至少 5120 MiB、短/Chunked 至少 4096 MiB；GPU 利用率达到 75% 即停止。下列单条命令不包含该外部资源门；请先检查，保留 OOM 和采样失败记录。

## Real Worker: 8192 rows / 真实 Worker：8192 行

```bash
python scripts/long_worker_probe.py --verl-dir "$VERL_DIR" --model-dir "$MODEL_DIR" \
  --out "$OUT_DIR/dense-8192.json" --mode dense --lengths 8192 --k 32 \
  --chunk-rows 256 --allocator-fraction 0.50 --warmup 0 --repeats 1
python scripts/long_worker_probe.py --verl-dir "$VERL_DIR" --model-dir "$MODEL_DIR" \
  --out "$OUT_DIR/chunked-8192.json" --mode chunked --lengths 8192 --k 32 \
  --chunk-rows 256 --allocator-fraction 0.50 --warmup 0 --repeats 1
python scripts/compare_capacity_outputs.py --dense "$OUT_DIR/dense-8192.call00.pt" \
  --candidate "$OUT_DIR/chunked-8192.call00.pt" --out "$OUT_DIR/compare-8192.json"
```

The comparison validates artifact identity, hashes, shape, offsets, every candidate value and ordered Top-K. A changed Top-K order or set is `FAIL_OR_TIE_UNKNOWN`: compact IDs alone cannot prove a tie. For the 9216-row cross-budget reference, use length 9216 with Dense fraction 0.55 and Chunked fraction 0.50; the higher-budget reference is not a same-budget capacity result.

比较器验证身份、哈希、形状、offset、全部候选值和有序 Top-K；次序或集合不同但无法证明并列时为 `FAIL_OR_TIE_UNKNOWN`。9216 行参考使用 Dense 0.55、Chunked 0.50，仅用于正确性比较。

## No artificial fraction: ABBA / 不设置人工 fraction

```bash
for item in dense-a0 chunk-b0 chunk-b1 dense-a1; do
  case "$item" in dense-*) mode=dense ;; *) mode=chunked ;; esac
  python scripts/long_worker_probe.py --verl-dir "$VERL_DIR" --model-dir "$MODEL_DIR" \
    --out "$OUT_DIR/$item.json" --mode "$mode" --lengths 1024 --k 32 \
    --chunk-rows 256 --warmup 1 --repeats 3
done
python scripts/compare_capacity_outputs.py --dense "$OUT_DIR/dense-a0.call01.pt" \
  --candidate "$OUT_DIR/chunk-b0.call01.pt" --out "$OUT_DIR/natural-ab.json"
```

Each command launches a fresh process. Omitting `--allocator-fraction` does not call the setter; confirm `allocator_fraction_installed=false` and the recorded actual fraction in each JSON. Compare A/A and B/B artifacts too. The 100 ms `nvidia-smi` samples include other processes and can miss peaks. Clock disagreement prevents a speedup claim; these 1024-row runs do not establish unrestricted capacity.

每条命令是独立进程。省略 fraction 后须在 JSON 核对实际值及 `allocator_fraction_installed=false`；同时比较 A/A、B/B 工件。设备采样包含其他进程且可能漏峰；时钟分歧未解，1024 行结果不证明加速或自然容量上限。

## Tests and report / 测试与报告

```bash
VERL_6920_SOURCE="$VERL_DIR" VERL_MASK_REGRESSION_SOURCE="$VERL_DIR" \
  VERL_MASK_REGRESSION_DEPS="$WORKER_DEPS" python -m pytest -q
bash scripts/build_experimental_report.sh
```

The two actual upstream metric parity tests explicitly skip without the fixed checkout and dependencies. The report builder needs XeLaTeX, BibTeX, Latin Modern and a full Chinese font; alternatively set `LATEX_ENGINE=tectonic` with Tectonic available. It writes only new experimental PDFs. New report text awaits Human review.

上游指标对照缺少固定 checkout 时会明确跳过。报告构建需要 XeLaTeX、BibTeX、Latin Modern 和完整中文字体；也可使用 Tectonic。构建只写新实验报告，新增正文仍待 Human 审核。
