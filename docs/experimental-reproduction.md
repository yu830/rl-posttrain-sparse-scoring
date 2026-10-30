# Experimental Worker reproduction / 实验版 Worker 复现

The commands invoke the real pinned verl `TeacherFSDPWorker` directly on one GPU. Inputs are deterministic synthetic text and candidate IDs; this is not Ray dispatch or RL training. 以下命令直接调用固定 verl 的真实单卡 Worker；输入为确定性合成文本和候选 ID，不代表 Ray 调度或完整 RL 训练。

## Environment and clean setup / 环境与干净安装

The Worker lock is for Linux x86_64, glibc >= 2.28 and CPython 3.12. It was installed with uv 0.12.3 from PyPI, with distribution hashes checked; the tested runtime reports PyTorch 2.14.1+cu130 and CUDA 13.0. Use a compatible NVIDIA driver. This is a separate environment from the earlier G2/SGLang snapshot: `requirements-g2.lock` pins `packaging==26.3`, which conflicts with the Worker dependency `pyvers` requiring `packaging<26`. The dedicated lock uses `packaging==25.0`. Its tested closure contains 105 packages. The model below is the pinned Qwen2.5-0.5B-Instruct snapshot; the final clean run downloaded that snapshot separately and verified the model and tokenizer hashes.

以下命令用于 Linux x86_64、glibc >= 2.28 和 CPython 3.12。Worker 使用独立的 105 项哈希锁文件，不与早期 G2/SGLang 环境混用；旧锁中的 `packaging==26.3` 与 Worker 的 `pyvers` 约束冲突，新锁采用 `packaging==25.0`。实测依赖安装、单独下载并哈希核对的模型，以及真实 Worker。

```bash
# Run from a checkout containing this document. Choose new paths outside it.
export SCORING_DIR="$PWD"
export VERL_DIR=/absolute/path/to/new-verl-checkout
export MODEL_DIR=/absolute/path/to/Qwen2.5-0.5B-Instruct
export OUT_DIR=/absolute/path/to/new-output-directory
export WORKER_ENV=/absolute/path/to/new-worker-environment

uv venv --python 3.12 "$WORKER_ENV"
uv pip sync --python "$WORKER_ENV/bin/python" \
  --index-url https://pypi.org/simple --require-hashes --only-binary torch \
  "$SCORING_DIR/requirements-worker-linux-py312.lock"
uv pip check --python "$WORKER_ENV/bin/python"
. "$WORKER_ENV/bin/activate"

git init "$VERL_DIR"
git -C "$VERL_DIR" remote add origin https://github.com/verl-project/verl.git
git -C "$VERL_DIR" fetch --depth=1 origin 87cd055e872f9fdd7b976fcd797a0934b6dba265
git -C "$VERL_DIR" checkout --detach FETCH_HEAD
git -C "$VERL_DIR" apply --check "$SCORING_DIR/integrations/verl_6920/memory-bounded-teacher.patch"
git -C "$VERL_DIR" apply "$SCORING_DIR/integrations/verl_6920/memory-bounded-teacher.patch"
python -m integrations.verl_6920.verify_patch --verl-dir "$VERL_DIR"

hf download Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-dir "$MODEL_DIR"
mkdir -p "$OUT_DIR"
export PYTHONPATH="$SCORING_DIR:$VERL_DIR"
export VERL_LOGGING_LEVEL=INFO
export VERL_6920_SOURCE="$VERL_DIR"
export VERL_MASK_REGRESSION_SOURCE="$VERL_DIR"
export VERL_MASK_REGRESSION_DEPS=''
python -m pytest -q -ra
```

The probe verifies the model and tokenizer SHA-256 hashes, pinned verl commit, actual Worker import location, input mask, offsets, hooks and saved outputs. The full test command exercises GPU cases only when CUDA is available. Before each Worker process check `nvidia-smi` and host memory; our gate required at least 4 GiB host available, at least 4096 MiB free GPU memory for short/Chunked runs or 5120 MiB for the long Dense run, and GPU utilization below 75%. The individual commands do not enforce that gate. Preserve OOMs, sampler failures and raw output.

探针会核对模型和 tokenizer 哈希、固定 verl commit、实际 Worker 来源、输入 mask、offset、hooks 和保存的输出。完整测试仅在 CUDA 可用时执行 GPU 用例。每个 Worker 进程前须检查资源：本次门槛为主机可用内存至少 4 GiB、短序列或 Chunked GPU 空闲至少 4096 MiB、长 Dense 至少 5120 MiB、GPU 利用率低于 75%；下列命令本身不执行这一检查。

## Clean-environment Worker smoke / 干净环境 Worker 检查

```bash
python scripts/long_worker_probe.py --verl-dir "$VERL_DIR" --model-dir "$MODEL_DIR" \
  --out "$OUT_DIR/dense-1024.json" --mode dense --lengths 1024 --k 32 \
  --chunk-rows 256 --allocator-fraction 0.50 --warmup 1 --repeats 2 --sample-device-ms 0
python scripts/long_worker_probe.py --verl-dir "$VERL_DIR" --model-dir "$MODEL_DIR" \
  --out "$OUT_DIR/chunked-1024.json" --mode chunked --lengths 1024 --k 32 \
  --chunk-rows 256 --allocator-fraction 0.50 --warmup 1 --repeats 2 --sample-device-ms 0
python scripts/compare_capacity_outputs.py --dense "$OUT_DIR/dense-1024.call01.pt" \
  --candidate "$OUT_DIR/chunked-1024.call01.pt" --out "$OUT_DIR/compare-1024.json"
```

`--sample-device-ms 0` records start/end device snapshots, not a device peak. The raw log and complete CPU output for every call are retained. 该设置仅记录进程开始和结束的设备快照，不能表示设备峰值。

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
    --chunk-rows 256 --warmup 1 --repeats 3 || break
done
python scripts/compare_capacity_outputs.py --dense "$OUT_DIR/dense-a0.call01.pt" \
  --candidate "$OUT_DIR/dense-a1.call01.pt" --out "$OUT_DIR/natural-aa.json"
python scripts/compare_capacity_outputs.py --dense "$OUT_DIR/chunk-b0.call01.pt" \
  --candidate "$OUT_DIR/chunk-b1.call01.pt" --out "$OUT_DIR/natural-bb.json"
python scripts/compare_capacity_outputs.py --dense "$OUT_DIR/dense-a0.call01.pt" \
  --candidate "$OUT_DIR/chunk-b0.call01.pt" --out "$OUT_DIR/natural-ab.json"
```

Each command launches a fresh process. Omitting `--allocator-fraction` does not call the setter; confirm `allocator_fraction_installed=false` and the recorded actual fraction in each JSON. Compare A/A and B/B artifacts too. The 100 ms `nvidia-smi` samples include other processes and can miss peaks. Clock disagreement prevents a speedup claim; these 1024-row runs do not establish unrestricted capacity.

每条命令是独立进程。省略 fraction 后须在 JSON 核对实际值及 `allocator_fraction_installed=false`；同时比较 A/A、B/B 工件。设备采样包含其他进程且可能漏峰；时钟分歧未解，1024 行结果不证明加速或自然容量上限。

## Reproduce the displayed matrix / 复现首页矩阵

These serial commands use the same lengths, ABBA process order, warmup/repeat counts, fraction and chunk-size points as the displayed matrix. Before each process, the function checks the same GPU-free-memory, utilization and host-memory gate described above. A process failure leaves its JSON/log for inspection and stops the matrix. Set a new `OUT_DIR`; never overwrite earlier evidence.

以下串行命令复现首页矩阵的长度、ABBA 进程顺序、预热/计数次数、fraction 与分块大小。函数在每个进程前自动检查上述 GPU 空闲显存、利用率和主机可用内存门槛。矩阵中的进程失败时停止并保留 JSON/日志。请使用新的 `OUT_DIR`。

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
run_worker() {
  local name="$1" n="$2" mode="$3" chunk="$4" warm="$5" repeat="$6"
  local gpu free util host_kib threshold=4096
  gpu="$(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)" || return 1
  IFS=, read -r free util <<< "$gpu"
  free="${free//[[:space:]]/}"; util="${util//[[:space:]]/}"
  host_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  if [ "$mode" = dense ] && [ "$n" -ge 8192 ]; then threshold=5120; fi
  if ! [[ "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ && "$host_kib" =~ ^[0-9]+$ ]] ||
     [ "$free" -lt "$threshold" ] || [ "$util" -ge 75 ] || [ "$host_kib" -lt 4194304 ]; then
    printf 'NOT_RUN: resource gate (GPU free=%s MiB, utilization=%s%%, host available=%s KiB)\n' "$free" "$util" "$host_kib"
    return 1
  fi
  python scripts/long_worker_probe.py --verl-dir "$VERL_DIR" --model-dir "$MODEL_DIR" \
    --out "$OUT_DIR/$name.json" --mode "$mode" --lengths "$n" --k 32 \
    --chunk-rows "$chunk" --allocator-fraction 0.50 --warmup "$warm" \
    --repeats "$repeat" --sample-device-ms 0 >"$OUT_DIR/$name.log" 2>&1
}
matrix_ok=1
for n in 1024 4096 8192; do
  run_worker "n${n}-A0" "$n" dense 256 1 5 || { matrix_ok=0; break; }
  run_worker "n${n}-B0" "$n" chunked 256 1 5 || { matrix_ok=0; break; }
  run_worker "n${n}-B1" "$n" chunked 256 1 5 || { matrix_ok=0; break; }
  run_worker "n${n}-A1" "$n" dense 256 1 5 || { matrix_ok=0; break; }
done
if [ "$matrix_ok" -eq 1 ]; then
  for n in 4096 8192; do
    run_worker "n${n}-C128" "$n" chunked 128 1 5 || { matrix_ok=0; break; }
    run_worker "n${n}-C512" "$n" chunked 512 1 5 || { matrix_ok=0; break; }
  done
fi
if [ "$matrix_ok" -eq 1 ]; then
  for n in 1024 4096 8192; do
    for candidate in "$OUT_DIR"/n${n}-*.call*.pt; do
      name="$(basename "$candidate" .pt)"
      python scripts/compare_capacity_outputs.py --dense "$OUT_DIR/n${n}-A0.call01.pt" \
        --candidate "$candidate" --out "$OUT_DIR/compare-${name}.json" || { matrix_ok=0; break 2; }
    done
  done
fi
if [ "$matrix_ok" -ne 1 ]; then printf '%s\n' 'Matrix failed; inspect the saved JSON and logs.'; fi
```

Capacity points below are separate cold processes at the same fraction. Dense 9216 OOM is an observed result, not an expected success; inspect its JSON and any other nonzero exit before continuing. The largest successful point is a tested point, not a proven maximum. These commands do not provide a Dense reference at 32768.

以下容量点是同一 fraction 下的独立冷进程。Dense 9216 的 OOM 是已观察结果，非成功预期；任何非零退出都应读取 JSON 后再继续。最大成功点是已测点，不是数学最大值；32768 没有同长度 Dense 参考。

```bash
run_worker cap-dense-8704 8704 dense 256 0 1
run_worker cap-dense-9216 9216 dense 256 0 1  # inspect JSON if this OOMs
run_worker cap-chunk-16384 16384 chunked 256 0 1
run_worker cap-chunk-32768 32768 chunked 256 0 1
```

The public [evidence manifest](../results/worker-evidence-manifest.json) binds the aggregate table to SHA-256 hashes of the archived raw results. The raw evidence archive remains in the project's private Google Drive workspace. 公开证据清单用 SHA-256 绑定汇总结果和原始记录；原始证据包保存在项目私有 Google Drive 工作区。

## Tests and report / 测试与报告

```bash
VERL_6920_SOURCE="$VERL_DIR" VERL_MASK_REGRESSION_SOURCE="$VERL_DIR" \
  VERL_MASK_REGRESSION_DEPS='' python -m pytest -q
bash scripts/build_experimental_report.sh
```

The two actual upstream metric parity tests explicitly skip without the fixed checkout and dependencies. The report builder needs XeLaTeX, BibTeX, Latin Modern and a full Chinese font; alternatively set `LATEX_ENGINE=tectonic` with Tectonic available. It writes only new experimental PDFs. New report text awaits Human review.

上游指标对照缺少固定 checkout 时会明确跳过。报告构建需要 XeLaTeX、BibTeX、Latin Modern 和完整中文字体；也可使用 Tectonic。构建只写新实验报告，新增正文仍待 Human 审核。

## Regenerate the figures / 重建图表

The published figures are derived from `results/worker-benchmark-summary.json`. Rebuild them with Matplotlib 3.11.1 in a separate plotting environment; font rendering can differ by platform. 图表基于仓库内汇总数据；使用独立绘图环境重建，字体渲染可能因平台不同而变化。

```bash
export PLOT_ENV=/absolute/path/to/a/new/plot-environment
uv venv --python 3.12 "$PLOT_ENV"
uv pip install --python "$PLOT_ENV/bin/python" matplotlib==3.11.1
"$PLOT_ENV/bin/python" scripts/plot_worker_results.py
```
