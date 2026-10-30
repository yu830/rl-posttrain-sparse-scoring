# verl #6920 的显存受控 Teacher 接入

补丁仅适用于 `verl-project/verl` 的
`87cd055e872f9fdd7b976fcd797a0934b6dba265`，连接真实 engine 构造与
`TeacherFSDPWorker`。安装点位于 FSDP 包装前；请求通过原
`self.engine.module(...)` 进入外层 FSDP forward，保留原参数注册、共享
embedding/head、eval/no-grad/offload、nested offsets、收集 rank 与 CPU 返回。
输出 TensorDict 仍只有 `teacher_on_student_logp`、`teacher_topk_ids`。
输入候选 ID 已对应因果 logits 行，本接入不另做 shift 或温度变换。

[英文复现说明](README.md)给出应用补丁和离线验证命令。
使用前设置 `SCORING_DIR`、`VERL_DIR`，将两者加入 `PYTHONPATH`，并设
`VERL_LOGGING_LEVEL=INFO` 观察路径和分块决策。源码清单记录固定 Git blob；
补丁保留上游 CRLF，换行感知检查使用
`git -c core.whitespace=cr-at-eol diff --check`。

默认 `teacher_memory_bounded=false` 保持原 worker 路径。现有
`distillation.teacher_fsdp_config` 新增三个显式选项：

- `teacher_memory_bounded`：启用本接入，默认 false。
- `teacher_lm_head_chunk_rows`：单次 LM Head 行数上限，默认 256。
- `teacher_head_memory_budget_mib`：可选的 Head 分配估计预算，默认 null。

现有 `teacher_models.<名称>` 增加与 `HFModelConfig` 同名、同默认值的
`use_remove_padding=true` 和 `override_config={}`，原 trainer 构造过程明确传入。
例如设置 `use_remove_padding=false`、
`override_config.attn_implementation=eager`，可走多条变长序列的 padded
Teacher 路径；候选按 engine 的真实 attention mask 映射，输出按原 offsets
恢复。单序列 rmpad 可用；多序列 rmpad 在实际 varlen attention 验证前明确拒绝。
省略选项时保留上游默认行为与 attention 默认值。

预算可在 forward 前缩小行块；不足以容纳紧凑输出和一行估计开销时拒绝。
这是 Head 分配估计，包含实际投影的 padding 行；它不对完整 GPU 显存、FSDP
参数、backbone 激活、CUDA workspace、碎片和其他进程作硬保证。
原 `distillation.teacher_chunk_size` 仍控制原稠密 gather 的归约分块。

未设置预算且所选行块覆盖全部实际模型行时，自动选择
`dense_full_row_fallback`：完整原生 dense forward 后在 FSDP 外做原 gather。
显式预算存在时禁止这种自动 dense 回退。日志记录执行路径、请求/实际行块、
估计字节、预算与 sizing reason。异常与 OOM 原样传播，不自动重试外层 FSDP。
LoRA、量化/QAT、替代或 fused forward、序列并行明确拒绝。

离线验证器只在临时副本应用补丁和检查 Python 语法；CPU 测试验证真实 nested
布局、预算/回退策略、原 trainer 构造表达式，以及禁用分支与原输出恢复代码的
AST 一致性。这些结果不能冒充真实 worker/GPU 运行。实际 worker 还需要完整
verl 运行依赖；应构造真实 `TeacherFSDPWorker`、调用 `reset()` 和
`compute_logprobs_at_ids()`，再单独报告运行证据。本地直接调用真实类不等于
验证了 Ray 调度、多 GPU 分片或完整 RL 训练。

应用补丁并设置上面的环境后，可以用以下命令复现直接 Worker 与上游损失函数检查。请使用新的输出文件名；这些命令不执行 Ray 远程调度。

```bash
python scripts/verl_teacher_worker_smoke.py --verl-dir "$VERL_DIR" --model-dir "$MODEL_DIR" \
  --mode dense-fallback --warmup 1 --repeats 2 --out "$OUT_DIR/worker-dense.json"
python scripts/verl_teacher_worker_smoke.py --verl-dir "$VERL_DIR" --model-dir "$MODEL_DIR" \
  --mode chunked --chunk-rows 16 --head-memory-budget-mib 64 --warmup 1 --repeats 2 \
  --out "$OUT_DIR/worker-chunked.json"
python scripts/verify_verl_consumer.py --verl-dir "$VERL_DIR" --input "$OUT_DIR/consumer.json" \
  --device cuda --out "$OUT_DIR/verl-consumer.json"
```
重复 Worker 检查时，先在新进程运行 dense-fallback，再将其 JSON 作为 --expected 传给新的 chunked 进程。脚本逐次核对已完成调用的紧凑输出哈希、dtype、形状和 offsets；不一致保留为失败。
