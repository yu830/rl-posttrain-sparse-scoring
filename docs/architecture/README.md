# Memory-bounded Teacher Scoring diagrams

The **system overview** shows input contract alignment, the readout policy, the real single-GPU FSDP2 Worker, the dense baseline and the tile-based output-head readout. The **readout lifecycle** diagram focuses on vocabulary-logit allocation and disposal.

English and Chinese diagrams use matching layouts. The SVGs are editable vector diagrams; PNGs provide a raster alternative. The dashed control line is an execution-policy input. The separate Worker-to-Student validation is not a completed Ray Trainer.

The 81.05% reduction refers to PyTorch **peak allocated memory during complete Worker calls** at the recorded single-GPU BF16 workload, not to all-device VRAM or RL throughput. Prior diagrams are preserved as historical snapshots.
