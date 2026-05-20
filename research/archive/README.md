# research/archive/

历史快照。这些文件描述的是已经被替换的实验设计 / 失效的实现状态，仅作历史记录用。

**不要**按这些文件里的脚本路径或 config 字段名去跑——大多数 flag/路径已经被重命名或删除。

| 文件 | 时间 | 描述 |
|------|------|------|
| `geodesic_vce_ablation_v1_use_vce_era.md` | commit `5716fb3` (2026-05-19) | 原始 Geodesic-VCE 4 组实验设计文档。**严重缺陷**：当时所谓的 "VCE" 实验依赖 `algorithm.use_vce=True` 配置项，但该 flag 在 `compute_self_teacher_advantage` 里只 parse 不使用，4 组实验里 B/C 两组实际跑的是 sequence-level GRPO，不是 V_CE。Geodesic 在 `loss_mode=vanilla` 路径上也没生效。详见 commit `27d97fb` 的 use_vce 移除 + `teacher_qv` 引入，以及外部 `research/geodesic_vce_ablation.md` 当前版本。 |
