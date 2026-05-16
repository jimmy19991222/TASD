# Research Documentation

> TCCA (Token-level Causal Credit Assignment) 项目研究文档。
> **入口顺序**：先 `paper_idea.md` 看方法 → `experiment_progress.md` 看现状 → 其他按需。

---

## 📚 Active 文档（4 篇）

| 文件 | 内容 | 何时看 |
|---|---|---|
| [paper_idea.md](paper_idea.md) | **TCCA 论文思路** — 创新点 / OPD 演化 / 完整 pipeline (Step 1-9) / 9 个一致性 check 点 | **先看这个** —— 论文写什么 |
| [experiment_progress.md](experiment_progress.md) | **实验进展报告** — 现状/排名/baseline matrix/实验时间线/诊断 insight/待跑/工程 TODO/Commit 链 | 看实验数据 / 决定下一步 |
| [design_history.md](design_history.md) | **TGDI 历史设计** — 从 Prior-Shift Tier 1 演化到 TCCA 的设计文档（已被 paper_idea 取代） | 写论文 Related Work 时回溯 |
| [submission_guide.md](submission_guide.md) | **任务提交指南** — 算法对比实验 Nebula sweep 提交流程 | 操作时查 |

## 🗄️ Archive（已弃用方案）

存档于 [archive/](archive/) 目录：

| 文件 | 弃用原因 |
|---|---|
| [archive/self_teacher_advantage.md](archive/self_teacher_advantage.md) | 旧 Self-Teacher Advantage 方法，已被 Prior-Shift / TCCA 系列取代 |
| [archive/on_policy_sd_dpo.md](archive/on_policy_sd_dpo.md) | 旧 token-level on-policy DPO 路线，已弃用 |
| [archive/proposal_review.md](archive/proposal_review.md) | 早期 proposal 审阅，思路已演化 |
| [archive/quick_submit_guide.md](archive/quick_submit_guide.md) | 旧版快速提交指南（已合并到 submission_guide.md） |
| [archive/experiment_submission_guide.md](archive/experiment_submission_guide.md) | 旧版完整提交指南（已合并） |

---

## 🎯 论文核心 claim (one-liner)

> **TCCA**：把"token 的 credit 应该多少"用 teacher 真实改写 + reward 复算的**因果反事实**方式回答——
> 不是 "teacher 在哪里惊讶"（Prior-Shift），不是 "teacher-student 哪里不合"（RLSD），
> 而是 **"如果在 t 处听 teacher，最终 reward 真涨了吗？"**。

## 🧭 当前状态（速览）

- **当前分支**：`teacher-guided-intervention` @ `8bcca90`
- **SwanLab projects**：`TGDI-Tier3` / `TGDI-local` / `PriorShift-Tier1` / `Baselines_v2` / `Baselines_v3`
- **凭证**：项目根 `.env`（已 gitignore）

详见 [experiment_progress.md](experiment_progress.md) 顶部一屏现状。
