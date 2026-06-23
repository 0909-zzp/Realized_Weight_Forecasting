# 长期记忆

## 项目约定

- 图形Lasso项目开发前必须先读 README.md 了解架构状态、参数值、模块边界和规范约束。
- 代码变更涉及模块增删、参数调整、架构修改时，必须先更新 README.md 对应章节，再实施代码变更。
- 参数仅通过 `code/共享模块.py` 集中管理。
- 新模块需在 README 的项目架构图、模块说明、全流程操作中同步登记。

## 文档产出

- 2026-06-21：生成了 `结构化思维笔记.md`，基于图形Lasso项目完整分析了结构化思维从概念到工程实践的映射，涵盖七维度（需求拆解、数据流、特征工程、实验管理、评估体系、高内聚低耦合、五步法示例）。

## 技术决策

- 2026-06-22：阶段二 Windows multiprocessing 调试 — ProcessPoolExecutor 在 spawn 模式下卡死。解决方案：`纯权重计算.py`（零项目依赖 Pool + freeze_support）。Ridge 退避链精简化 `[1e-4, 1e-3, 1e-2]`。云端腾讯云 8vCPU 竞价实例 Pool 不兼容，转向本地/AutoDL。
- 2026-06-22：规则文件分户：`vibe-coding.mdc`（用户级全局）和 `图形Lasso项目规范.mdc`（项目级）。`.gitignore` 从排除 `.codebuddy/` 改为只排除 `.codebuddy/memory/`，规则文件可提交 Git。
- 2026-06-22：.npy 快读转换（`_convert_npy.py`）— 2436天 .RData → .npy，I/O 加速 5-10×。R 备份脚本 `R.R` 超参与 Python 对齐。
