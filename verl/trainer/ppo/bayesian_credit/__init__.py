# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Bayesian Credit Assignment from a Self-Distilled Teacher.

每个文件 = 一个独立的 advantage estimator，通过 @register_adv_est 注册。
core_algos.py 不被触碰。新增 estimator 都通过 import 副作用完成注册。

当前实现：
    - rlsd.py          : baseline，公式见 arXiv:2604.03128v2
    - prior_shift.py   : ours, A_t = A_seq · KL(P_T(·|y_≤t) ‖ P_T(·|y_<t)) / mean_t  （Tier 1 首发）
    - posterior_shift.py: ours, A_t ∝ log P_T(y_t|x,y,r) − log P_T(y_t|x,y_<t)（TODO）
"""

from . import rlsd          # noqa: F401  (registers @register_adv_est("rlsd"))
from . import prior_shift   # noqa: F401  (registers @register_adv_est("prior_shift"))
