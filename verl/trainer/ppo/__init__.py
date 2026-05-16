# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Trigger registration of bayesian_credit advantage estimators (rlsd / prior_shift / ...).
# Must run before any AdvantageEstimator dispatch.
from . import bayesian_credit  # noqa: E402, F401
from . import dpo_tgs           # noqa: E402, F401  (registers @register_adv_est("dpo_teacher_guided"))
