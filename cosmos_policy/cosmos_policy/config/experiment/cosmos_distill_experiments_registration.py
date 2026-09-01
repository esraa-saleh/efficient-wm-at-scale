# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Registration shim, nothing else.

The cosmos_distill_experiments job's Hydra experiment definition lives in
scripts/cosmos_distill_experiments/experiment.py, alongside that job's launcher and README (so the
whole recipe is in one place). It still needs to be *imported* from somewhere under this exact
config/experiment/ package, because config_v2.py's
import_all_modules_from_package("cosmos_policy.config.experiment", ...) only
pkgutil.iter_modules()s files physically inside this directory -- it will not discover a module
living under scripts/. Importing it here (for its module-level ConfigStore.store() side effects) is
the whole point of this file.
"""

from cosmos_policy.scripts.cosmos_distill_experiments import experiment  # noqa: F401
from cosmos_policy.scripts.cosmos_distill_experiments.kd import net_experiments  # noqa: F401
