<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# aMUSEd

<TODO>

aMUSEd is a masked transformer model trained on LAION-5B and LAION-5B aesthetic subsets. There are two checkpoints, one for 256x256 and one for 512x512 image resolution.

See the model card <TODO link> for more information on training procedure and data used.

## AmusedPipeline

[[autodoc]] AmusedPipeline
	- __call__
	- all
	- enable_fused_rms_norm
	- disable_fused_rms_norm
	- enable_xformers_memory_efficient_attention
	- disable_xformers_memory_efficient_attention