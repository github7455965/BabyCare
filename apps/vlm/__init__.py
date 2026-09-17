"""apps.vlm app。

Step 4：仅含 LlamaManager（llama-server 进程管理）。
后续步骤：VLMPromptConfig / VLMCheckState / DismissedState 模型（步骤5）、
PromptCursor / frame_selector / prompt_runner / vlm_worker（步骤6-8）、
HA 通知（步骤9）、HTTP API（步骤10）。
"""