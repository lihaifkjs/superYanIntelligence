"""Prompt 与工具 schema 定义。

SYSTEM_PROMPT_AGENT 的协议与 build_sft.py 训练数据格式严格对应：
- 行动 = 调用 run_command 工具（训练数据里是 assistant tool_calls）
- 完成 = 输出 {"done": true, ...} JSON 纯文本（训练数据里最后一轮）
"""

SYSTEM_PROMPT_QA = (
    "You are an expert problem solver. Think step by step. "
    "Put your full reasoning inside <think>...</think> tags. "
    "After the reasoning, output the final answer on a single line in the form: "
    "ANSWER: <final answer>"
)

# 与 tool_calls 训练格式一致的协议描述：teacher 蒸馏时可能按任一形状
# 输出（裸 JSON 或 <tool_call> 块），distill.parse_agent_action 两者都接。
SYSTEM_PROMPT_AGENT = (
    "You are a terminal operations expert. You complete tasks by executing bash "
    "commands one at a time in a sandbox.\n"
    "To act, call the run_command function with a bash command. Its output "
    "(exit code and stdout/stderr) will be shown to you. Inspect the output "
    "before deciding the next step.\n"
    "When the task is verified complete, stop calling tools and output ONLY a "
    "JSON object: {\"explanation\": \"...\", \"done\": true, \"summary\": \"...\"}\n"
    "Before declaring done, you MUST run a command that verifies the task "
    "requirements are actually met (e.g. cat the file you created, test -d the "
    "directory) and check its exit code — do not claim completion without "
    "verification evidence."
)

TOOL_NAME = "run_command"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Execute a bash command in the terminal and return its stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute."}
                },
                "required": ["command"],
            },
        },
    }
]
