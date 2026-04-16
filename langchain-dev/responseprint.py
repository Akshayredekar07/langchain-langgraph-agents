import json

def pretty_agent_response(response: dict):
    msgs = response.get("messages", [])
    print(f"Total messages: {len(msgs)}\n")

    for i, m in enumerate(msgs, 1):
        role = type(m).__name__
        print(f"[{i}] {role}")

        # Main text
        content = getattr(m, "content", "")
        if content:
            print(f"  content: {content}")

        # Tool calls (if any)
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            print("  tool_calls:")
            for tc in tool_calls:
                print(f"    - {tc.get('name')}({tc.get('args')}) id={tc.get('id')}")

        # Tool message metadata
        tool_name = getattr(m, "name", None)
        tool_call_id = getattr(m, "tool_call_id", None)
        if tool_name or tool_call_id:
            print(f"  tool: name={tool_name}, tool_call_id={tool_call_id}")

        # Token usage (if available)
        usage = getattr(m, "usage_metadata", None) or {}
        if usage:
            print(
                f"  tokens: in={usage.get('input_tokens')}, "
                f"out={usage.get('output_tokens')}, total={usage.get('total_tokens')}"
            )

        print()  # blank line between messages



def inspect_obj(obj, name="obj", depth=0, max_depth=4):
    indent = "  " * depth
    print(f"{indent}{name}: {type(obj)}")

    if depth >= max_depth:
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            inspect_obj(v, f"{name}[{k!r}]", depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            inspect_obj(v, f"{name}[{i}]", depth + 1, max_depth)
    else:
        attrs = [a for a in dir(obj) if not a.startswith("_")]
        if attrs:
            print(f"{indent}  attrs: {attrs}")




# last_msg = response["messages"][-1]
# print(type(last_msg))
# print(last_msg)               # readable repr
# print(last_msg.content)       # final text
# print(last_msg.__dict__)      # works if this is an object with __dict__
# # or, for pydantic-like objects:
# # print(last_msg.model_dump())
