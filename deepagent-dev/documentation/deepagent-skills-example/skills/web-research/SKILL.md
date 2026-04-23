---
name: web-research
description: >
  Use this skill for structured web research on any topic not covered by HuggingFace
  Hub tools directly — such as comparing frameworks, reading blog posts, finding
  GitHub repos, or researching recent AI news. Trigger on: "latest news",
  "compare X vs Y", "how does X work", "find a tutorial", "research this topic".
license: MIT
metadata:
  author: langchain-ai (adapted)
  version: "1.0"
  source: "https://github.com/langchain-ai/deepagents/tree/main/libs/cli/examples/skills"
---

# Web Research Skill

## Overview
A structured research workflow. Always plan before searching, delegate parallel
subtasks to subagents when the scope is large, and synthesize into a clean output.

---

## When to Use This Skill vs HF MCP Tools

| Task | Use |
|---|---|
| Find a model/dataset/paper on HuggingFace | `hf-hub-search` skill |
| Compare two frameworks (e.g., LangChain vs LlamaIndex) | This skill |
| Read a blog post / GitHub README | This skill |
| Find recent AI news or benchmarks | This skill |
| Get HF documentation | `hf-hub-search` skill (documentation_search) |

---

## Research Workflow

### Step 1: Plan
Write todos before searching. Break the research into parallel subtasks:
```
✅ Subtask 1: Find overview / definition
✅ Subtask 2: Find recent benchmarks / papers
✅ Subtask 3: Find code examples / GitHub repos
✅ Subtask 4: Synthesize findings
```

### Step 2: Search (use write_file to store intermediate results)
- Search each subtask separately
- Write raw findings to `/research/<topic>-raw.md` immediately
- Do NOT keep all results in context — offload to files

### Step 3: Synthesize
- Read all raw files back
- Write final synthesis to `/research/<topic>-summary.md`
- Format: Executive summary → Key findings → Code examples → References

### Step 4: Present
- Return the synthesized summary to the user
- Mention the files written if user wants the full detail

---

## Output Template

```markdown
## Research: [Topic]

### Summary (2-3 sentences)
...

### Key Findings
- Finding 1
- Finding 2
- Finding 3

### Code / Implementation
\```python
# ...
\```

### Sources
- [Title](url)
- [Title](url)
```

---

## Notes
- For large research tasks (5+ subtopics), delegate to a `researcher` subagent
  using the `task` tool to avoid blocking the main conversation.
- Always write intermediate results to files — never accumulate in context.
- Cite sources. Do not fabricate citations.