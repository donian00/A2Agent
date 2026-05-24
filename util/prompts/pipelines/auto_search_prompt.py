TASK_INSTRUECTION="""
Given the following GitHub problem description, your objective is to localize the **exact** code locations that need modification at THREE granularities:
  1. **File** — the source file containing the bug or new behavior.
  2. **Module** — the class (or top-level module-scope function) inside that file.
  3. **Function/Entity** — the specific method/function that must be edited.

You MUST report all three granularities in your final answer. Reporting only file paths is INSUFFICIENT — module and function predictions are evaluated independently and contribute to your score.

Follow these steps to localize the issue:
## Step 1: Categorize and Extract Key Problem Information
 - Classify the problem statement into the following categories:
    Problem description, error trace, code to reproduce the bug, and additional context.
 - Identify candidate modules in the '{package_name}' package mentioned in each category.
 - Use extracted keywords and line numbers to search for relevant code references for additional context.

## Step 2: Locate Referenced Modules and Functions
- Accurately determine specific files → modules → functions
    - Explore the repo to familiarize yourself with its structure.
    - Analyze the described execution flow to identify specific classes / methods being referenced.
    - Inspect class hierarchies and method signatures to pin down the exact function.
- Pay special attention to distinguishing between modules with similar names using context and described execution flow.
- Output Format for each location:
    - File only:                'file_path'
    - Module (class):           'file_path:ClassName'
    - Function (method/entity): 'file_path:ClassName.method_name' or 'file_path:top_level_function'
    - Example: for method `calculate_sum` of class `MathUtils` in `src/helpers/math_helpers.py`:
        - file:    src/helpers/math_helpers.py
        - module:  src/helpers/math_helpers.py:MathUtils
        - entity:  src/helpers/math_helpers.py:MathUtils.calculate_sum

## Step 3: Analyze and Reproducing the Problem
- Clarify the Purpose of the Issue
    - If expanding capabilities: Identify where and how to incorporate new behavior, fields, or modules.
    - If addressing unexpected behavior: Focus on localizing modules containing potential bugs.
- Reconstruct the execution flow
    - Identify main entry points triggering the issue.
    - Trace function calls, class interactions, and sequences of events.
    - Identify potential breakpoints causing the issue.
    Important: Keep the reconstructed flow focused on the problem, avoiding irrelevant details.

## Step 4: Locate Areas for Modification (file → module → function)
- For every candidate file, ALSO determine WHICH class and WHICH method inside it require changes. Do not stop at the file level.
- Use `get_entity_contents` to read the actual class/method body and verify the function-level location.
- Consider upstream and downstream dependencies that may affect or be affected by the issue.
- If applicable, identify where to introduce new fields, functions, or variables.
- Think Thoroughly: List multiple potential solutions and consider edge cases that could impact the resolution.

**IMPORTANT — Commit what you observed.** If during exploration you read a file, class, or function that turns out to be the actual bug location, you MUST include it in your final answer. Failing to commit observed gold locations is the most common failure mode. Before you finalize, re-scan your trajectory: every file/class/function you inspected that is plausibly responsible MUST appear in the final answer at the appropriate granularity.

## Available Tools and When to Use Them
You have 8 tools. Choose the most appropriate tool for each step — do NOT repeatedly call the same tool.

### Search Tools (discover and find)
| Tool | When to Use |
|------|-------------|
| `explore_tree_structure` | **First step**: understand repo layout and module hierarchy. |
| `search_summary` | **Semantic file discovery**: search all file summaries by concept or description (e.g., "validation logic", "serialization"). Returns top matching files with their summaries. **Use this when** the issue is vague, uses domain-level language, or you don't know exact identifiers — it bridges the gap between the problem description and actual file locations. |
| `search_code_snippets` | **Find code by keyword**: locate specific functions, classes, error messages, or variable names across the codebase. Best when you have exact identifiers to search for. |
| `search_commit` | **Find relevant commits**: search past commits by keyword. Returns commit SHAs, messages, dates, and changed files. **Use this to** find co-changed files that reveal hidden dependencies, discover past fixes for similar problems, and identify modification patterns. |

### View Tools (inspect in detail)
| Tool | When to Use |
|------|-------------|
| `get_entity_contents` | **Read actual code**: once you know the file/class/function path, read its implementation to understand the logic and confirm it is the right location. |
| `view_summary` | **Understand a file's purpose**: view the LLM-generated summary of a specific file to quickly understand what it does, its key classes/functions, and dependencies without reading the full code. |

### Finish
| Tool | When to Use |
|------|-------------|
| `finish` | **Submit final answer**: call this after you have localized all relevant locations. |

### Recommended workflow:
1. `explore_tree_structure` → get repo overview
2. `search_summary` → find relevant files by concept
3. `view_summary` → understand key files' purposes
4. `search_commit` → find related change history and co-changed files
5. `search_code_snippets` → locate specific code elements by keyword
6. `get_entity_contents` → read actual code of key functions/classes
7. `finish` → output your final answer

**IMPORTANT**: Use at least 4 different tool types before calling `finish`. Each tool provides a different perspective — `search_code_snippets` finds exact matches, `search_summary` and `view_summary` provide semantic understanding, `get_entity_contents` lets you verify code logic, and `search_commit` reveals historical context and co-changed files. Do NOT over-rely on any single tool — if you find yourself calling the same tool repeatedly, switch to a different one.

## Output Format for Final Results:
Your final output should list the locations requiring modification, wrapped with triple backticks ```
For EACH location you MUST report all available granularities:
  - the file path
  - the class name (module-level), if the change is inside a class
  - the function name (entity-level): either `ClassName.method` or top-level `function_name`
  - line numbers when known
Order locations by importance. Aim for ~5 files. Within each file, list every relevant module/function — not just one.

Module-level and function-level predictions are scored independently from file-level. Reporting only the file path will lose ~2/3 of the available credit. Always include `class:` and `function:` when the answer lies inside a class or specific function.

### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
function: MyClass1.my_function1

full_path2/file2.py
line: 76
class: MyClass2
function: MyClass2.my_function2

full_path3/file3.py
line: 24
line: 156
function: my_function3
```

Return just the location(s)

Note: Your thinking should be thorough and so it's fine if it's very long.
"""

FAKE_USER_MSG_FOR_LOC = (
    'Verify if the found locations contain all the necessary information to address the issue, and check for any relevant references in other parts of the codebase that may not have appeared in the search results. '
    'If not, continue searching for additional locations related to the issue.\n'
    'Verify that you have carefully analyzed the impact of the found locations on the repository, especially their dependencies. '
    'If you think you have solved the task, please send your final answer (including the former answer and reranking) to user through message and then call `finish` to finish.\n'
    'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
)