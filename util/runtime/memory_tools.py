from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

# ── Commit History Tools (Episodic Memory) ───────────────────────────────────

_SEARCH_COMMIT_DESCRIPTION = """Searches the repository's commit history by keyword to find relevant past changes.
Returns matching commits with their SHA, message, date, changed files, linked issues, and a compact diff for the top results.
Use this to understand how specific features, bugs, or components were modified over time.

** Example Usage:
# Find commits related to authentication fixes
search_commit(query="fix authentication")

# Find commits that modified the parser
search_commit(query="refactor parser", top_k=3)
"""

SearchCommitTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='search_commit',
        description=_SEARCH_COMMIT_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Keywords to search in commit messages.',
                },
                'top_k': {
                    'type': 'integer',
                    'description': 'Maximum number of commits to return. Default 5.',
                    'default': 5,
                },
            },
            'required': ['query'],
        },
    ),
)


# ── File Summary Tools (Semantic Memory) ─────────────────────────────────────

_VIEW_SUMMARY_DESCRIPTION = """Views the LLM-generated natural language summary of a specific file's functionality.
Use this to quickly understand what a file does, its key classes/functions, and dependencies without reading the full code.
The summary describes the file's purpose and role in the codebase in 2-4 sentences.

** Example Usage:
# View summary of a specific file
view_summary(file_path="src/auth/login.py")
"""

ViewSummaryTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='view_summary',
        description=_VIEW_SUMMARY_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'Path to the file to summarize.',
                },
            },
            'required': ['file_path'],
        },
    ),
)

_SEARCH_SUMMARY_DESCRIPTION = """Searches file summaries by keyword to find files related to a concept or component.
Each summary describes the file's purpose, key classes, functions, and dependencies in natural language.
Use this to discover which files are relevant to a particular topic before diving into the code.

** Example Usage:
# Find files related to database connections
search_summary(query="database connection")

# Find files related to user authentication
search_summary(query="authentication login", top_k=5)
"""

SearchSummaryTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='search_summary',
        description=_SEARCH_SUMMARY_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Keywords to search in file summaries.',
                },
                'top_k': {
                    'type': 'integer',
                    'description': 'Maximum number of files to return. Default 10.',
                    'default': 10,
                },
            },
            'required': ['query'],
        },
    ),
)
