from IPython import get_ipython
from plugins.location_tools.repo_ops.repo_ops import (
    search_code_snippets,
    get_entity_contents,
    explore_graph_structure,
    explore_tree_structure,
    search_commit,
    view_summary,
    search_summary,
)

from IPython.utils.capture import capture_output
from IPython.terminal.interactiveshell import TerminalInteractiveShell

# Tools that can be excluded via --exclude_tools
_TOOL_REGISTRY = {
    'search_code_snippets': search_code_snippets,
    'get_entity_contents': get_entity_contents,
    'explore_graph_structure': explore_graph_structure,
    'explore_tree_structure': explore_tree_structure,
    'search_commit': search_commit,
    'view_summary': view_summary,
    'search_summary': search_summary,
}

# Module-level set of excluded tools (set by caller before execute_ipython)
_excluded_tools: set = set()


def set_excluded_tools(exclude: list[str] | None):
    global _excluded_tools
    _excluded_tools = set(exclude or [])


def execute_ipython(code_to_execute):
    # Manually initialize an IPython shell
    ipython_shell = TerminalInteractiveShell.instance()

    # Inject only non-excluded tools into the IPython environment
    for name, func in _TOOL_REGISTRY.items():
        if name not in _excluded_tools:
            ipython_shell.user_ns[name] = func
        else:
            # Remove from namespace if previously injected
            ipython_shell.user_ns.pop(name, None)

    # Execute the code in the IPython shell
    with capture_output() as captured:
        ipython_shell.run_cell(code_to_execute)

    output = ''
    if captured.stdout:
        output += captured.stdout
    if captured.stderr:
        output += captured.stderr

    return output if output else None
