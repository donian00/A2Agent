from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)


_FINISH_DESCRIPTION = """Finish the interaction and submit your final answer.
You MUST provide the answer parameter containing all located files, classes, functions, and line numbers in the specified format.

** Example:
finish(answer=\"\"\"```
src/auth/login.py
line: 45
class: LoginHandler
function: LoginHandler.authenticate

src/auth/utils.py
line: 12
function: validate_token
```\"\"\")
"""

FinishTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='finish',
        description=_FINISH_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'answer': {
                    'type': 'string',
                    'description': 'Your final answer with located files, classes, functions, and line numbers wrapped in triple backticks.',
                },
            },
            'required': ['answer'],
        },
    ),
)