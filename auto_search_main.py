import argparse
import os
import json
import logging
import logging.handlers
import time
import toml
from queue import Empty
from typing import List
from tqdm import tqdm
from copy import deepcopy
from datasets import load_dataset
import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_messages_tokens(messages: list) -> int:
    """Estimate total token count for a list of chat messages.

    Counts content, tool_calls (function name + arguments), and per-message
    overhead.  Uses a 1.15x multiplier to compensate for the gap between
    cl100k_base and the actual Qwen tokenizer.
    """
    raw = 0
    for msg in messages:
        raw += 4  # message overhead
        content = msg.get("content", "")
        if content:
            raw += len(_ENCODING.encode(content))
        # Count tokens in tool_calls (assistant messages with function calls)
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw += len(_ENCODING.encode(fn.get("name", "")))
            raw += len(_ENCODING.encode(fn.get("arguments", "")))
    return int(raw * 1.15)  # safety multiplier for tokenizer mismatch


def get_context_budget(messages: list, max_context: int = 32768) -> int:
    """Return remaining token budget after accounting for messages and a response buffer."""
    used = count_messages_tokens(messages)
    return max(0, max_context - used - 2000)  # 2000 buffer for LLM response

def trim_context_proactively(
    messages: list,
    max_context: int = 32768,
    keep_recent: int = 6,
    trim_threshold: float = 0.75,
) -> list:
    """Remove old middle messages when token count exceeds threshold.

    Always keeps:
      - messages[0] (system prompt)
      - messages[1] (user instruction)
      - the most recent ``keep_recent`` messages

    Removed messages are replaced by a single placeholder summary.
    The function also ensures the kept tail does not start with a ``tool``
    role message (which would be an orphan tool response without a preceding
    assistant tool_call), shifting the boundary forward if needed.
    """
    total_tokens = count_messages_tokens(messages)
    if total_tokens <= max_context * trim_threshold:
        return messages  # nothing to trim

    head = messages[:2]  # system + user instruction
    tail = messages[-keep_recent:] if keep_recent > 0 else []

    # Ensure tail does not start with a tool-role message (orphan tool response)
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]

    # Nothing left to trim (conversation is too short)
    if len(head) + len(tail) >= len(messages):
        return messages

    removed_count = len(messages) - len(head) - len(tail)
    placeholder = {
        "role": "user",
        "content": (
            f"[{removed_count} earlier exploration messages were removed to stay "
            "within context limits. Continue your analysis with the tools available.]"
        ),
    }

    trimmed = head + [placeholder] + tail
    new_tokens = count_messages_tokens(trimmed)
    logging.info(
        f"Proactive trim: {total_tokens} -> {new_tokens} tokens "
        f"(removed {removed_count} messages, kept {keep_recent} recent)"
    )
    return trimmed


from util.runtime.execute_ipython import execute_ipython, set_excluded_tools
from util.runtime import function_calling
from util.actions.action_parser import ResponseParser
from util.actions.action import ActionType
from util.prompts.prompt import PromptManager
from util.prompts import general_prompt
from util.prompts.pipelines import auto_search_prompt as auto_search
from util.cost_analysis import calc_cost
from util.utils import *
from util.process_output import (
    parse_raw_loc_output,
    get_loc_results_from_raw_outputs,
    merge_sample_locations,
)
from plugins import LocationToolsRequirement
from plugins.location_tools.repo_ops.repo_ops import (
    set_current_issue,
    reset_current_issue,
)
import litellm
from litellm import Message as LiteLLMMessage
from openai import APITimeoutError


import signal
from time import sleep
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import torch.multiprocessing as mp
from util.runtime.fn_call_converter import (
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
    STOP_WORDS as NON_FNCALL_STOP_WORDS
)
# litellm.set_verbose=True
# os.environ['LITELLM_LOG'] = 'DEBUG

# filter to keep only the ids listed in data[used_list]
def filter_dataset(dataset, filter_column: str, used_list: str):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = toml.load(file)
            if used_list in data:
                selected_ids = data[used_list]
                logging.info(
                    f'Filtering {len(selected_ids)} tasks from "selected_ids"...'
                )
                def filter_function(example):
                    return example[filter_column] in selected_ids  # Replace 'id' with the actual field name in the dataset
                filtered_dataset = dataset.filter(filter_function)
                # subset = dataset[dataset[filter_column].isin(selected_ids)]
                logging.info(f'Retained {len(filtered_dataset)} tasks after filtering')
                return filtered_dataset
    return dataset

# build the per-task system/user instruction strings for a single instance
def get_task_instruction(instance: dict, include_pr=False, include_hint=False):
    instruction = auto_search.TASK_INSTRUECTION.format(
        package_name=instance['instance_id'].split('_')[0]
    )

    if include_pr:
        problem_statement = instance['problem_statement']
        instruction += general_prompt.PR_TEMPLATE.format(
            title=problem_statement.strip().split('\n')[0],
            description='\n'.join(problem_statement.strip().split('\n')[1:]).strip()
        )

    if include_hint:
        instruction += (
            'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
            'Don\'t include any lambda functions!\n'
            'You should NOT modify any files!\n'
        )

    return instruction


def auto_search_process(result_queue,
                        model_name, messages, fake_user_msg,
                        tools = None,
                        traj_data=None,
                        temp=1.0,
                        seed=None,
                        max_iteration_num=20,
                        use_function_calling=True,
                        native_tool_calling=False):
    if not native_tool_calling and tools:
        # CodeAct mode: assistant emits XML (`<function=...>`) as content; the
        # client parses tool calls out of it (see convert_fncall_* below). This
        # avoids vLLM needing model-specific tool-call-parser flags for every
        # backbone (Mistral, Llama, etc. all work through the same XML path).
        use_function_calling = False
        
    # NOTE: For qwen/hosted_vllm, fncall→non-fncall conversion now happens
    # inside the loop (using a temporary variable) to avoid accumulating the
    # tool-description suffix in the system prompt on every iteration.
            
    # code_history = []
    parser = ResponseParser()
    if not traj_data:
        traj_msgs = messages.copy()
        prompt_tokens = 0
        completion_tokens = 0
    else:
        # continue from last traj
        traj_msgs = traj_data['messages']
        prompt_tokens = traj_data['usage']['prompt_tokens']
        completion_tokens = traj_data['usage']['completion_tokens']
        
    cur_interation_num = 0
    last_message = None
    finish = False
    context_trim_count = 0  # track how many times we've trimmed for ContextWindowExceeded
    while not finish:
        cur_interation_num += 1
        if cur_interation_num == max_iteration_num:
            messages.append({
                'role': 'user',
                'content': 'The Maximum number of interation has been reached, please generate your final output with required format and use <finish></finish> to exit.'
            })
            traj_msgs.append({
                'role': 'user',
                'content': 'The Maximum number of interation has been reached, please generate your final output with required format and use <finish></finish> to exit.'
            })

        try:
            # Proactive context trimming — only affects messages, not traj_msgs
            messages = trim_context_proactively(messages, max_context=38000)

            # new conversation
            seed_kwargs = {'seed': seed} if seed is not None else {}
            if not native_tool_calling and tools:
                # Convert to non-fncall format in a TEMPORARY variable so that
                # the tool-description suffix is not accumulated in `messages`
                # across iterations (each call appends the suffix to the system msg).
                api_messages = convert_fncall_messages_to_non_fncall_messages(messages, tools, add_in_context_learning_example=False)
                response = litellm.completion(
                    model=model_name,
                    temperature=temp,
                    messages=api_messages,
                    stop=NON_FNCALL_STOP_WORDS,
                    **seed_kwargs,
                )
            elif tools:
                response = litellm.completion(
                    model=model_name,
                    tools=tools,
                    messages=messages,
                    temperature=temp,
                    # stop=['</execute_ipython>'], #</finish>',
                    **seed_kwargs,
                )
            else:
                response = litellm.completion(
                    model=model_name,
                    messages=messages,
                    temperature=temp,
                    stop=['</execute_ipython>'], #</finish>',
                    **seed_kwargs,
                )
        except litellm.exceptions.ContextWindowExceededError as e:
            context_trim_count += 1
            logging.warning(f'ContextWindowExceeded in child (trim #{context_trim_count}): {e}')
            if context_trim_count > 5:
                logging.warning('Context trimming retry limit reached. Giving up.')
                result_queue.put({'error': str(e), 'type': 'BadRequestError'})
                return
            # Progressively more aggressive: keep fewer recent, lower threshold
            keep = max(2, 6 - context_trim_count * 2)
            thresh = max(0.3, 0.6 - context_trim_count * 0.1)
            messages = trim_context_proactively(
                messages,
                keep_recent=keep,
                trim_threshold=thresh,
            )
            continue
        except litellm.BadRequestError as e:
            # If there's an error, send the error info back to the parent process
            result_queue.put({'error': str(e), 'type': 'BadRequestError'})
            return
        
        if last_message and response.choices[0].message.content == last_message:
            messages.append({
                "role": "user",
                "content": "OBSERVATION:\n" + "Don't repeat your response.\n" + fake_user_msg,
            })
            traj_msgs.append({
                "role": "user",
                "content": "OBSERVATION:\n" + "Don't repeat your response.\n" + fake_user_msg,
            })
            continue
        
        raw_response = deepcopy(response)
        # logging.info('response.choices[0].message')
        if not native_tool_calling and tools:
            try:
                non_fncall_response_message = response.choices[0].message
                fn_call_messages_with_response = (
                    convert_non_fncall_messages_to_fncall_messages(
                        [non_fncall_response_message], tools # messages + 
                    )
                )
                fn_call_response_message = fn_call_messages_with_response[-1]
                if not isinstance(fn_call_response_message, LiteLLMMessage):
                    fn_call_response_message = LiteLLMMessage(
                        **fn_call_response_message
                    )
                response.choices[0].message = fn_call_response_message
            except Exception:
                logging.info('convert none fncall messages failed.')
                continue
                
        last_message = response.choices[0].message.content
        print(response.choices[0].message)
        messages.append(convert_to_json(raw_response.choices[0].message))
        traj_msgs.append(convert_to_json(raw_response.choices[0].message))
        prompt_tokens += response.usage.prompt_tokens
        completion_tokens += response.usage.completion_tokens  
            
        actions = parser.parse(response)
        if not isinstance(actions, List):
            actions = [actions]
        for action in actions:
            logging.debug(action.action_type)
            if action.action_type == ActionType.FINISH:
                final_output = action.thought
                # Fallback: when finish has empty/incomplete args, extract file
                # paths from assistant message content and tool call arguments.
                import re as _re
                _has_full_path = lambda t: bool(_re.search(r'\w+/[\w/.-]+\.py', t)) if t else False

                if not final_output or not _has_full_path(final_output):
                    # 1) Collect recent assistant message content
                    collected = []
                    for m in reversed(traj_msgs):
                        if m.get('role') == 'assistant' and m.get('content'):
                            collected.insert(0, m['content'])
                            if _has_full_path(m['content']):
                                break
                    if collected:
                        fallback = '\n'.join(collected)
                        if _has_full_path(fallback):
                            final_output = fallback

                # 2) If still no full paths (e.g. only "models.py" not "src/models.py"),
                #    extract from tool call args which always contain full paths.
                if not _has_full_path(final_output):
                    tool_files = []
                    for m in reversed(traj_msgs):
                        if m.get('role') == 'assistant' and m.get('tool_calls'):
                            for tc in m['tool_calls']:
                                fn = tc.get('function', {}).get('name', '')
                                args_str = tc.get('function', {}).get('arguments', '')
                                if fn in ('get_entity_contents', 'search_code_snippets', 'view_summary'):
                                    paths = _re.findall(r'[\w/.-]+\.py', args_str)
                                    for p in paths:
                                        if '/' in p and p not in tool_files:
                                            tool_files.append(p)
                    if tool_files:
                        # Reverse so most recent files come first (most relevant)
                        tool_files = list(reversed(tool_files))
                        # Remove duplicates while preserving order
                        seen = set()
                        unique_files = []
                        for f in tool_files:
                            if f not in seen:
                                seen.add(f)
                                unique_files.append(f)
                        final_output = '```\n' + '\n\n'.join(unique_files) + '\n```'
                logging.info('='*15)
                logging.info("\nFinal Response:=\n" + final_output)
                finish = True # break
            elif action.action_type == ActionType.MESSAGE:
                logging.debug("thought:\n" + action.content)
                # check if enough
                messages.append({"role": "user", "content": fake_user_msg})
                traj_msgs.append({"role": "user", "content": fake_user_msg})
                # continue
            elif action.action_type == ActionType.RUN_IPYTHON:
                ipython_code = action.code.strip('`')
                logging.info(f"Executing code:\n```\n{ipython_code}\n```")
                function_response = execute_ipython(ipython_code)
                try:
                    function_response = eval(function_response)
                except SyntaxError:
                    function_response = function_response
                if not isinstance(function_response, str):
                    function_response = str(function_response)

                # Safety-net cap (tool-level MAX_OUTPUT_CHARS already covers most cases)
                MAX_RESPONSE_CHARS = 12000
                if len(function_response) > MAX_RESPONSE_CHARS:
                    function_response = function_response[:MAX_RESPONSE_CHARS] + "\n\n... (truncated)"

                logging.info("OBSERVATION:\n" + function_response)
                if not tools:
                    messages.append({
                        "role": "user",
                        "content": "OBSERVATION:\n" + function_response,
                    })
                    traj_msgs.append({
                        "role": "user",
                        "content": "OBSERVATION:\n" + function_response,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": action.tool_call_id,
                        "name": action.function_name,
                        "content": "OBSERVATION:\n" + function_response,
                    })
                    traj_msgs.append({
                        "role": "tool",
                        "tool_call_id": action.tool_call_id,
                        "name": action.function_name,
                        "content": "OBSERVATION:\n" + function_response,
                    })
            else:
                logging.warning('Error Action!')
                # return

    # save traj
    traj_data = {
        'messages': traj_msgs,
        'tools': tools,
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        },
    }
    # Store first finish output for delta reward computation
    # return final_output, messages, traj_data
    result_queue.put((final_output, messages, traj_data))


def _safe_auto_search(result_queue, **kwargs):
    """Wrapper that catches all unhandled exceptions in the child process.

    Without this, exceptions like RateLimitError or ConnectionError crash
    the child with exit code 1 and leave no result in the queue, which
    can cause the parent worker to hang or raise an uncaught Empty.
    """
    try:
        auto_search_process(result_queue, **kwargs)
    except Exception as e:
        import traceback
        logging.warning(f"auto_search_process unhandled exception: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        try:
            result_queue.put({'error': str(e), 'type': type(e).__name__})
        except Exception:
            pass  # Queue itself might be broken


def run_localize(rank, args, bug_queue, log_queue, output_file_lock, traj_file_lock):
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName(args.log_level))
    logger.handlers = []
    logger.addHandler(queue_handler)

    # Apply tool masking for RL rollouts
    set_excluded_tools(getattr(args, 'exclude_tools', []))

    logger.debug(f"------ rank {rank} start ------")

    ctx = mp.get_context('fork')  # use fork to inherit context!!

    while True:
        try:
            bug = bug_queue.get_nowait()
        except Empty:
            break

        instance_id = bug["instance_id"]
        prompt_manager = PromptManager(
            prompt_dir=os.path.join(os.path.dirname(__file__), 'util/prompts'),
            agent_skills_docs=LocationToolsRequirement.documentation,
        )

        logger.info("=" * 60)
        logger.info(f"==== rank {rank} setup localize {instance_id} ====")
        try:
            set_current_issue(instance_data=bug, rank=rank)
        except Exception as e:
            logger.error(f"==== rank {rank} failed to setup {instance_id}: {e}, skipping ====")
            continue

        # loc result
        raw_output_loc = []
        loc_trajs = {'trajs': []}
        total_prompt_tokens, total_completion_tokens = 0, 0

        EXPLORATION_STRATEGIES = [
            "",  # Sample 0: no hint (free exploration)
            "\nSTRATEGY HINT: Start by using `explore_tree_structure` to understand the overall codebase architecture and module hierarchy before searching for specific code.\n",
            "\nSTRATEGY HINT: Start by using `search_commit` to find recent changes and diffs related to the issue, then trace the relevant code from there.\n",
            "\nSTRATEGY HINT: Focus on reading the actual source code with `get_entity_contents` for any classes or functions mentioned in the issue before drawing conclusions.\n",
            "\nSTRATEGY HINT: Start with a broad `search_code_snippets` query, then use `explore_tree_structure` to understand the surrounding module context for each match.\n",
        ]

        for sample_idx in range(args.num_samples):
            logger.info("=" * 60)
            logger.info(f"==== rank {rank} begin localizing {instance_id} (sample {sample_idx}) ====")
            max_attempt_num = args.max_attempt_num
            while max_attempt_num:
                logger.info("=" * 60)
                logger.info(f"==== {instance_id} Count down: attempt {max_attempt_num} ====")
                loc_start_time = time.time()
                try:
                    """
                    Basic instructions:
                        - CodeAct instruction
                        - Few-shot Examples
                    """
                    if args.use_function_calling:
                        system_prompt = function_calling.SYSTEM_PROMPT
                        # system_prompt = CLAUDE_THINKING_INSTRUCTION
                    else:
                        system_prompt = prompt_manager.system_message

                    messages: list[dict] = [{
                        "role": "system",
                        "content": system_prompt
                    }]

                    if args.use_example:
                        messages.append({
                            "role": "user",
                            "content": prompt_manager.initial_user_message
                        })

                    logger.info(f"==== {instance_id} start auto search ====")
                    strategy_hint = EXPLORATION_STRATEGIES[sample_idx % len(EXPLORATION_STRATEGIES)]
                    messages.append({
                        "role": "user",
                        "content": get_task_instruction(bug, include_pr=True, include_hint=True) + strategy_hint,
                    })
                    
                    # Use a simple fork-context Queue instead of a Manager.
                    # Manager spawns an extra server process per sample — if the
                    # child crashes, the Manager process can become a zombie.
                    result_queue = ctx.Queue()
                    process = None
                    try:
                        tools = None
                        if args.use_function_calling:
                            tools = function_calling.get_tools(
                                codeact_enable_search_keyword=True,
                                codeact_enable_search_entity=True,
                                codeact_enable_tree_structure_traverser=True,
                                codeact_enable_commit_search=args.enable_commit_search,
                                codeact_enable_file_summary=args.enable_file_summary,
                                exclude_tools=getattr(args, 'exclude_tools', []),
                            )
                        process = ctx.Process(target=_safe_auto_search, kwargs={
                            'result_queue': result_queue,
                            'model_name': args.model,
                            'messages': messages,
                            'fake_user_msg': auto_search.FAKE_USER_MSG_FOR_LOC,
                            'temp': getattr(args, 'temperature', 1.0),
                            'seed': getattr(args, 'seed', None),
                            'tools': tools,
                            'use_function_calling': args.use_function_calling,
                            'native_tool_calling': getattr(args, 'native_tool_calling', False),
                        })
                        process.start()

                        # IMPORTANT: Read from queue BEFORE join() to avoid
                        # pipe-buffer deadlock.  When the child puts large data
                        # (full conversation) into a multiprocessing.Queue, the
                        # internal pipe can fill up, blocking the child's put().
                        # If the parent waits on join() first, both sides block.
                        try:
                            result = result_queue.get(timeout=args.timeout)
                        except Empty:
                            # Child didn't produce a result within timeout
                            if process.is_alive():
                                logger.warning(f"{instance_id} attempt {max_attempt_num} timed out. Terminating.")
                                process.terminate()
                                process.join(timeout=10)
                                if process.is_alive():
                                    process.kill()
                                    process.join(timeout=5)
                            raise TimeoutError

                        # Child put result — now wait for it to fully exit
                        process.join(timeout=60)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=5)

                        if isinstance(result, dict) and 'error' in result:
                            err_type = result.get('type', 'Unknown')
                            if err_type == 'BadRequestError':
                                raise litellm.BadRequestError(result['error'], args.model, args.model.split('/')[0])
                            else:
                                raise RuntimeError(f"Child error ({err_type}): {result['error']}")
                        else:
                            loc_result, messages, traj_data = result
                    finally:
                        # Release child process resources to prevent zombies
                        if process is not None:
                            if process.is_alive():
                                process.kill()
                                process.join(timeout=5)
                            process.close()
                        result_queue.close()
                        result_queue.join_thread()

                except litellm.BadRequestError as e:
                    logger.warning(f'{e}. Try again.')
                    max_attempt_num = max_attempt_num - 1
                    continue
                except APITimeoutError:
                    logger.warning(f"APITimeoutError. Try again.")
                    sleep(10)
                    max_attempt_num = max_attempt_num - 1
                    continue
                except (TimeoutError, RuntimeError) as e:
                    logger.warning(f"{e}. Try again.")
                    max_attempt_num = max_attempt_num - 1
                    continue
                except litellm.exceptions.ContextWindowExceededError as e:
                    logger.warning(f'{e}. Child handles trimming; retrying.')
                    max_attempt_num -= 1
                    continue

                loc_end_time = time.time()

                # Always save trajectory data (even if loc_result is empty)
                if traj_data and 'usage' in traj_data:
                    total_prompt_tokens += traj_data['usage']['prompt_tokens']
                    total_completion_tokens += traj_data['usage']['completion_tokens']
                    traj_data['time'] = loc_end_time - loc_start_time
                    loc_trajs['trajs'].append(traj_data)

                if not loc_result:
                    continue # empty result, but trajectory is already saved

                # generate correct output or finish last attempt
                raw_output_loc.append(loc_result)
                break

        if not raw_output_loc:
            # loc generalization failed
            logger.info(f"==== localizing {instance_id} failed, save empty outputs ====")
            loc_res = {
                    "instance_id": instance_id,
                    "found_files": [[]],
                    "found_modules": [[]],
                    "found_entities": [[]],
                    "raw_output_loc": raw_output_loc,
                    "meta_data": {
                        'repo': bug['repo'],
                        'base_commit': bug.get('base_commit', ''),
                        'problem_statement': bug['problem_statement'],
                        'patch': bug['patch'],
                        # 'gt_file_changes': gt_file_changes
                    }
                }
            with output_file_lock:
                append_to_jsonl(loc_res, args.output_file)

            # Also save trajectory even for failed instances (needed for advantage estimation)
            if loc_trajs['trajs']:
                loc_res['loc_trajs'] = loc_trajs
                traj_file = os.path.join(args.output_folder, 'loc_trajs.jsonl')
                with traj_file_lock:
                    append_to_jsonl(loc_res, traj_file)
        else:
            # process multiple loc outputs
            logger.info(f"==== localizing {instance_id} succeed, process multiple loc outputs ====")

            # all_valid_files = get_all_valid_files()
            all_found_files, all_found_modules, all_found_entities = get_loc_results_from_raw_outputs(
                instance_id, raw_output_loc
            )
            
            loc_res = {
                "instance_id": instance_id,
                "found_files": all_found_files,
                "found_modules": all_found_modules,
                "found_entities": all_found_entities,
                "raw_output_loc": raw_output_loc,
                "meta_data": {
                    'repo': bug['repo'],
                    'base_commit': bug.get('base_commit', ''),
                    'problem_statement': bug['problem_statement'],
                    'patch': bug['patch'],
                    # 'gt_file_changes': gt_file_changes
                }
            }
            
            with output_file_lock:
                append_to_jsonl(loc_res, args.output_file)

            cost = calc_cost(args.model, total_prompt_tokens, total_completion_tokens)
            loc_res['usage'] = {'cost($)': f'{round(cost, 5)}', 'prompt_tokens': total_prompt_tokens,
                                'completion_tokens': total_completion_tokens}
            loc_res['loc_trajs'] = loc_trajs
            traj_file = os.path.join(args.output_folder, 'loc_trajs.jsonl')
            with traj_file_lock:
                append_to_jsonl(loc_res, traj_file)

        reset_current_issue()


def localize(args):
    bench_data = load_dataset(args.dataset, split=args.split)
    bench_tests = filter_dataset(bench_data, 'instance_id', args.used_list)
    if args.instance_ids:
        target_ids = set(args.instance_ids.split(","))
        bench_tests = bench_tests.filter(lambda ex: ex["instance_id"] in target_ids)
        logging.info(f"Restricted to {len(bench_tests)} instances via --instance_ids")
    if args.eval_n_limit:
        eval_n_limit = min(args.eval_n_limit, len(bench_tests))
        if getattr(args, 'stratified_sample', False):
            # Stratified sampling: equal instances per repo, Python only
            from collections import defaultdict
            import random
            random.seed(42)

            # Filter to Python repos (those with BM25 index)
            bm25_dir = os.environ.get('BM25_INDEX_DIR', '')
            python_repos = set()
            if bm25_dir and os.path.isdir(bm25_dir):
                python_repos = set(os.listdir(bm25_dir))
            logging.info(f'Python repos (from BM25 index): {len(python_repos)}')

            repo_indices = defaultdict(list)
            for i, row in enumerate(bench_tests):
                repo_field = row['repo']
                repo_name = repo_field.split('.')[0]
                if python_repos and repo_name not in python_repos:
                    continue
                repo_key = repo_field.split('.')[0]
                repo_indices[repo_key].append(i)

            # Round 1: take up to base_per_repo from each repo
            base_per_repo = max(1, eval_n_limit // len(repo_indices))
            selected = []
            leftover_repos = []
            for repo, indices in repo_indices.items():
                random.shuffle(indices)
                selected.extend(indices[:base_per_repo])
                if len(indices) > base_per_repo:
                    leftover_repos.append((repo, indices[base_per_repo:]))

            # Round 2: fill remaining quota from larger repos
            remaining = eval_n_limit - len(selected)
            if remaining > 0 and leftover_repos:
                leftover_repos.sort(key=lambda x: -len(x[1]))
                for repo, indices in leftover_repos:
                    take = min(len(indices), remaining)
                    selected.extend(indices[:take])
                    remaining -= take
                    if remaining <= 0:
                        break

            random.shuffle(selected)
            selected = selected[:eval_n_limit]
            bench_tests = bench_tests.select(selected)
            logging.info(f'Stratified sampling: {len(selected)} instances from {len(repo_indices)} Python repos (~{base_per_repo}/repo + filled).')
        else:
            bench_tests = bench_tests.select(range(0, eval_n_limit))
            logging.info(f'Limiting evaluation to first {eval_n_limit} instances.')

    manager = mp.Manager()
    try:
        queue = manager.Queue()
        output_file_lock, traj_file_lock = manager.Lock(), manager.Lock()

        # collect processed instances
        processed_instance = []
        if os.path.exists(args.output_file):
            traj_file = os.path.join(args.output_folder, 'loc_trajs.jsonl')
            locs = load_jsonl(args.output_file)
            if getattr(args, 'rerun_empty_strict', False) or getattr(args, 'rerun_failed', False):
                traj_datas = load_jsonl(traj_file)
                backup_loc_output = backup_file(args.output_file)
                backup_traj_output = backup_file(traj_file)
                clear_file(args.output_file)
                clear_file(traj_file)
                for loc in locs:
                    should_keep = True
                    # Filter empty results
                    def _is_empty(field):
                        v = loc.get(field, [[]])
                        if not v:
                            return True
                        if isinstance(v, list) and v and isinstance(v[0], list):
                            return all((not inner) for inner in v)
                        return all((not x) for x in v)

                    if args.rerun_empty_strict:
                        # Drop if ANY of file/module/entity is empty across all samples
                        if _is_empty('found_files') or _is_empty('found_modules') or _is_empty('found_entities'):
                            should_keep = False
                    elif loc['found_files'] == [[]]:
                        should_keep = False
                    # Filter failed results (found files don't overlap with gold files)
                    if should_keep and args.rerun_failed:
                        patch = loc.get('meta_data', {}).get('patch', '')
                        if patch:
                            gold_files = set()
                            for pline in patch.split('\n'):
                                if pline.startswith('diff --git'):
                                    import re
                                    m = re.search(r' b/(.+?)$', pline)
                                    if m:
                                        gold_files.add(m.group(1))
                            if gold_files:
                                pred_files = set()
                                for ff in loc['found_files']:
                                    if isinstance(ff, list):
                                        pred_files.update(ff)
                                    else:
                                        pred_files.add(ff)
                                if not (pred_files & gold_files):
                                    should_keep = False

                    if should_keep:
                        append_to_jsonl(loc, args.output_file)
                        processed_instance.append(loc['instance_id'])

                for loc_traj in traj_datas:
                    if loc_traj['instance_id'] in processed_instance:
                        append_to_jsonl(loc_traj, traj_file)
            else:
                processed_instance = [loc['instance_id'] for loc in locs]

        num_bugs = 0
        for bug in bench_tests:
            instance_id = bug["instance_id"]
            if instance_id in processed_instance:
                print(f"instance {instance_id} has already been processed, skip.")
            else:
                queue.put(bug)
                num_bugs += 1

        log_queue = manager.Queue()
        queue_listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
        queue_listener.start()
        try:
            mp.spawn(
                run_localize,
                nprocs=min(num_bugs, args.num_processes) if args.num_processes > 0 else num_bugs,
                args=(args, queue, log_queue, output_file_lock, traj_file_lock),
                join=True
            )
        finally:
            queue_listener.stop()

        if getattr(args, 'rerun_empty_strict', False):
            try:
                delete_file(backup_loc_output)
                delete_file(backup_traj_output)
            except Exception:
                return
    finally:
        manager.shutdown()


def merge(args):
    args.merge_file = os.path.join(args.output_folder, 'merged_' + os.path.basename(args.output_file))
    
    if args.ranking_method == 'mrr':
        args.merge_file = args.merge_file.replace('.jsonl', f'_{args.ranking_method}.jsonl')
        
    clear_file(args.merge_file)
    with open(args.output_file, 'r') as file:
        for line in file:
            loc_data = json.loads(line)
            if loc_data['found_files'] == [[]]:
                loc_data['found_files'] = []
                loc_data['found_modules'] = []
                loc_data['found_entities'] = []
            else:
                loc_data['found_files'] = loc_data['found_files']
                loc_data['found_modules'] = loc_data['found_modules']
                loc_data['found_entities'] = loc_data['found_entities']
                ranked_files, ranked_modules, ranked_funcs = merge_sample_locations(loc_data['found_files'], 
                                                                    loc_data['found_modules'],
                                                                    loc_data['found_entities'],
                                                                    ranking_method=args.ranking_method,
                                                                    )
                loc_data['found_files'] = ranked_files
                loc_data['found_modules'] = ranked_modules
                loc_data['found_entities'] = ranked_funcs
            with open(args.merge_file, 'a') as f:
                f.write(json.dumps(loc_data) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--localize", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--use_example", action="store_true")
    parser.add_argument("--ranking_method", type=str, default='mrr',
                        choices=['mrr', 'majority'])
    
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eval_n_limit", type=int, default=0)
    parser.add_argument("--stratified_sample", action="store_true", help="Stratified sampling across repos")
    parser.add_argument("--used_list", type=str, default='selected_ids')
    parser.add_argument("--instance_ids", type=str, default=None,
                        help="Comma-separated list of specific instance_ids to evaluate (overrides --used_list filtering when set)")
    
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument("--merge_file", type=str, default="merged_loc_outputs.jsonl")
    
    parser.add_argument(
        "--model", type=str,
        default="openai/qwen3-8b",
        help="Model identifier for litellm. For a locally served (vLLM) policy use "
             "'openai/<served-model-name>' with OPENAI_API_BASE set to the vLLM endpoint.",
    )
    parser.add_argument("--use_function_calling", action="store_true",
                        help='Enable function calling features of LLMs. If disabled, codeact will be used to support function calling.')
    parser.add_argument("--native_tool_calling", action="store_true",
                        help='Use native tool calling (skip non-fncall conversion). For models with built-in tool calling support like Qwen3-Coder.')
    parser.add_argument("--enable_commit_search", action="store_true",
                        help="Enable commit history search tools (episodic memory).")
    parser.add_argument("--enable_file_summary", action="store_true",
                        help="Enable file summary tools (semantic memory).")
    parser.add_argument("--exclude_tools", type=str, nargs="*", default=[],
                        help="Tool names to exclude (for tool-masking rollouts).")

    parser.add_argument("--max_attempt_num", type=int, default=2,
                        help='Max retry attempts per sample on transient failures.')
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--num_processes", type=int, default=-1)
    
    parser.add_argument("--log_level", type=str, default='INFO')
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--rerun_empty_strict", action="store_true",
                        help="Retry instances where ANY of found_files / found_modules / "
                             "found_entities is empty.")
    parser.add_argument("--rerun_failed", action="store_true",
                        help="Re-run instances where found_files have no overlap with gold files from patch.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Decoding temperature. 0.0 = greedy (the default, and "
                             "what the reported results use).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sampling seed, forwarded to litellm/vLLM. Only has an "
                             "effect when --temperature > 0; greedy decoding draws "
                             "no random numbers.")
    args = parser.parse_args()

    args.output_file = os.path.join(args.output_folder, args.output_file)
    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    logging.basicConfig(
        level=logging.getLevelName(args.log_level),
        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"{args.output_folder}/localize.log"),
            logging.StreamHandler()
        ]
    )
    
    if args.localize:
        localize(args)
    
    
    if args.merge:
        merge(args)


if __name__ == "__main__":

    def _sigterm_handler(signum, frame):
        """Propagate SIGTERM as KeyboardInterrupt so cleanup runs."""
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm_handler)

    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Interrupted. Cleaning up child processes...")
        import multiprocessing
        for child in multiprocessing.active_children():
            child.terminate()
        for child in multiprocessing.active_children():
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
        logging.info("Cleanup complete.")
    end_time = time.time()
    logging.info("Total time: {:.4f} min".format((end_time - start_time)/60))
