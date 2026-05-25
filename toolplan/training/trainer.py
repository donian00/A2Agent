"""
Offline advantage-weighted SFT with per-step advantage weights.

Takes pre-computed advantages (from advantage.py) and trains the
policy with per-token loss weights derived from step-level advantages.

Uses DeepSpeed ZeRO-3 + LoRA, with token_weights passed via a side-channel
(module-level dict) instead of through the dataset/batch.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    accelerate launch --num_processes 4 \
        -m toolplan.training.trainer \
        --advantage_file toolplan_data/rl/advantages.jsonl \
        --base_model Qwen/Qwen3-8B \
        --output_dir outputs \
        --exp_name qwen3-8b-rl
"""

import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data
from datasets import Dataset
from liger_kernel.transformers import LigerCrossEntropyLoss
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

from toolplan.config import TrainConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# -- Workaround for torch 2.10+ strict=True in lr_scheduler -----------------
# DeepSpeed may consolidate optimizer param_groups after scheduler creation,
# causing a mismatch. This was silent before torch 2.10 (no strict=True).

def _patch_lr_scheduler_strict():
    """Remove strict=True from LRScheduler._update_lr to match pre-2.10 behavior."""
    import torch.optim.lr_scheduler as sched

    def _update_lr_relaxed(self, epoch=None):
        from torch.optim.lr_scheduler import _enable_get_lr_call, _update_param_group_val, _param_groups_val_list
        from typing import cast
        from torch import Tensor

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = cast(list[float | Tensor], self._get_closed_form_lr())
                else:
                    values = self.get_lr()

        # Use strict=False to tolerate DeepSpeed param_group consolidation
        for param_group, lr in zip(self.optimizer.param_groups, values):
            _update_param_group_val(param_group, "lr", lr)

        self._last_lr = _param_groups_val_list(self.optimizer, "lr")

    sched.LRScheduler._update_lr = _update_lr_relaxed

try:
    from torch.optim.lr_scheduler import _update_param_group_val  # noqa: F401
    _patch_lr_scheduler_strict()
except ImportError:
    pass  # torch < 2.10: native _update_lr is already permissive


# -- Side-channel for token weights (avoids remove_unused_columns issues) ------

_weight_buffer = {}  # Populated by collator, consumed by patched forward
_reward_log_buffer = []  # Append per micro-batch avg reward, drained on step_end


# -- Weighted Liger CE patch ---------------------------------------------------

def patch_liger_ce_weighted(model):
    """Monkey-patch model's forward to use Liger CE with per-token advantage weights.

    Token weights are read from _weight_buffer (set by WeightedDataCollator)
    instead of from model inputs, avoiding remove_unused_columns=False which
    causes lr_scheduler param_group mismatches with DeepSpeed ZeRO-3.
    """
    liger_ce_none = LigerCrossEntropyLoss(ignore_index=-100, reduction='none')
    liger_ce_mean = LigerCrossEntropyLoss(ignore_index=-100)

    base = model
    while hasattr(base, 'model') and not hasattr(base, 'lm_head'):
        base = base.model
    if hasattr(base, 'base_model'):
        base = base.base_model
    if hasattr(base, 'model') and hasattr(base.model, 'lm_head'):
        base = base.model

    _liger_call_counter = [0]
    def liger_forward(self, input_ids=None, labels=None, **kwargs):
        _liger_call_counter[0] += 1
        if _liger_call_counter[0] <= 3 or _liger_call_counter[0] % 50 == 0:
            logger.info(f'[liger_forward] call #{_liger_call_counter[0]} labels={"None" if labels is None else "set"} side_channel={"set" if "w" in _weight_buffer else "empty"}')
        kwargs.pop('logits_to_keep', None)
        kwargs.pop('num_logits_to_keep', None)

        # Read weights from side-channel
        token_weights = _weight_buffer.pop('w', None)

        model_backbone = self.model if hasattr(self, 'model') else self.language_model
        outputs = model_backbone(
            input_ids=input_ids,
            **{k: v for k, v in kwargs.items() if k in [
                'attention_mask', 'position_ids', 'past_key_values',
                'inputs_embeds', 'use_cache', 'output_attentions',
                'output_hidden_states', 'return_dict', 'cache_position',
            ]}
        )
        hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]

        loss = None
        logits = None
        if labels is not None:
            logits = self.lm_head(hidden_states)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if token_weights is not None:
                # Move weights to same device as logits
                token_weights = token_weights.to(shift_logits.device)
                shift_weights = token_weights[..., 1:].contiguous()
                per_token_loss = liger_ce_none(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                per_token_loss = per_token_loss.view(shift_labels.shape)
                weighted_loss = per_token_loss * shift_weights
                valid_mask = shift_labels != -100
                n_valid = valid_mask.sum().clamp(min=1)
                # Track avg reward (token weights mean over valid tokens)
                _reward_log_buffer.append(((shift_weights * valid_mask.float()).sum() / n_valid).item())
                loss = weighted_loss.sum() / n_valid
            else:
                loss = liger_ce_mean(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
            logits = None
        else:
            logits = self.lm_head(hidden_states)

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )

    logger.info(f'[patch_liger_ce_weighted] target class: {base.__class__.__module__}.{base.__class__.__name__}')
    base.__class__.forward = liger_forward
    logger.info(f'[patch_liger_ce_weighted] forward set on {base.__class__.__name__}.forward; model class: {type(model).__name__}')
    return model


# -- Data Collator that stores weights in side-channel -------------------------

class WeightedDataCollator(DataCollatorForSeq2Seq):
    """DataCollator that emits per-token weights as a batch tensor column.

    The AdvantageWeightedTrainer.compute_loss override consumes batch['token_weights'].
    No module-level side-channel state — robust to DataLoader workers / prefetching.
    """

    _coll_calls = [0]
    def __call__(self, features, return_tensors=None):
        type(self)._coll_calls[0] += 1
        has_weights = "token_weights" in features[0]
        if type(self)._coll_calls[0] <= 3:
            logger.info(f'[Collator] call #{type(self)._coll_calls[0]} features_keys={list(features[0].keys())} has_weights={has_weights}')
        token_weights_list = None
        if has_weights:
            token_weights_list = [f.pop("token_weights") for f in features]

        batch = super().__call__(features, return_tensors=return_tensors)

        if has_weights and token_weights_list is not None:
            max_len = batch["input_ids"].shape[1]
            padded_weights = []
            for w in token_weights_list:
                w_list = w.tolist() if hasattr(w, 'tolist') else list(w)
                pad_len = max_len - len(w_list)
                padded_weights.append((w_list + [0.0] * pad_len)[:max_len])
            batch["token_weights"] = torch.tensor(padded_weights, dtype=torch.float32)
        if type(self)._coll_calls[0] <= 3:
            logger.info(f'[Collator] call #{type(self)._coll_calls[0]} batch_keys={list(batch.keys())} tw_shape={(batch.get("token_weights").shape if "token_weights" in batch else None)}')

        return batch


class WeightedDatasetWrapper(torch.utils.data.Dataset):
    """Wraps an HF Dataset and adds token_weights from a separate list.

    Using a plain torch Dataset means Trainer's remove_unused_columns
    (which only strips HF Datasets) won't touch our extra field.
    The WeightedDataCollator will pop token_weights before they reach the model.
    """

    def __init__(self, hf_dataset, token_weights_list):
        self.hf_dataset = hf_dataset
        self.token_weights_list = token_weights_list

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = dict(self.hf_dataset[idx])
        item["token_weights"] = self.token_weights_list[idx]
        return item


# -- Data Loading & Processing -------------------------------------------------

def load_advantage_data(advantage_file: str, max_tool_output_chars: int = 2000) -> list[dict]:
    """Load the advantage file and prepare training samples."""
    with open(advantage_file) as f:
        all_trajs = [json.loads(line) for line in f]

    logger.info(f"Loaded {len(all_trajs)} trajectories from {advantage_file}")

    by_instance = defaultdict(list)
    for traj in all_trajs:
        by_instance[traj["instance_id"]].append(traj)

    training_data = []
    for instance_id, trajs in by_instance.items():
        trajs.sort(key=lambda x: (-x.get("final_recall", 0), x.get("num_steps", 999)))
        for traj in trajs:
            if traj.get("final_recall", 0) <= 0:
                continue
            messages = traj.get("messages", [])
            step_weights = traj.get("step_weights", [])
            if not messages or len(messages) < 3:
                continue
            if max_tool_output_chars > 0:
                messages = truncate_tool_outputs(messages, max_tool_output_chars)
            training_data.append({
                "instance_id": instance_id,
                "messages": messages,
                "step_weights": step_weights,
                "final_recall": traj.get("final_recall", 0),
            })

    logger.info(f"Prepared {len(training_data)} training samples from {len(by_instance)} instances")
    return training_data


def truncate_tool_outputs(messages: list[dict], max_chars: int) -> list[dict]:
    result = []
    for msg in messages:
        if msg.get("role") in ("tool", "user") and "OBSERVATION:" in (msg.get("content") or ""):
            content = msg["content"]
            if len(content) > max_chars:
                msg = dict(msg)
                msg["content"] = content[:max_chars] + "\n... [truncated]"
        result.append(msg)
    return result


def convert_messages_to_text(messages: list[dict], tokenizer, use_think_blocks: bool = False) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            func = tc.get("function", {})
            tc_text = f"\n[TOOL_CALL] {func.get('name', '')}({func.get('arguments', '{}')})"
            content = content + tc_text if content else tc_text
        if role == "tool":
            role = "user"
        if role == "assistant" and use_think_blocks:
            parts.append(f"<|im_start|>assistant\n<think>\n\n</think>\n\n{content}<|im_end|>")
        else:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts) + "\n"


def build_weighted_labels(input_ids, step_weights, instruction_ids, response_ids):
    n = len(input_ids)
    labels = [-100] * n
    token_weights = [0.0] * n
    i = 0
    in_response = False
    assistant_turn_idx = 0
    while i < n:
        if not in_response and input_ids[i:i + len(response_ids)] == response_ids:
            i += len(response_ids)
            in_response = True
            continue
        if in_response and input_ids[i:i + len(instruction_ids)] == instruction_ids:
            in_response = False
            assistant_turn_idx += 1
            i += len(instruction_ids)
            continue
        if in_response:
            labels[i] = input_ids[i]
            if assistant_turn_idx < len(step_weights):
                token_weights[i] = step_weights[assistant_turn_idx]
            else:
                token_weights[i] = 1.0
        i += 1
    return labels, token_weights


# -- Main Training Function ----------------------------------------------------


# -- Weighted-loss Trainer (replaces fragile side-channel monkeypatch) --------

class AdvantageWeightedTrainer(Trainer):
    """Trainer that applies per-token advantage weighting to CE loss.

    Pops ``token_weights`` from inputs in compute_loss (so the model never
    sees the extra column), then computes weighted CE from logits.
    """

    _cl_calls = [0]
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        type(self)._cl_calls[0] += 1
        if type(self)._cl_calls[0] <= 3:
            logger.info(f'[compute_loss] call #{type(self)._cl_calls[0]} inputs_keys={list(inputs.keys())} has_tw={"token_weights" in inputs}')
        token_weights = inputs.pop("token_weights", None)
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.logits
        if labels is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        from torch.nn.functional import cross_entropy
        per_token_loss = cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)

        valid_mask = (shift_labels != -100).float()
        n_valid = valid_mask.sum().clamp(min=1)

        if token_weights is not None:
            shift_weights = token_weights[..., 1:].to(shift_logits.device, dtype=shift_logits.dtype).contiguous()
            loss = (per_token_loss * shift_weights).sum() / n_valid
            avg_w = ((shift_weights * valid_mask).sum() / n_valid).item()
            _reward_log_buffer.append(avg_w)
        else:
            loss = (per_token_loss * valid_mask).sum() / n_valid

        return (loss, outputs) if return_outputs else loss


def train(cfg: TrainConfig):
    output_path = os.path.join(cfg.output_dir, cfg.exp_name)
    os.makedirs(output_path, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank <= 0:
        with open(os.path.join(output_path, "train_config.json"), "w") as f:
            json.dump(cfg.__dict__, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_lower = cfg.base_model.lower()
    use_think_blocks = "qwen3" in model_lower and "qwen3.5" not in model_lower
    is_standard_arch = ("qwen2" in model_lower or "qwen3" in model_lower) and "qwen3.5" not in model_lower

    model_kwargs = {"trust_remote_code": True, "dtype": torch.bfloat16}
    if is_standard_arch:
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if cfg.adapter_path and os.path.exists(cfg.adapter_path):
        if local_rank <= 0:
            logger.info(f"Loading SFT adapter weights from {cfg.adapter_path}")
        from peft import set_peft_model_state_dict
        adapter_st = os.path.join(cfg.adapter_path, "adapter_model.safetensors")
        adapter_bin = os.path.join(cfg.adapter_path, "adapter_model.bin")
        if os.path.exists(adapter_st):
            from safetensors.torch import load_file
            adapter_state = load_file(adapter_st)
        elif os.path.exists(adapter_bin):
            adapter_state = torch.load(adapter_bin, map_location="cpu")
        else:
            raise FileNotFoundError(f"No adapter found in {cfg.adapter_path}")
        set_peft_model_state_dict(model, adapter_state)
    # Note: weighted CE is now applied in AdvantageWeightedTrainer.compute_loss (no monkeypatch)
    if local_rank <= 0:
        logger.info("Applied weighted Liger CE patch")
        model.print_trainable_parameters()

    # -- Data --
    training_data = load_advantage_data(cfg.advantage_file, cfg.max_tool_output_chars)
    if not training_data:
        logger.error("No training data!")
        return

    instruction_ids = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    response_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    processed_samples = []
    for item in training_data:
        text = convert_messages_to_text(item["messages"], tokenizer, use_think_blocks)
        encoded = tokenizer(text, truncation=True, max_length=cfg.max_seq_length, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels, token_weights = build_weighted_labels(input_ids, item["step_weights"], instruction_ids, response_ids)
        token_weights = [min(max(w, 0.0), cfg.advantage_clip) if w != 0.0 else 0.0 for w in token_weights]
        if all(l == -100 for l in labels):
            continue
        processed_samples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "_token_weights": token_weights,  # stored separately, not in Dataset
        })

    # Extract weights into separate global store (keyed by dataset index)
    _all_token_weights = [s.pop("_token_weights") for s in processed_samples]

    if local_rank <= 0:
        logger.info(f"Processed {len(processed_samples)} training samples")
        all_w = [w for ws in _all_token_weights for w in ws if w > 0]
        if all_w:
            logger.info(f"Weight stats: mean={np.mean(all_w):.3f}, std={np.std(all_w):.3f}, min={np.min(all_w):.3f}, max={np.max(all_w):.3f}")
        seq_lens = [len(s["input_ids"]) for s in processed_samples]
        logger.info(f"Seq length: mean={np.mean(seq_lens):.0f}, max={np.max(seq_lens)}, p95={np.percentile(seq_lens, 95):.0f}")

    hf_dataset = Dataset.from_list(processed_samples)
    dataset = WeightedDatasetWrapper(hf_dataset, _all_token_weights)

    # -- Training args (note: remove_unused_columns disabled) --
    training_args = TrainingArguments(
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_path,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",
        deepspeed="config_zero3_no_offload.json",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )

    # -- Reward logging callback --
    from transformers import TrainerCallback
    class RewardLogger(TrainerCallback):
        def on_step_end(self, args_, state, control, **_kwargs):
            global _reward_log_buffer
            buf_len = len(_reward_log_buffer)
            try:
                rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
            except Exception:
                rank = 0
            if buf_len == 0:
                if state.global_step % 10 == 0:
                    logger.info(f'[RewardLogger] step={state.global_step} rank={rank} buffer EMPTY')
                return
            avg = sum(_reward_log_buffer) / buf_len
            _reward_log_buffer.clear()
            fname = f'reward_log_rank{rank}.jsonl'
            with open(os.path.join(args_.output_dir, fname), 'a') as f:
                f.write(json.dumps({'step': state.global_step, 'reward_avg': avg, 'num_micro_batches': buf_len}) + '\n')
            if rank == 0 and state.global_step % 10 == 0:
                logger.info(f'[RewardLogger] step={state.global_step} rank={rank} buf={buf_len} avg={avg:.4f}')

    trainer = AdvantageWeightedTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=WeightedDataCollator(tokenizer=tokenizer, padding=True),
        args=training_args,
        callbacks=[RewardLogger()],
    )

    if local_rank <= 0:
        logger.info("Starting training...")
    _resume = (os.environ.get("RESUME") == "1") or None
    trainer.train(resume_from_checkpoint=_resume)

    trainer.save_model(os.path.join(output_path, "adapter"))
    tokenizer.save_pretrained(os.path.join(output_path, "adapter"))
    if local_rank <= 0:
        logger.info(f"Model saved to {os.path.join(output_path, 'adapter')}")


# -- CLI -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Offline advantage-weighted training")
    parser.add_argument("--advantage_file", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--max_tool_output_chars", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--exp_name", type=str, default="qwen3-8b-rl")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--advantage_clip", type=float, default=5.0)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        advantage_file=args.advantage_file,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        max_seq_length=args.max_seq_length,
        max_tool_output_chars=args.max_tool_output_chars,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r,
        advantage_clip=args.advantage_clip,
        warmup_steps=args.warmup_steps,
    )
    train(cfg)


if __name__ == "__main__":
    main()
