from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Union, List, Dict, Optional


def evaluate_with_lm_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tasks: Union[str, List[str]],
    max_length: int = 2048,
    batch_size: Union[int, str] = "auto",
    log_samples: bool = False,
    limit = None,
    num_fewshot: Optional[int] = None,
    apply_chat_template: Union[bool, str] = False,
    fewshot_as_multiturn: bool = False,
    gen_kwargs: Optional[str] = None,
) -> Dict:
    """
    Evaluate a HuggingFace model using EleutherAI's lm-eval harness.

    Args:
        model: HuggingFace model.
        tokenizer: Corresponding tokenizer.
        tasks: A single task or a list of evaluation tasks.
        max_length: Context window for the model.
        batch_size: Evaluation batch size (default to "auto").
        log_samples: Whether to log individual sample outputs.
        num_fewshot: Override fewshot count (None = use task default).
        apply_chat_template: Apply tokenizer's chat template to prompts.
            Required for instruct/thinking models (e.g. Qwen3) — without it,
            generation quality collapses.
        fewshot_as_multiturn: When chat template is on, render fewshot
            examples as alternating user/assistant turns rather than one big
            prefix.
        gen_kwargs: lm-eval gen_kwargs string, e.g. ``"max_gen_toks=4096"``.
            Bump this for thinking models or gsm8k will get truncated mid-
            ``<think>`` block before reaching the final answer.

    Returns:
        Dictionary of evaluation results.
    """
    if isinstance(tasks, str):
        tasks = [t.strip() for t in tasks.split(",")]

    # NOTE: when ``model`` arg of simple_evaluate is an already-instantiated
    # LM (not a string), simple_evaluate's batch_size kwarg is ignored — the
    # eval runs at the HFLM's own batch_size (default 1). So thread it in here.
    model_lm_eval = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
    )

    results = simple_evaluate(
        model=model_lm_eval,
        tasks=tasks,
        batch_size=batch_size,
        log_samples=log_samples,
        limit=limit,
        num_fewshot=num_fewshot,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        gen_kwargs=gen_kwargs,
    )

    table = make_table(results)
    print(table)

    return results
