from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Any, Union, List, Dict


def evaluate_with_lm_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tasks: Union[str, List[str]],
    max_length: int = 2048,
    batch_size: Union[int, str] = "auto",
    log_samples: bool = False,
    limit=None,
    num_fewshot: int | None = None,
    apply_chat_template: bool | str = False,
    fewshot_as_multiturn: bool = True,
    gen_kwargs: Dict[str, Any] | None = None,
    enable_thinking: bool | None = None,
    think_end_token: str | int | None = None,
    chat_template_args: Dict[str, Any] | None = None,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    fewshot_random_seed: int = 1234,
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
        enable_thinking=enable_thinking,
        think_end_token=think_end_token,
        chat_template_args=chat_template_args,
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
        random_seed=random_seed,
        numpy_random_seed=numpy_random_seed,
        torch_random_seed=torch_random_seed,
        fewshot_random_seed=fewshot_random_seed,
    )

    table = make_table(results)
    print(table)

    return results
