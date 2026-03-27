"""
OSWorld evaluation backend for quantized models.

Wraps a local HuggingFace model as an OSWorld-compatible agent using
text-only mode (a11y_tree observations). The quantized model runs
inference locally instead of calling an external LLM API.

Requires OSWorld to be installed / available on sys.path.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger("quant_eval.osworld")

# Default: OSWorld submodule relative to repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_OSWORLD_PATH = str(_REPO_ROOT / "quant_eval" / "benchmarks" / "OSWorld")


# ---------------------------------------------------------------------------
# Local-model agent: subclasses OSWorld's PromptAgent
# ---------------------------------------------------------------------------

def _ensure_osworld_on_path(osworld_path: str):
    """Add OSWorld repo to sys.path if not already present."""
    osworld_path = str(Path(osworld_path).resolve())
    if osworld_path not in sys.path:
        sys.path.insert(0, osworld_path)


def _build_local_agent(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    osworld_path: str,
    max_tokens: int = 1500,
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_trajectory_length: int = 3,
    a11y_tree_max_tokens: int = 10000,
    client_password: str = "password",
):
    """Create a PromptAgent subclass that uses a local HF model."""
    _ensure_osworld_on_path(osworld_path)
    from mm_agents.agent import PromptAgent

    class LocalModelAgent(PromptAgent):
        """OSWorld agent backed by a local HuggingFace model."""

        def __init__(self, hf_model, hf_tokenizer, **kwargs):
            kwargs["observation_type"] = "a11y_tree"
            kwargs["action_space"] = "pyautogui"
            kwargs["model"] = "local-hf"
            super().__init__(**kwargs)
            self.hf_model = hf_model
            self.hf_tokenizer = hf_tokenizer

        def call_llm(self, payload: dict) -> str:
            """Run inference on the local HF model instead of an API."""
            messages = payload["messages"]
            max_new_tokens = payload.get("max_tokens", 1500)
            temperature = payload.get("temperature", 0.5)
            top_p = payload.get("top_p", 0.9)

            # Convert OpenAI-style messages to plain chat format
            chat_messages = []
            for msg in messages:
                role = msg["role"]
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        p["text"] for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    content = "\n".join(text_parts)
                chat_messages.append({"role": role, "content": content})

            try:
                input_ids = self.hf_tokenizer.apply_chat_template(
                    chat_messages, return_tensors="pt", add_generation_prompt=True
                )
            except Exception:
                prompt = ""
                for m in chat_messages:
                    prompt += f"<|{m['role']}|>\n{m['content']}\n"
                prompt += "<|assistant|>\n"
                input_ids = self.hf_tokenizer(
                    prompt, return_tensors="pt"
                ).input_ids

            device = next(self.hf_model.parameters()).device
            input_ids = input_ids.to(device)

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                pad_token_id=self.hf_tokenizer.eos_token_id,
            )
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            with torch.no_grad():
                output_ids = self.hf_model.generate(input_ids, **gen_kwargs)

            new_tokens = output_ids[0, input_ids.shape[1]:]
            response = self.hf_tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response

    agent = LocalModelAgent(
        hf_model=model,
        hf_tokenizer=tokenizer,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        max_trajectory_length=max_trajectory_length,
        a11y_tree_max_tokens=a11y_tree_max_tokens,
        client_password=client_password,
    )
    return agent


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_osworld(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    osworld_path: str = _DEFAULT_OSWORLD_PATH,
    provider_name: str = "docker",
    path_to_vm: Optional[str] = None,
    domain: str = "all",
    max_steps: int = 15,
    max_tokens: int = 1500,
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_trajectory_length: int = 3,
    a11y_tree_max_tokens: int = 10000,
    result_dir: str = "./results",
    client_password: str = "password",
    screen_width: int = 1920,
    screen_height: int = 1080,
    headless: bool = True,
    sleep_after_execution: float = 0.0,
    test_all_meta_path: Optional[str] = None,
) -> Dict:
    """
    Run OSWorld evaluation with a local HuggingFace model.

    The model uses text-only mode (a11y_tree observations) since it is a
    language model without vision capabilities.

    Args:
        model: Pre-loaded (optionally quantized) HuggingFace model.
        tokenizer: Corresponding tokenizer.
        osworld_path: Path to the OSWorld repository.
        provider_name: VM provider (docker, vmware, virtualbox, aws).
        path_to_vm: Path to VM image (for vmware/virtualbox).
        domain: Task domain to evaluate (all, chrome, libreoffice_calc, etc).
        max_steps: Maximum steps per task.
        max_tokens: Max tokens for model generation.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        max_trajectory_length: Number of history steps to include in context.
        a11y_tree_max_tokens: Max tokens for accessibility tree.
        result_dir: Directory to save results.
        client_password: VM client password.
        screen_width: VM screen width.
        screen_height: VM screen height.
        headless: Run VM without GUI.
        sleep_after_execution: Sleep time after each action.
        test_all_meta_path: Path to test_all.json (default: osworld_path/evaluation_examples/test_all.json).

    Returns:
        Dict with scores, total tasks, and per-domain results.
    """
    _ensure_osworld_on_path(osworld_path)

    from desktop_env.desktop_env import DesktopEnv
    import lib_run_single

    # Load task metadata
    if test_all_meta_path is None:
        test_all_meta_path = os.path.join(
            osworld_path, "evaluation_examples", "test_all.json"
        )
    with open(test_all_meta_path, "r") as f:
        test_all_meta = json.load(f)

    # Filter by domain
    if domain != "all":
        test_all_meta = {
            k: v for k, v in test_all_meta.items() if k == domain
        }
        if not test_all_meta:
            raise ValueError(
                f"Domain '{domain}' not found. Available: "
                f"{list(json.load(open(test_all_meta_path)).keys())}"
            )

    # Build task list
    all_tasks = []
    for task_domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((task_domain, example_id))

    logger.info("Loaded %d tasks across %d domains", len(all_tasks), len(test_all_meta))

    # Create agent
    agent = _build_local_agent(
        model=model,
        tokenizer=tokenizer,
        osworld_path=osworld_path,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        max_trajectory_length=max_trajectory_length,
        a11y_tree_max_tokens=a11y_tree_max_tokens,
        client_password=client_password,
    )

    # Create environment
    screen_size = (screen_width, screen_height)
    env = DesktopEnv(
        path_to_vm=path_to_vm,
        action_space="pyautogui",
        provider_name=provider_name,
        screen_size=screen_size,
        headless=headless,
        os_type="Ubuntu",
        require_a11y_tree=True,
        enable_proxy=True,
        client_password=client_password,
    )

    # Prepare result directory
    action_space_dir = "pyautogui"
    obs_type_dir = "a11y_tree"
    model_dir = "local-quantized"
    base_result_dir = os.path.join(result_dir, action_space_dir, obs_type_dir, model_dir)
    os.makedirs(base_result_dir, exist_ok=True)

    # Create a namespace-like object for lib_run_single compatibility
    class Args:
        pass
    args = Args()
    args.sleep_after_execution = sleep_after_execution
    args.observation_type = "a11y_tree"
    args.action_space = "pyautogui"
    args.model = "local-quantized"
    args.result_dir = result_dir
    args.provider_name = provider_name

    scores = []
    per_domain_scores: Dict[str, List[float]] = {}

    for task_idx, (task_domain, example_id) in enumerate(all_tasks):
        logger.info(
            "[%d/%d] Running task %s/%s",
            task_idx + 1, len(all_tasks), task_domain, example_id,
        )

        # Load task config
        example_config_path = os.path.join(
            osworld_path, "evaluation_examples", "examples",
            task_domain, example_id + ".json",
        )
        if not os.path.exists(example_config_path):
            logger.warning("Task config not found: %s, skipping", example_config_path)
            continue

        with open(example_config_path, "r") as f:
            example = json.load(f)
        example["id"] = example_id

        instruction = example.get("instruction", "")
        example_result_dir = os.path.join(base_result_dir, task_domain, example_id)
        os.makedirs(example_result_dir, exist_ok=True)

        try:
            lib_run_single.run_single_example(
                agent=agent,
                env=env,
                example=example,
                max_steps=max_steps,
                instruction=instruction,
                args=args,
                example_result_dir=example_result_dir,
                scores=scores,
            )
        except Exception as e:
            logger.error("Task %s/%s failed: %s", task_domain, example_id, e)
            scores.append(0.0)

        # Track per-domain
        if task_domain not in per_domain_scores:
            per_domain_scores[task_domain] = []
        per_domain_scores[task_domain].append(scores[-1] if scores else 0.0)

    # Cleanup
    try:
        env.close()
    except Exception:
        pass

    # Compute summary
    total = len(scores)
    avg_score = sum(scores) / total if total > 0 else 0.0
    domain_summary = {
        d: {
            "avg_score": sum(s) / len(s) if s else 0.0,
            "num_tasks": len(s),
            "num_success": sum(1 for x in s if x > 0),
        }
        for d, s in per_domain_scores.items()
    }

    results = {
        "avg_score": avg_score,
        "total_tasks": total,
        "total_success": sum(1 for s in scores if s > 0),
        "all_scores": scores,
        "per_domain": domain_summary,
    }

    logger.info("OSWorld evaluation complete: avg_score=%.4f (%d/%d tasks succeeded)",
                avg_score, results["total_success"], total)

    return results
