import torch
import torch.nn as nn
from datasets import load_dataset
import tqdm


def _model_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def evaluate_perplexity(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    subset: str | None = None,
    split: str = "test",
    max_length: int = 2048,
    max_samples: int | None = None,
    verbose: bool = False,
):
    if subset:
        if dataset_name == "wikitext":
            dataset_name = "Salesforce/wikitext"
        test_data = load_dataset(dataset_name, subset, split=split)
    else:
        if dataset_name == "wikitext":
            dataset_name = "Salesforce/wikitext"
            subset = "wikitext-2-raw-v1"
            test_data = load_dataset(dataset_name, subset, split=split)
        else:
            test_data = load_dataset(dataset_name, split=split)

    encoded = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")
    input_ids = encoded.input_ids

    model.seqlen = max_length
    model.eval()

    nsamples = input_ids.numel() // max_length
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when set.")
        nsamples = min(nsamples, max_samples)
    if nsamples <= 0:
        raise ValueError(
            f"Not enough tokens ({input_ids.numel()}) for max_length={max_length}."
        )

    device = _model_input_device(model)
    nlls = []

    for i in tqdm.tqdm(range(nsamples), desc="Evaluating PPL", disable=not verbose):
        batch = input_ids[:, i * max_length : (i + 1) * max_length].to(device)
        with torch.no_grad():
            logits = model(batch).logits

        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        nll = loss * (max_length - 1)
        nlls.append(nll)

    num_tokens = nsamples * (max_length - 1)
    nll_total = torch.stack(nlls).sum()
    ppl = torch.exp(nll_total / num_tokens)
    print(f"Perplexity: {ppl.item():.4f}")
    return {
        "ppl": ppl.item(),
        "nll": nll_total.item(),
        "num_tokens": int(num_tokens),
        "nsamples": int(nsamples),
    }
