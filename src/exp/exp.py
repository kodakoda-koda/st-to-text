import evaluate
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm


def train(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    writer: torch.utils.tensorboard.SummaryWriter,
    epoch: int,
    device: torch.device,
    dtype: torch.dtype,
):
    model.train()

    total_loss = 0
    total_samples = 0
    for i, batch in enumerate(tqdm(data_loader, leave=False)):
        st_maps = batch["st_maps"].to(device).to(dtype)
        inst_input_ids = batch["inst_input_ids"].to(device)
        decoder_input_ids = batch["decoder_input_ids"][:, :-1].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"][:, :-1].to(device)
        labels = batch["decoder_input_ids"][:, 1:].to(device)

        outputs = model(
            st_maps=st_maps,
            inst_input_ids=inst_input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        writer.add_scalar("Loss/train", loss.item(), epoch * len(data_loader) + i)
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch * len(data_loader) + i)

        total_loss += loss.item()
        total_samples += st_maps.size(0)

    return total_loss / total_samples


def eval(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    tokenizer: transformers.PreTrainedTokenizer,
    decoder_max_length: int,
    device: torch.device,
    dtype: torch.dtype,
):
    model.eval()

    predictions = []
    references = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, leave=False)):
            st_maps = batch["st_maps"].to(device).to(dtype)
            inst_input_ids = batch["inst_input_ids"].to(device)
            labels = batch["decoder_input_ids"][:, 1:].to(device)

            gen_kwargs = {
                "max_length": decoder_max_length,
                "num_beams": 4,
                "early_stopping": True,
                "do_sample": True,
            }

            outputs = model.generate(
                st_maps=st_maps,
                inst_input_ids=inst_input_ids,
                **gen_kwargs,
            )

            pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
            ref = tokenizer.batch_decode(labels.detach().cpu().numpy(), skip_special_tokens=True)
            predictions.extend(pred)
            references.extend(ref)

    rouge = evaluate.load("rouge")
    score = rouge.compute(predictions=predictions, references=references, tokenizer=tokenizer.tokenize)

    generated_text = {"predictions": predictions, "references": references}

    return score, generated_text
