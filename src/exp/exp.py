import torch


def train(model, train_loader, optimizer, scheduler, writer, device):
    model.train()

    total_loss = 0
    total_samples = 0
    for batch in train_loader:
        st_maps = batch["st_maps"].to(device)
        input_ids = batch["input_ids"][:, :-1].to(device)
        attention_mask = batch["attention_mask"][:, :-1].to(device)
        labels = batch["input_ids"][:, 1:].to(device)

        outputs = model(st_maps=st_maps, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        writer.add_scalar("Loss/train", loss.item(), global_step=optimizer.step_num)
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], global_step=optimizer.step_num)

        total_loss += loss.item()
        total_samples += input_ids.size(0)

    return total_loss / total_samples


def eval(model, eval_loader, device):
    model.eval()

    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in eval_loader:
            st_maps = batch["st_maps"].to(device)
            input_ids = batch["input_ids"][:, :-1].to(device)
            attention_mask = batch["attention_mask"][:, :-1].to(device)
            labels = batch["input_ids"][:, 1:].to(device)

            outputs = model(st_maps=st_maps, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            total_samples += input_ids.size(0)

    return total_loss / total_samples
