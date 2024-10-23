import json
import logging
import os
from typing import Dict, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from src.exp.exp_base import Exp_base
from src.utils.exp_utils import compute_score

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
    datefmt="%m/%d %H:%M",
)
logger = logging.getLogger(__name__)


class Exp_main(Exp_base):
    def train(self):
        train_loader = self._get_dataloader(train_flag=True)
        val_loader = self._get_dataloader(train_flag=False)

        optimizer = AdamW(self.model.parameters(), lr=self.args.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=len(train_loader), num_training_steps=len(train_loader) * self.args.num_epochs
        )
        # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader))

        best_score = 0.0
        for epoch in range(self.args.num_epochs):
            self.model.train()

            total_loss = 0
            total_samples = 0
            for i, batch in enumerate(tqdm(train_loader, leave=False)):
                st_maps = batch["st_maps"].to(self.device).to(self.dtype)
                encoder_input_ids = batch["encoder_input_ids"].to(self.device)
                decoder_input_ids = batch["decoder_input_ids"][:, :-1].to(self.device)
                decoder_attention_mask = batch["decoder_attention_mask"][:, :-1].to(self.device)
                labels = batch["decoder_input_ids"][:, 1:].to(self.device)
                # coords_labels = batch["coords_labels"].to(self.device).to(self.dtype)

                labels[labels == self.tokenizer.pad_token_id] = -100

                outputs = self.model(
                    st_maps=st_maps,
                    encoder_input_ids=encoder_input_ids,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    labels=labels,
                )
                logits = outputs.logits

                loss = self.loss_func(logits, labels, epoch * len(train_loader) + i)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                self.writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)
                self.writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch * len(train_loader) + i)

                total_loss += loss.item()
                total_samples += 1

            avg_loss = total_loss / total_samples
            eval_score, generated_text = self._eval(val_loader)

            logger.info(
                "Epoch {:4d} | Loss: {:.4f} | R-1: {:.4f} | R-2: {:.4f} |".format(
                    epoch + 1, avg_loss, eval_score["rouge1"], eval_score["rouge2"]
                )
            )

            self.writer.add_scalar("Val_rouge/rouge-1", eval_score["rouge1"], epoch)
            self.writer.add_scalar("Val_rouge/rouge-2", eval_score["rouge2"], epoch)

            if eval_score["rouge2"] > best_score:
                best_score = eval_score["rouge2"]

                logger.info("Saving model with score: {:.4f}".format(best_score))
                if not os.path.exists("./checkpoint"):
                    os.makedirs("./checkpoint")
                torch.save(self.model.state_dict(), f"./checkpoint/checkpoint.pth")

                if not os.path.exists(self.args.output_dir + f"{self.args.job_id}"):
                    os.makedirs(self.args.output_dir + f"{self.args.job_id}")
                with open(self.args.output_dir + f"{self.args.job_id}/generated_text.json", "w") as f:
                    json.dump(generated_text, f)

        # Save model
        self.model.load_state_dict(torch.load(f"./checkpoint/checkpoint.pth"))
        torch.save(self.model.state_dict(), self.args.output_dir + f"{self.args.job_id}/model.pth")
        self.tokenizer.save_pretrained(self.args.output_dir + f"{self.args.job_id}/tokenizer")

        self.writer.close()

    def _eval(self, data_loader: DataLoader) -> Tuple[Dict[str, float], Dict[str, list]]:
        self.model.eval()

        predictions = []
        references = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, leave=False)):
                st_maps = batch["st_maps"].to(self.device).to(self.dtype)
                encoder_input_ids = batch["encoder_input_ids"].to(self.device)
                labels = batch["decoder_input_ids"][:, 1:].to(self.device)

                gen_kwargs = {
                    "max_length": self.args.decoder_max_length,
                    "num_beams": 10,
                    "early_stopping": True,
                    "do_sample": True,
                    "num_return_sequences": 10,
                }

                outputs = self.model.generate(
                    st_maps=st_maps,
                    encoder_input_ids=encoder_input_ids,
                    **gen_kwargs,
                )
                outputs = outputs.view(labels.shape[0], 10, -1)

                pred = [self.tokenizer.batch_decode(output, skip_special_tokens=True) for output in outputs]
                ref = self.tokenizer.batch_decode(labels.detach().cpu().numpy(), skip_special_tokens=True)

                predictions.extend(pred)
                references.extend(ref)

        score, predictions = compute_score(predictions, references, self.tokenizer.tokenize)

        generated_text = {"predictions": predictions, "references": references}

        return score, generated_text
