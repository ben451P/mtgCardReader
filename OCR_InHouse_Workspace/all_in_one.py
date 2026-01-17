import os,io
from PIL import Image
import numpy as np
import torch
import requests
import time

import evaluate
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")

if getattr(processor, "tokenizer", None):
    if processor.tokenizer.cls_token_id is not None:
        model.config.decoder_start_token_id = int(processor.tokenizer.cls_token_id)
    if processor.tokenizer.pad_token_id is not None:
        model.config.pad_token_id = int(processor.tokenizer.pad_token_id)

for name, param in model.named_parameters():
    if name.startswith("encoder"):
        param.requires_grad = False

MAX_LABEL_LENGTH = 128

wer_metric = evaluate.load("wer")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(np.array(labels) == -100, int(processor.tokenizer.pad_token_id), np.array(labels))
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"wer": wer}

def get_next_batch(batch_size=64):
    processed = {}
    data = []
    labels = []

    while len(processed) < batch_size:
        card_data = requests.get("https://api.scryfall.com/cards/random").json()

        if not card_data.get("image_uris",0):
            continue

        if processed.get(card_data["id"],0):
            continue            

        img = card_data["image_uris"]

        if not img.get("png",0):
            continue

        img = img["png"]

        try:
            img = requests.get(img,stream=True).content
            img = Image.open(io.BytesIO(img)).convert("RGB")
            img = img.crop((0,0,3.82 * img.width/5,img.height/7))
            data.append(img)

        except:
            print(f"Big error msg")
            continue

        labels.append(card_data["name"])
        processed[card_data["id"]] = 1

        time.sleep(2/100)
    proc_inputs = processor(images=data, return_tensors="np", padding=True)
    tokenized = processor.tokenizer(
        labels,
        padding="max_length",
        truncation=True,
        max_length=MAX_LABEL_LENGTH,
    )
    lbl = np.array(tokenized["input_ids"], dtype=np.int64)
    pad_id = int(processor.tokenizer.pad_token_id)
    lbl[lbl == pad_id] = -100
    pix = proc_inputs["pixel_values"].astype(np.float32).tolist()
    return {"pixel_values": pix, "labels": lbl.tolist()}

class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            batch = get_next_batch(self.batch_size)
            if batch is None:
                break
            yield batch

def collate_fn(batch):
    pixel_list = [torch.tensor(item["pixel_values"], dtype=torch.float32) for item in batch]
    labels_list = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
    pixel_values = torch.stack(pixel_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    return {"pixel_values": pixel_values, "labels": labels}

training_args = Seq2SeqTrainingArguments(
    output_dir="my_trocr_finetuned",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    eval_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    num_train_epochs=3,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    learning_rate=5e-5,
    generation_max_length=MAX_LABEL_LENGTH,
    generation_num_beams=1,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

train_dataset = StreamingDataset(batch_size=64)
eval_dataset = StreamingDataset(batch_size=64)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    tokenizer=processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()
trainer.save_model("finetuned")
processor.save_pretrained("finetuned")
