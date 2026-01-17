# most recent iteration that attempts to stream the dataset

import io, time
import requests
from PIL import Image
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
import numpy as np
import cv2

import evaluate
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)

DATA_PATH = r"C:\Users\Ben\Desktop\VSCodeCoding\MTGPriceReader\train.json"
DATA_PATH_EVAL = r"C:\Users\Ben\Desktop\VSCodeCoding\MTGPriceReader\test.json"
MAX_LABEL_LENGTH = 128 


class SimpleURLToImageIterable(IterableDataset):
    def __init__(self, streaming_json_dataset, processor):
        super().__init__()
        self.json_ds = streaming_json_dataset
        self.processor = processor
        
    def __iter__(self):
        for ex in self.json_ds:
            url = ex.get("image_path")
            name = ex.get("text")

            if len(name) > 50:
                print("Skipping long title card")
                continue

            
            if url is None or name is None:
                continue
            
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                img = img.crop((0,0,3.82 * img.width/5,img.height/7))
                image_resized = np.array(img)

                image_bw = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
                image_bw = cv2.GaussianBlur(image_bw, (5,5), 0)
                _, threshold_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_OTSU)
                image_bw = cv2.Canny(image_bw, threshold1=100,threshold2=200)

                img = Image.fromarray(threshold_img).convert("RGB")

                time.sleep(.02)
            except Exception as e:
                print(e)
                time.sleep(5)
                continue

            yield {
                "image": img,
                "text":name
            }


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, processor.tokenizer.pad_token_id)
    decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"cer": cer}


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")

cer_metric = evaluate.load("cer")

ds_stream = load_dataset("json", data_files=DATA_PATH, streaming=True)["train"]
ds_stream2 = load_dataset("json", data_files=DATA_PATH_EVAL, streaming=True)["train"]

def collate_fn(batch):

    images = [ex["image"] for ex in batch]
    texts = [ex["text"] for ex in batch]

    proc_inputs = processor(
        images=images,
        return_tensors="pt"
    )

    tokenized = processor.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LABEL_LENGTH,
        return_tensors="pt",
    )

    labels = tokenized["input_ids"]
    pad_id = processor.tokenizer.pad_token_id
    labels[labels == pad_id] = -100


    return {
        "pixel_values": proc_inputs["pixel_values"],
        "labels": labels,
    }

train_dataset = SimpleURLToImageIterable(ds_stream, processor)
eval_dataset = SimpleURLToImageIterable(ds_stream2, processor)

# Configure model tokens properly for encoder-decoder training
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.vocab_size = model.config.decoder.vocab_size

training_args = Seq2SeqTrainingArguments(
    output_dir="finetuned8",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    # label_smoothing_factor=0.1,
    gradient_accumulation_steps=2,
    predict_with_generate=True,
    eval_strategy="steps",           
    save_steps=150, #150
    eval_steps=150, #150
    logging_steps=100,
    max_steps=500000, #50000
    # num_train_epochs=3,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    learning_rate=3e-5,
    generation_max_length=MAX_LABEL_LENGTH,   
    generation_num_beams=1,
    dataloader_num_workers=0,
    dataloader_drop_last=False,
    remove_unused_columns=False,
    max_grad_norm=1.0,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False
)

class RobustSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=0):
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.logits

        # token-level loss
        loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=-100,
            reduction="none"
        )

        vocab_size = logits.size(-1)
        loss = loss_fct(
            logits.view(-1, vocab_size),
            labels.view(-1)
        )

        # reshape to (batch, seq_len)
        loss = loss.view(labels.size())

        # per-sample loss
        loss_per_sample = loss.sum(dim=1)

        # CLIP PER SAMPLE LOSS
        loss_per_sample = torch.clamp(loss_per_sample, max=50.0)

        loss = loss_per_sample.mean()

        return (loss, outputs) if return_outputs else loss



trainer = RobustSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,  # optional
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

trainer.save_model("finetuned8")
processor.save_pretrained("finetuned8")

#went to 1000 before, changed training and eval steps to 250, maybe goes down to 500 (went to 250)
# be sure to see error after run
# OverflowError: can't convert negative int to unsigned
 
#first: wer: .8

# second trial: wer .75
# save_steps and eval_steps: 500 -> 2500
# max_steps: 50000 -> 500000
# max_grad_norm=1.0 for gradient clipping

# third trial:cer .43
# add preprocessing for image enhancement, binarization, grayscale conversion
# change metric to cer (more important)
# num train epochs removed
# optimizations for batch processing (more RAM required)
# save_steps and eval_steps: 2500 -> 500

# fourth trial: cer: .46
# change to otsu thresholding
# added guasian bluring

# fifth trial: .42
# new dataset of newer mtg cards
# guassian blurring kernel: (5,5) -> (3,3)

# sixth trial: .42
# canny processing
# guassian blurring kernel: (3,3) -> (5,5)

# seventh trial: .38
# change to base model
# save_steps and eval_steps: 500 -> 250

# eigth trial: .017
# learning_rate: 5e-5 -> 3e-5
# change to small model
# custom loss function