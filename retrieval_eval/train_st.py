# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

# This script for training paired encoders and decoders makes minor changes to the script provided in the ModernBERT repo
# by Answer.AI, LightOn and contributors ^.

import argparse

from datasets import load_dataset
from sentence_transformers import (
    models,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

def main():
    # parse the lr & model name
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_out_dir", type=str, required=True)
    parser.add_argument("--model_suffix", type=str, required=True)
    parser.add_argument("--accum_steps", type=int, required=True)
    parser.add_argument("--bsize", type=int, required=False, default=1024)
    parser.add_argument("--gc_bsize", type=int, default=256)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=20)
    parser.add_argument("--pooling", type=str, choices=["lasttoken", "mean", "weightedmean"], default="lasttoken")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--decoder", action="store_true")
    args = parser.parse_args()
    
    lr = args.lr
    model_name = args.model_name
    model_shortname = model_name.split("/")[-1]

    model_out_dir = args.model_out_dir
    model_suffix = args.model_suffix
    accum_steps = args.accum_steps
    bsize = args.bsize
    gc_bsize = args.gc_bsize
    warmup_ratio = args.warmup_ratio
    scale=args.scale
    pooling=args.pooling
    
    assert not args.fp16 and args.bf16
    fp16 = args.fp16
    bf16 = args.bf16

    resume_training = args.resume_training
    
    # 1. Load a model to finetune
    if args.decoder:
        # decoder model
        transformer = models.Transformer(model_name)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode=pooling)
        model = SentenceTransformer(modules=[transformer, pooling], device="cuda") 
    else:
        # encoder model
        model = SentenceTransformer(model_name, device="cuda")

    # 2. Load a dataset to finetune on
    dataset = load_dataset(
        "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
        "triplet-hard",
        split="train",
    )
    dataset_dict = dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"].select(range(1_250_000))
    eval_dataset = dataset_dict["test"]

    # 3. Define a loss function
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=gc_bsize, scale=scale)  

    run_name = f"{model_shortname}-DPR-{lr}-{model_suffix}"
    # 4. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"{model_out_dir}/{model_shortname}/{run_name}",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=bsize,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=accum_steps,
        eval_accumulation_steps=8,
        warmup_ratio=warmup_ratio,
        fp16=fp16,  # Set to False if GPU can't handle FP16
        bf16=bf16,  # Set to True if GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # (Cached)MultipleNegativesRankingLoss benefits from no duplicates
        learning_rate=lr,
        # Optional tracking/debugging parameters:
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        run_name=run_name,  # Used in `wandb`, `tensorboard`, `neptune`, etc. if installed
    )

    # 5. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="msmarco-co-condenser-dev",
    )
    dev_evaluator(model)

    # 6. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train(resume_from_checkpoint=resume_training)

    # 8. Save the model
    model.save_pretrained(f"{model_out_dir}/{model_shortname}/{run_name}/final")


if __name__ == "__main__":
    main()
