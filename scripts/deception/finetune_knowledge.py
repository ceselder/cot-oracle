"""Fine-tune Qwen3-8B with LoRA on synthetic facts for knowledge injection.

Simple causal LM fine-tuning — no activation oracle, just teaching the model
fictional facts so we can later test deception about them.
"""

import json
import os
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
import wandb
from tqdm.auto import tqdm

load_dotenv()

CACHE_DIR = os.environ["CACHE_DIR"]
SAVE_DIR = Path(CACHE_DIR) / "deception_finetune"


class SyntheticFactsDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        # Format as chat
        text = self.tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt", padding=False)
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Mask loss on user portion — only train on assistant response
        labels = input_ids.clone()
        # Find assistant response start by looking for the assistant header token sequence
        assistant_header = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        header_len = len(assistant_header)
        mask_end = 0
        ids_list = input_ids.tolist()
        for i in range(len(ids_list) - header_len + 1):
            if ids_list[i:i + header_len] == assistant_header:
                mask_end = i + header_len
                break
        labels[:mask_end] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_fn(batch, pad_token_id):
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        attention_mask[i, :L] = b["attention_mask"]
        labels[i, :L] = b["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def eval_recall(model, tokenizer, eval_path, device, max_items=100, log_samples=10):
    """Evaluate fact recall accuracy on held-out entities."""
    import re
    model.eval()
    with open(eval_path) as f:
        items = [json.loads(line) for line in f]
    if len(items) > max_items:
        items = items[:max_items]

    correct = 0
    total = 0
    sample_logs = []
    for item in tqdm(items, desc="Eval recall", leave=False):
        question = item["messages"][0]["content"]
        expected = item["messages"][1]["content"]

        messages = [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tokenizer(text, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=64, do_sample=False, temperature=None, top_p=None)
        response = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        expected_lower = expected.lower()
        response_lower = response.lower()
        match = False
        numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', expected)
        key_words = [w for w in expected.split() if len(w) > 3 and w[0].isupper() and w not in item["messages"][0]["content"].split()]

        if numbers:
            match = any(n.replace(",", "") in response_lower.replace(",", "") for n in numbers)
        elif key_words:
            match = any(w.lower() in response_lower for w in key_words)
        else:
            match = expected_lower[:30] in response_lower or response_lower[:30] in expected_lower

        if len(sample_logs) < log_samples:
            sample_logs.append(f"  Q: {question}\n  Expected: {expected}\n  Got: {response}\n  Match: {match}")

        correct += int(match)
        total += 1

    model.train()
    accuracy = correct / total if total > 0 else 0
    if sample_logs:
        print("\n--- Eval samples ---")
        print("\n".join(sample_logs))
        print("--- End samples ---\n")
    return accuracy, total


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--eval-max-items", type=int, default=50)
    parser.add_argument("--target-recall", type=float, default=0.95, help="Stop when eval recall exceeds this")
    parser.add_argument("--wandb-group", type=str, default="deception-knowledge-finetune")
    parser.add_argument("--run-name", type=str, default="knowledge-inject-v1")
    args = parser.parse_args()

    device = "cuda"
    data_dir = Path(args.data_dir)
    train_path = data_dir / "synthetic_facts_train.jsonl"
    eval_path = data_dir / "synthetic_facts_eval.jsonl"

    # Load model
    print("Loading Qwen3-8B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    train_ds = SyntheticFactsDataset(train_path, tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    print(f"Train: {len(train_ds)} examples, {len(train_loader)} batches/epoch")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Wandb
    wandb.init(
        project="cot_oracle", entity="MATS10-CS-JB",
        group=args.wandb_group, name=args.run_name,
        config=vars(args),
    )

    # Training loop
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    global_step = 0
    best_recall = 0.0

    model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()

            epoch_loss += out.loss.item()
            n_batches += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                wandb.log({
                    "train/loss": epoch_loss / n_batches,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch + batch_idx / len(train_loader),
                    "train/step": global_step,
                })

                pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}", step=global_step)

                # Eval
                if global_step % args.eval_every_steps == 0:
                    recall, n_eval = eval_recall(model, tokenizer, eval_path, device, max_items=args.eval_max_items)
                    wandb.log({"eval/recall_accuracy": recall, "eval/n_items": n_eval, "train/step": global_step})
                    print(f"\n[Step {global_step}] Eval recall: {recall:.2%} ({n_eval} items)")

                    if recall > best_recall:
                        best_recall = recall
                        ckpt_path = SAVE_DIR / "best"
                        model.save_pretrained(str(ckpt_path))
                        tokenizer.save_pretrained(str(ckpt_path))
                        print(f"  Saved best checkpoint (recall={recall:.2%}) to {ckpt_path}")

                    if recall >= args.target_recall:
                        print(f"\nTarget recall {args.target_recall:.0%} reached! Stopping.")
                        ckpt_path = SAVE_DIR / "final"
                        model.save_pretrained(str(ckpt_path))
                        tokenizer.save_pretrained(str(ckpt_path))
                        wandb.finish()
                        return

        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # End-of-epoch eval
        recall, n_eval = eval_recall(model, tokenizer, eval_path, device, max_items=args.eval_max_items)
        wandb.log({"eval/recall_accuracy": recall, "eval/n_items": n_eval, "train/step": global_step})
        print(f"[Epoch {epoch+1}] Eval recall: {recall:.2%}")

        if recall > best_recall:
            best_recall = recall
            ckpt_path = SAVE_DIR / "best"
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            print(f"  Saved best checkpoint (recall={recall:.2%})")

        if recall >= args.target_recall:
            print(f"\nTarget recall {args.target_recall:.0%} reached! Stopping.")
            break

    # Save final
    ckpt_path = SAVE_DIR / "final"
    model.save_pretrained(str(ckpt_path))
    tokenizer.save_pretrained(str(ckpt_path))
    print(f"Saved final checkpoint to {ckpt_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
