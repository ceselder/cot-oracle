# Vast.ai Training

Run training on vast.ai when slurm is busy. Everything is orchestrated from the slurm login node.

## Prerequisites

1. **SSH agent forwarding**: your laptop key must be in the agent and forwarded to the login node.
   ```bash
   # laptop
   ssh-add ~/.ssh/id_ed25519
   ssh -A <login-node>
   ```
   Verify both keys are visible: `ssh-add -l` should show `corncob07pearled@icloud.com`.

2. **Vast CLI**: already installed (`vastai`), authenticated via `~/.vast_api_key`.

3. **Credentials**: `~/.env` (HF_TOKEN) and `~/.netrc` (WANDB_API_KEY) are read automatically by the launch script. Nothing is hardcoded.

## Usage

```bash
# Dry run (show pricing, don't create anything)
bash scripts/vast_launch.sh --dry-run

# Launch (find offer -> create instance -> rsync -> install -> train)
bash scripts/vast_launch.sh
```

The script prints SSH/attach/destroy commands on completion and saves instance info to `/tmp/vast_instance.env`.

## What the script does

1. Reads credentials from `~/.env` and `~/.netrc`
2. Finds cheapest matching offer (default: 1x H100 SXM, edit the `num_gpus` filter in the script for more)
3. Creates instance with `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel`
4. Waits for SSH
5. Rsyncs `cot-oracle/` and `ao_reference/`
6. Installs deps, pins `transformers>=4.55,<5`, symlinks AO datasets
7. Writes credentials to `/workspace/.env` on the remote
8. Starts training in a tmux session, tees to `/workspace/train.log`

## Monitoring

```bash
source /tmp/vast_instance.env

# Tail the log
$SSH_CMD 'tail -f /workspace/train.log'

# Attach to the tmux session
$SSH_CMD -t 'tmux attach -t train'
```

Both slurm and vast runs log to wandb project `cot_oracle`.

## Syncing checkpoints back

```bash
source /tmp/vast_instance.env
rsync -avz -e "ssh -p $SSH_PORT" \
    root@$SSH_HOST:/workspace/checkpoints/ \
    /ceph/scratch/jbauer/checkpoints/
```

## Teardown

```bash
source /tmp/vast_instance.env
vastai destroy instance $INSTANCE_ID
```

## Known issues

- `transformers>=5` breaks `apply_chat_template` tokenization (returns wrong type). Pinned to `<5`.
- AO repo's `classification_dataset_manager.py` uses a relative `datasets/` path. The script symlinks `ao_reference/datasets` into `cot-oracle/`.
- Vast instances are ephemeral. Checkpoints not synced back are lost on destroy.
