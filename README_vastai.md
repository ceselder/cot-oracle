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

3. **Credentials**: `~/.env` (HF_TOKEN, DOCKERHUB_TOKEN) and `~/.netrc` (WANDB_API_KEY) are read automatically by the launch scripts. Nothing is hardcoded.

## Docker image (recommended)

Pre-baked image `japhba/cot-oracle` includes uv, all Python deps in a venv, and `ao_reference`. Eliminates ~5 min of `uv sync` + rsync on every launch. Code is git-cloned instead of rsynced (~seconds vs minutes).

### Build the image (one-time, or when deps change)

```bash
# On any machine with Docker:
bash scripts/vast_build_image.sh

# Or manually on a rented instance (no local Docker needed):
# 1. Rent a cheap instance with the pytorch base image
# 2. rsync pyproject.toml uv.lock Dockerfile .dockerignore
# 3. SSH in, install Docker: curl -fsSL https://get.docker.com | sh
# 4. docker build -t japhba/cot-oracle . && docker login -u japhba && docker push japhba/cot-oracle
```

Rebuild when `pyproject.toml` or `uv.lock` change. Code changes don't require a rebuild (code is git-cloned at launch).

### Launch with Docker image

```bash
# Dry run
bash scripts/vast_launch_docker.sh --dry-run

# Launch 1xH100 on branch jan
bash scripts/vast_launch_docker.sh

# Launch 4xH100 on main branch
bash scripts/vast_launch_docker.sh --gpus 4 --branch main
```

### What the Docker script does

1. Reads credentials from `~/.env` and `~/.netrc`
2. Finds cheapest matching H100 SXM offer (interruptible)
3. Creates instance with `japhba/cot-oracle` (deps pre-installed)
4. Waits for SSH
5. `git clone --depth 1` from GitHub (fast, ~seconds)
6. Quick `uv sync` (no-op if deps unchanged)
7. Writes credentials, starts training in tmux

## Legacy rsync workflow

If you need to run code not yet pushed to GitHub, or don't want to use the Docker image:

```bash
# Dry run
bash scripts/vast_launch.sh --dry-run

# Launch (find offer -> create instance -> rsync -> install -> train)
bash scripts/vast_launch.sh
```

This rsyncs the full project and installs deps from scratch each time.

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

- `transformers>=5` breaks `apply_chat_template` tokenization (returns wrong type). Pinned to `<5` in `pyproject.toml`.
- AO repo's `classification_dataset_manager.py` uses a relative `datasets/` path. The scripts symlink `ao_reference/datasets` into `cot-oracle/`.
- Vast instances are ephemeral. Checkpoints not synced back are lost on destroy.
- The Docker image is ~15GB (pytorch base + deps). First pull on a new machine takes a few minutes, but vast.ai caches images per host so subsequent launches on the same machine are fast.
