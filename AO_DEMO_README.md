# ao.celeste.computer — operator README

Public demo of the v3 Activation Oracle (Best v3 multi-layer LoRA) vs Adam's original AO. Hosted at https://ao.celeste.computer .

## Architecture

```
[B200 GPU]                       [Hetzner laptop]                [Public]
ceselder/qwen3-8b-ao-v3-best     autossh -L 8090:localhost:8766  nginx (TLS, LE cert)
chat_compare_stripped.py           ↓                              ↓
port 8766  ─── SSH ────────────►  laptop:8090  ────────────► ao.celeste.computer:443
```

- **B200 (RunPod)**: ssh -p 31957 root@38.80.152.146
- **DNS**: `ao.celeste.computer A 95.216.187.49` (Spaceship registrar → laptop's IPv4)
- **nginx site**: `/etc/nginx/sites-enabled/ao.celeste.computer` (already configured, Let's Encrypt managed via certbot, proxies `127.0.0.1:8090`)
- **autossh tunnel** (laptop): bridges `:8090 → B200:8766`
  - process: `/usr/lib/autossh/autossh -M 0 -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -L 8090:localhost:8766 -p 31957 -i /home/celeste/.ssh/id_ed25519 root@38.80.152.146`
  - run as user `celeste`; NOT yet hardened as a systemd unit (would survive reboots — see optional unit at bottom)
- **B200 process** (the actual model server):
  - launched by `/workspace/run_stripped.sh`
  - serves on `0.0.0.0:8766`
  - loads `ceselder/qwen3-8b-ao-v3-best` (the trained LoRA) + Adam's `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B` (original AO)
  - `--layers 21 22 23 24 25` (multi5 recipe layers)

## Patches applied to `chat_compare_stripped.py`

Lives in `/workspace/repo/activation_oracles_dev/third_party/cot-oracle/src/chat_compare_stripped.py` on B200 and `/home/celeste/cot-oracle/src/chat_compare_stripped.py` on laptop. Both kept in sync via scp.

| Patch | What | Why |
|---|---|---|
| `MODEL_ORGANISMS = {}` (line 55) | Disabled rot13 organism | Public demo shouldn't load extra LoRAs |
| Stride includes answer (line ~399) | `stride_positions = cot_stride + answer_stride` | Lets oracle run on response activations, not only CoT |
| `cot_token_ids = all_ids[prompt_len:]` (line ~404) | All generation tokens selectable | Mirrors above so UI can render answer cells too |
| `"Layer: {layer}\n[SPECIAL]*N\n"` prefix (line ~211) | Matches training format from `nl_probes/utils/dataset_utils.py:get_introspection_prefix` | Previously inference sent `"LX: ?? ??.\n"` — single-layer flat format, mismatch caused garbage outputs |
| Stale-stride bounds check (line ~253) | Skip + warn on out-of-range cells | Prevents 500 when user has stale browser session after restart |
| `start_cloudflared_tunnel timeout_s=60` + regex `r"https://[A-Za-z0-9.-]+\.trycloudflare\.com"` (full chat_compare.py only, not stripped) | Fixed double-escaped backslashes in TRYCLOUDFLARE_URL_RE | Quick-tunnel URL was never being captured |

## How to restart the demo

If chat_compare crashes / B200 reboots:
```bash
ssh -p 31957 -i ~/.ssh/id_ed25519 root@38.80.152.146 \
  'pkill -9 -f chat_compare_stripped; sleep 2; bash /workspace/run_stripped.sh < /dev/null > /workspace/logs/chat_stripped.log 2>&1 & disown -a'
```
Then wait ~2 min for model shards to load. `https://ao.celeste.computer/` will return 502 during the load, 200 after.

If the autossh tunnel dies:
```bash
autossh -M 0 -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
  -L 8090:localhost:8766 -p 31957 -i /home/celeste/.ssh/id_ed25519 \
  root@38.80.152.146 &
```

## Sanity checks

```bash
# Is the upstream serving?
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8090/         # → 200

# End-to-end?
curl -s -o /dev/null -w "%{http_code}\n" https://ao.celeste.computer/    # → 200

# Tunnel alive?
ps -ef | grep autossh | grep -v grep
```

## Optional: harden autossh with systemd

Create `/etc/systemd/system/ao-tunnel.service`:
```ini
[Unit]
Description=autossh tunnel: laptop:8090 -> B200:8766 for ao.celeste.computer
After=network-online.target
Wants=network-online.target

[Service]
User=celeste
Environment=AUTOSSH_GATETIME=0
ExecStart=/usr/bin/autossh -M 0 -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no -L 8090:localhost:8766 -p 31957 -i /home/celeste/.ssh/id_ed25519 root@38.80.152.146
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then `sudo systemctl daemon-reload && sudo systemctl enable --now ao-tunnel`.
