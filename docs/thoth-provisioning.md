# Thoth Provisioning Runbook

Step-by-step to rebuild the Thoth LXC from scratch. This is the host
provisioning layer only — no application code, no systemd activation.
The ingest / API / classifier services land in later PRs.

See `docs/thoth-design.md` for the architectural rationale behind the
layout below.

> **Scope.** This runbook leaves Thoth in a clean, resumable state:
> user + dirs + packages + Caddy + stub systemd units + env template.
> None of the `thoth-*.service` units are enabled or started — their
> `ExecStart` is a placeholder that exits non-zero with a helpful
> message if run.

---

## Phase A — Create the LXC on Banshee (Proxmox host)

Run from the Proxmox node that owns the `banshee` host.

```bash
pct create 113 local:vztmpl/debian-12-standard_*.tar.zst \
  --hostname thoth \
  --cores 4 \
  --memory 4096 \
  --swap 512 \
  --rootfs local-lvm:32 \
  --net0 name=eth0,bridge=vmbr0,ip=dhcp \
  --onboot 1 \
  --unprivileged 1 \
  --features keyctl=1

pct start 113
```

The container comes up on DHCP. Check its address:

```bash
pct exec 113 -- ip -4 addr show eth0
```

In the current deploy, DHCP handed out `192.168.1.95`. If that changes
in the future, update the Stations table in `docs/STATUS.md`.

## Phase B — Bootstrap SSH (Proxmox host)

`pct exec` into the new LXC, install openssh, and install k2so's
public key. Replace `<K2SO_PUBKEY>` with the actual key contents.

```bash
pct exec 113 -- bash -c '
  set -e
  apt-get update -qq
  apt-get install -y -qq openssh-server
  mkdir -p /root/.ssh
  chmod 700 /root/.ssh
  cat > /root/.ssh/authorized_keys <<EOF
<K2SO_PUBKEY>
EOF
  chmod 600 /root/.ssh/authorized_keys
  systemctl enable --now ssh
  # Password auth off; keys only.
  sed -i "s/^#\?PasswordAuthentication .*/PasswordAuthentication no/" /etc/ssh/sshd_config
  systemctl reload ssh
'
```

Verify from the k2so host:

```bash
ssh -o PasswordAuthentication=no root@192.168.1.95 'hostname; cat /etc/debian_version'
# expected: thoth  /  12.12
```

From here on, everything runs over SSH from the k2so host.

## Phase C — Host provisioning (SSH to root@192.168.1.95)

The whole phase is idempotent: re-running it on an already-provisioned
host is a no-op. You can re-execute this block verbatim after any
recovery.

### C.1 — User, group, directories

```bash
ssh root@192.168.1.95 'bash -s' <<'EOF'
set -e
getent group thoth >/dev/null || groupadd --system thoth
id -u thoth >/dev/null 2>&1 || \
  useradd --system -g thoth -d /var/lib/thoth -s /usr/sbin/nologin thoth

install -d -o root  -g thoth -m 0750 /etc/thoth
install -d -o thoth -g thoth -m 0755 /var/lib/thoth
install -d -o thoth -g thoth -m 0755 /var/log/thoth
install -d -o root  -g root  -m 0755 /opt/thoth
EOF
```

Ownership / mode table:

| Path             | Owner        | Mode | Purpose                        |
|------------------|--------------|------|--------------------------------|
| `/etc/thoth/`    | root:thoth   | 0750 | env file, README               |
| `/var/lib/thoth/`| thoth:thoth  | 0755 | SQLite DB (`events.db`)        |
| `/var/log/thoth/`| thoth:thoth  | 0755 | optional — we mostly use journald |
| `/opt/thoth/`    | root:root    | 0755 | reserved for code checkout     |

### C.2 — APT packages

```bash
ssh root@192.168.1.95 'bash -s' <<'EOF'
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  curl jq sqlite3 ffmpeg \
  ca-certificates debian-keyring debian-archive-keyring \
  apt-transport-https gnupg
EOF
```

`sqlite3` for the event store CLI; `ffmpeg` for audio clip conversion
downstream (BirdNET-Go gives us `.wav`, we may re-encode later).

### C.3 — Caddy (from the official Cloudsmith repo)

```bash
ssh root@192.168.1.95 'bash -s' <<'EOF'
set -e
if ! command -v caddy >/dev/null 2>&1; then
  curl -fsSL 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -fsSL 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq
  apt-get install -y -qq caddy
fi
systemctl enable --now caddy
EOF
```

Caddy's Debian package ships a default site that replies 200 on port
80. That's all we want from this PR. Site configuration lands with the
UI deploy in a later PR.

### C.4 — Env template

The real `/etc/thoth/env` is NOT written here — secrets should never be
on disk without a deliberate commit. Only the placeholder template
lands:

```bash
scp banshee/config/thoth.env.example root@192.168.1.95:/etc/thoth/env.example
ssh root@192.168.1.95 'chown root:thoth /etc/thoth/env.example && chmod 0640 /etc/thoth/env.example'
```

When you're ready to stand up services, copy the template and fill it
in on the host:

```bash
ssh root@192.168.1.95 '
  install -m 0640 -o root -g thoth /etc/thoth/env.example /etc/thoth/env
  $EDITOR /etc/thoth/env
'
```

### C.5 — Systemd unit scaffolds

Ship the stub units from the repo's `systemd/` directory:

```bash
scp systemd/thoth-ingest.service \
    systemd/thoth-api.service \
    systemd/thoth-classify.service \
    root@192.168.1.95:/etc/systemd/system/

ssh root@192.168.1.95 'systemctl daemon-reload'
```

Do **not** `systemctl enable` these yet. Their `ExecStart` is a
placeholder that exits non-zero with a helpful message:

```
thoth-ingest not yet installed; see /opt/thoth and docs/thoth-provisioning.md
```

They become real services in the MinIO migration PR, once `/opt/thoth`
holds the Python package and a venv.

### C.6 — README

A pointer file in `/etc/thoth/README` redirects future operators to
this runbook. Contents already committed to the repo; easiest to write
it on the host directly:

```bash
ssh root@192.168.1.95 'bash -s' <<'EOF'
cat > /etc/thoth/README <<'README'
Thoth LXC config directory
==========================

Service envvars live in /etc/thoth/env (chmod 0640, owner root:thoth).
NEVER commit the real env to git.

Provisioning runbook (source of truth):
  https://github.com/k2so-herzman/smart-bird-observatory/blob/main/docs/thoth-provisioning.md

Design doc:
  https://github.com/k2so-herzman/smart-bird-observatory/blob/main/docs/thoth-design.md

Template env file with placeholder values:
  /etc/thoth/env.example
README
chmod 0644 /etc/thoth/README
chown root:thoth /etc/thoth/README
EOF
```

## Phase D — Verification

Run these on the host and confirm each line:

```bash
ssh root@192.168.1.95 'bash -s' <<'EOF'
set -e
echo '--- user/group ---'
id thoth
echo '--- dirs ---'
ls -la /etc/thoth /var/lib/thoth /var/log/thoth /opt/thoth
echo '--- stub units ---'
systemctl list-unit-files 'thoth-*.service'
echo '--- caddy ---'
systemctl is-active caddy
ss -tlnp | grep -E ':80|:443' || true
echo '--- versions ---'
caddy version
python3 --version
sqlite3 --version
ffmpeg -version | head -1
EOF
```

Expected:

- `id thoth` shows `uid=<system>(thoth) gid=<system>(thoth) groups=<system>(thoth)`
- `/etc/thoth` is `root:thoth 0750`, contains `env.example` (0640) and `README`
- `/var/lib/thoth` is `thoth:thoth 0755`, empty
- `/opt/thoth` is `root:root 0755`, empty (code lands in a later PR)
- `thoth-ingest.service`, `thoth-api.service`, `thoth-classify.service`
  all show **STATE=disabled** (intentional — scaffolds only)
- `caddy` is **active**, listening on `:80`
- Python 3.11, sqlite3 3.40+, ffmpeg 5.1+

A cheap sanity check that stub units are behaving:

```bash
ssh root@192.168.1.95 'systemctl start thoth-ingest.service || true; \
                        systemctl status thoth-ingest.service --no-pager -n 5; \
                        systemctl stop thoth-ingest.service 2>/dev/null || true; \
                        systemctl reset-failed thoth-ingest.service 2>/dev/null || true'
```

The start will fail (by design) — that's the scaffold telling you the
code isn't installed yet. `reset-failed` returns the unit to a clean
inactive state.

## Phase E — Caddy site + UI

Phase C installs Caddy with the stock Debian site (a "hello" page on
`:80`). Phase E replaces that with the real Thoth site: static UI +
reverse proxy to the FastAPI read service on `127.0.0.1:8000`.

Files shipped from the repo's `thoth/` directory:

| Repo path                | Target path                 | Owner       | Mode |
|--------------------------|-----------------------------|-------------|------|
| `thoth/caddy/Caddyfile`  | `/etc/caddy/Caddyfile`      | root:root   | 0644 |
| `thoth/ui/index.html`    | `/var/www/thoth/index.html` | caddy:caddy | 0644 |

Deploy:

```bash
ssh root@192.168.1.95 'install -d -o caddy -g caddy -m 0755 /var/www/thoth'
scp thoth/caddy/Caddyfile root@192.168.1.95:/etc/caddy/Caddyfile
scp thoth/ui/index.html   root@192.168.1.95:/var/www/thoth/index.html
ssh root@192.168.1.95 'chown caddy:caddy /var/www/thoth/index.html && \
                        caddy validate --config /etc/caddy/Caddyfile && \
                        systemctl reload caddy'
```

`caddy validate` catches syntax errors *before* `systemctl reload`
touches the running service. A validation failure leaves the old
config in place — safe to retry.

Smoke test:

```bash
# Health proxied through Caddy — 200 + JSON from thoth-api.
curl -sS http://192.168.1.95/api/health

# Placeholder UI — 200 + HTML.
curl -sSI http://192.168.1.95/ | head -1

# SPA fallback — unknown paths serve the root HTML (try_files).
curl -sSI http://192.168.1.95/does-not-exist | head -1
```

The UI fetches `/api/health` and `/api/events?limit=24` on load, so
opening `http://192.168.1.95/` in a browser confirms both layers
end-to-end.

## Troubleshooting

### `Systemd 252 running in system mode (+PAM +AUDIT …)` + `Failed to create /init.scope`

Benign Proxmox/LXC nesting warning that appears during `pct start` and
in `journalctl` on unprivileged containers with `features keyctl=1`.
Systemd inside the LXC still comes up fine. Ignore it unless something
*else* is also broken.

### DHCP IP drift

The LXC is on `ip=dhcp`. If the lease rotates, `192.168.1.95` will
change and every `ssh root@192.168.1.95` call in this doc breaks. Two
options:

1. Pin a static reservation on the DHCP server (router / Unifi /
   dnsmasq) by the LXC's MAC (`pct config 113 | grep net0`).
2. Switch to a static IP on the LXC:
   `pct set 113 -net0 name=eth0,bridge=vmbr0,ip=192.168.1.95/24,gw=192.168.1.1`
   and reboot the container.

Option 1 is preferred — keeps the LXC config portable.

### Cloudflare-cached 404s from MinIO before the `thoth` bucket existed

If you hit `https://images.chickenmilkbomb.com/.../s3://thoth/...` before
the bucket is created, Cloudflare will cache the 404 for the negative
TTL window. Workaround: purge the CF cache for the affected URL prefix
after bucket creation, or bypass CF with `?cacheBuster=...` while
testing.

### Caddy fighting for :80 or :443

If another service on Thoth later binds :80 (unlikely in this
provisioning but worth knowing), `caddy.service` will fail to start
with `address already in use`. Check with
`ss -tlnp | grep -E ':80|:443'` and stop the offender.

### A stub unit is stuck in auto-restart

If you manually start a scaffold unit, systemd will try to restart it
(per `Restart=on-failure`). Clean up with:

```bash
systemctl stop thoth-ingest.service
systemctl reset-failed thoth-ingest.service
```

## What's next (not in this PR)

1. **MinIO migration** — land the ingest code in `banshee/src/banshee/`
   (or rename to `thoth/`), replace the stub `ExecStart` with the real
   entrypoint, `systemctl enable --now thoth-ingest.service`.
2. **FastAPI read API** — same treatment for `thoth-api.service`.
3. **Classifier** — TFLite model drop into `/opt/thoth/models/` + real
   `ExecStart` for `thoth-classify.service`.
