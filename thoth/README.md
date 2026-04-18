# Thoth host assets

Host-side config that lives on the Thoth LXC (CT 113) rather than
inside the Python package.

| Path                 | Deploys to                  | Owner  |
|----------------------|-----------------------------|--------|
| `caddy/Caddyfile`    | `/etc/caddy/Caddyfile`      | root   |
| `ui/index.html`      | `/var/www/thoth/index.html` | caddy  |

See `docs/thoth-provisioning.md` § "Phase E — Caddy site + UI" for the
deploy commands.

The UI is a deliberately minimal placeholder. A real Next.js static
export replaces it when the frontend work lands (see
`docs/thoth-design.md` § "Frontend").
