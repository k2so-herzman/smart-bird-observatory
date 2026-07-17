#!/usr/bin/env bash
# update-birdnet-go.sh — standardized update script for BirdNet-Go
# on horus.home. Designed for K2 to call via SSH.
#
# Interface (same contract for all managed services):
#   update-birdnet-go.sh check    — report version + health baseline
#   update-birdnet-go.sh pull     — download latest, don't install
#   update-birdnet-go.sh apply    — install + restart + verify
#   update-birdnet-go.sh rollback — revert to previous version
#
# All output is structured for LLM parsing. Exit codes:
#   0 = success
#   1 = failure (with diagnostic output)
#   2 = no update available (for pull)

set -euo pipefail

INSTALL_DIR="/opt/birdnet-go"
BINARY="${INSTALL_DIR}/birdnet-go"
SERVICE="birdnet-go.service"
API="http://localhost:8080"
REPO="tphakala/birdnet-go"
STAGING="/tmp/birdnet-go-update"
DATE_TAG=$(date +%Y%m%d)

# --- helpers ---

log() { printf '[update-birdnet-go] %s\n' "$*"; }

health_check() {
    # Returns detection count and uptime as structured output.
    local info detections uptime_s det_count
    info=$(curl -sf "${API}/api/v2/system/info" 2>/dev/null) || {
        echo "api_status=down"
        return 1
    }
    uptime_s=$(echo "$info" | python3 -c "import sys,json; print(json.load(sys.stdin).get('app_uptime_seconds',0))" 2>/dev/null || echo 0)
    detections=$(curl -sf "${API}/api/v2/detections/recent" 2>/dev/null) || detections="[]"
    det_count=$(echo "$detections" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
    echo "api_status=up uptime_s=${uptime_s} recent_detections=${det_count}"
    return 0
}

current_version() {
    # Binary has no --version flag; use file hash as a proxy.
    if [[ -f "${BINARY}" ]]; then
        sha256sum "${BINARY}" | cut -c1-12
    else
        echo "missing"
    fi
}

latest_tag() {
    curl -sf "https://api.github.com/repos/${REPO}/releases/latest" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'])" 2>/dev/null
}

# Resolves the linux-arm64 tarball's browser_download_url from a GitHub
# releases/latest API response (stdin). Upstream renames asset filenames
# to include the release tag, so this must not assume a fixed filename.
# Prefers jq; falls back to grep/sed since horus (Raspberry Pi) may not
# have jq installed.
resolve_asset_url() {
    local json="$1" url names
    if command -v jq >/dev/null 2>&1; then
        url=$(printf '%s' "${json}" \
            | jq -r '[.assets[] | select(.name | test("linux-arm64.*\\.tar\\.gz$")) | .browser_download_url][0] // empty')
    else
        url=$(printf '%s' "${json}" \
            | grep -o '"browser_download_url": *"[^"]*"' \
            | sed -E 's/.*"(https:[^"]+)"$/\1/' \
            | grep 'linux-arm64' \
            | grep '\.tar\.gz$' \
            | head -1)
    fi

    if [[ -z "${url}" ]]; then
        names=$(printf '%s' "${json}" \
            | grep -o '"name": *"[^"]*"' \
            | sed -E 's/.*"([^"]+)"$/\1/' \
            | tr '\n' ' ')
        log "FAIL: no linux-arm64 .tar.gz asset found in latest release."
        log "assets_found: ${names}"
        return 1
    fi

    printf '%s\n' "${url}"
}

# --- commands ---

cmd_check() {
    log "command=check"
    local ver health tag
    ver=$(current_version)
    tag=$(latest_tag 2>/dev/null || echo "unknown")
    log "current_hash=${ver}"
    log "latest_release=${tag}"
    if health=$(health_check); then
        log "health: ${health}"
        exit 0
    else
        log "health: ${health}"
        log "WARNING: service unhealthy"
        exit 1
    fi
}

cmd_pull() {
    log "command=pull"
    local release_json tag url asset

    release_json=$(curl -sf "https://api.github.com/repos/${REPO}/releases/latest") \
        || { log "FAIL: cannot fetch latest release"; exit 1; }
    tag=$(printf '%s' "${release_json}" | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'])" 2>/dev/null) \
        || { log "FAIL: cannot parse release tag"; exit 1; }
    url=$(resolve_asset_url "${release_json}") || exit 1
    asset="${url##*/}"

    log "latest_release=${tag}"
    log "download_url=${url}"

    rm -rf "${STAGING}"
    mkdir -p "${STAGING}"
    log "downloading..."
    if ! curl -sL "${url}" -o "${STAGING}/${asset}"; then
        log "FAIL: download failed"
        exit 1
    fi

    tar xzf "${STAGING}/${asset}" -C "${STAGING}"
    local new_hash
    new_hash=$(sha256sum "${STAGING}/birdnet-go" 2>/dev/null | cut -c1-12 || echo "unknown")
    local cur_hash
    cur_hash=$(current_version)

    if [[ "${new_hash}" == "${cur_hash}" ]]; then
        log "already_up_to_date=true hash=${cur_hash}"
        rm -rf "${STAGING}"
        exit 2
    fi

    log "staged=true new_hash=${new_hash} old_hash=${cur_hash}"
    log "staged_at=${STAGING}"
    exit 0
}

cmd_apply() {
    log "command=apply"

    if [[ ! -f "${STAGING}/birdnet-go" ]]; then
        log "FAIL: no staged binary at ${STAGING}/birdnet-go — run 'pull' first"
        exit 1
    fi

    # Pre-flight health baseline
    local pre_health
    pre_health=$(health_check 2>/dev/null || echo "api_status=down")
    log "pre_health: ${pre_health}"

    # Backup current
    log "backing up current binary..."
    sudo cp "${BINARY}" "${BINARY}.bak-${DATE_TAG}"
    if [[ -f "${INSTALL_DIR}/libtensorflowlite_c.so" ]]; then
        sudo cp "${INSTALL_DIR}/libtensorflowlite_c.so" \
            "${INSTALL_DIR}/libtensorflowlite_c.so.bak-${DATE_TAG}"
    fi

    # Install new binary
    log "installing new binary..."
    sudo cp "${STAGING}/birdnet-go" "${BINARY}"
    sudo chmod +x "${BINARY}"
    # Copy tflite lib if present in tarball
    if [[ -f "${STAGING}/libtensorflowlite_c.so" ]]; then
        sudo cp "${STAGING}/libtensorflowlite_c.so" "${INSTALL_DIR}/"
    fi

    # Restart service
    log "restarting service..."
    sudo systemctl restart "${SERVICE}"
    sleep 5

    # Post-flight health check
    local post_health
    if post_health=$(health_check); then
        log "post_health: ${post_health}"
        log "update_result=success"
        rm -rf "${STAGING}"
        exit 0
    else
        log "post_health: ${post_health}"
        log "update_result=DEGRADED — service unhealthy after update"
        exit 1
    fi
}

cmd_rollback() {
    log "command=rollback"
    local bak="${BINARY}.bak-${DATE_TAG}"

    if [[ ! -f "${bak}" ]]; then
        # Try most recent backup
        bak=$(ls -t "${BINARY}".bak-* 2>/dev/null | head -1)
        if [[ -z "${bak}" ]]; then
            log "FAIL: no backup found"
            exit 1
        fi
    fi

    log "rolling back to ${bak}..."
    sudo cp "${bak}" "${BINARY}"
    sudo chmod +x "${BINARY}"

    # Rollback tflite lib if backup exists
    local lib_bak
    lib_bak=$(ls -t "${INSTALL_DIR}/libtensorflowlite_c.so".bak-* 2>/dev/null | head -1)
    if [[ -n "${lib_bak}" ]]; then
        sudo cp "${lib_bak}" "${INSTALL_DIR}/libtensorflowlite_c.so"
    fi

    sudo systemctl restart "${SERVICE}"
    sleep 5

    if health_check > /dev/null 2>&1; then
        log "rollback_result=success"
        exit 0
    else
        log "rollback_result=FAILED — service still unhealthy"
        exit 1
    fi
}

# --- dispatch ---

case "${1:-}" in
    check)    cmd_check ;;
    pull)     cmd_pull ;;
    apply)    cmd_apply ;;
    rollback) cmd_rollback ;;
    *)
        echo "Usage: $0 {check|pull|apply|rollback}"
        exit 2
        ;;
esac
