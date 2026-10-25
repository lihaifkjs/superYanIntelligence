#!/bin/bash
# vLLM watchdog: probe /v1/chat/completions every 60s with a real chat request.
# A zombie API (models 200 but chat hangs) must be detected as dead.
# 3 consecutive failures -> snapshot vllm.log -> restart via w1_start_vllm_perf.sh (EAGER=1).
# Disable switch: /root/.vllm_watchdog_disabled exists -> probe-and-log only, never restart.
# Single instance: exits if another watchdog is already running.

VLLM_URL="http://127.0.0.1:8000/v1/chat/completions"
VLLM_LOG="/root/autodl-tmp/vllm.log"
WATCHDOG_LOG="/root/autodl-tmp/watchdog.log"
DISABLE_FLAG="/root/.vllm_watchdog_disabled"
START_SCRIPT="/root/scripts/w1_start_vllm_perf.sh"
MAX_FAILS=3
PROBE_TIMEOUT=30
PROBE_BODY='{"model":"qwen35b","messages":[{"role":"user","content":"ping"}],"max_tokens":2}'

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$WATCHDOG_LOG"
}

# single-instance guard: exit if another watchdog (exact cmdline) is running.
# pgrep -f also matches the wrapping bash -c of an SSH session, so compare cmdline exactly.
self_pid=$$
for pid in $(pgrep -f "scripts/vllm_watchdog.sh" || true); do
    [ "$pid" = "$self_pid" ] && continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    case "$cmdline" in
        "bash /root/scripts/vllm_watchdog.sh "|"bash /root/scripts/vllm_watchdog.sh")
            echo "watchdog already running (pid: $pid), exit" >> "$WATCHDOG_LOG"
            exit 0
            ;;
    esac
done

log "watchdog started (pid=$self_pid, max_fails=$MAX_FAILS, probe_timeout=${PROBE_TIMEOUT}s)"

fail_count=0
while true; do
    # /v1/models unreachable also counts as a failure: a fully dead vLLM must
    # still trigger the restart path (3 consecutive failed cycles -> restart).
    if ! curl -s -m 5 "http://127.0.0.1:8000/v1/models" > /dev/null 2>&1; then
        fail_count=$((fail_count + 1))
        log "probe cycle: FAIL ($fail_count/$MAX_FAILS) /v1/models unreachable"
    else
        resp=$(curl -s -m "$PROBE_TIMEOUT" -X POST "$VLLM_URL" \
            -H "Content-Type: application/json" \
            -d "$PROBE_BODY" 2>&1)
        if [ $? -eq 0 ] && echo "$resp" | grep -q '"choices"'; then
            fail_count=0
            log "probe cycle: OK"
        else
            fail_count=$((fail_count + 1))
            log "probe cycle: FAIL ($fail_count/$MAX_FAILS) resp_tail=$(echo "$resp" | tail -c 200 | tr '\n' ' ')"
        fi
    fi

    if [ "$fail_count" -ge "$MAX_FAILS" ]; then
        if [ -f "$DISABLE_FLAG" ]; then
            log "service DEAD detected but disabled flag present, probe-only mode, NOT restarting"
        else
            log "service DEAD ($fail_count consecutive fails), snapshot log and restart"
            cp "$VLLM_LOG" "$VLLM_LOG.dead.$(date +%H%M)" 2>> "$WATCHDOG_LOG"
            log "log snapshot saved, running: EAGER=1 bash $START_SCRIPT"
            EAGER=1 bash "$START_SCRIPT" >> "$WATCHDOG_LOG" 2>&1
            log "restart script finished, reset fail_count"
        fi
        fail_count=0
    fi
    sleep 60
done
