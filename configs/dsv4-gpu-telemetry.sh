#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

hostname_short="$(hostname -s)"
lock_dir="/logs/.gpu-telemetry-${hostname_short}.lock"
if ! mkdir "${lock_dir}" 2>/dev/null; then
    exit 0
fi

output="/logs/gpu-telemetry-${hostname_short}.csv"
throttle_output="/logs/gpu-throttle-${hostname_short}.csv"
(
    printf '%s\n' 'timestamp,index,uuid,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu' > "${output}"
    printf '%s\n' 'timestamp,index,uuid,sw_power_cap,hw_slowdown,hw_thermal_slowdown,sw_thermal_slowdown' > "${throttle_output}"
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu \
            --format=csv,noheader,nounits >> "${output}" 2>/dev/null || true
        nvidia-smi \
            --query-gpu=timestamp,index,uuid,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown \
            --format=csv,noheader,nounits >> "${throttle_output}" 2>/dev/null || true
        sleep 1
    done
) &
disown
