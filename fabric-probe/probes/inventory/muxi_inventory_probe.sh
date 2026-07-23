#!/usr/bin/env bash
# 沐曦只读 inventory probe：输出稳定的 TSV 字段，并保留原始命令段。
set -uo pipefail

POD="${1:?pod name required}"
required_missing=0

emit_meta() { printf 'META\t%s\t%s\n' "$1" "$2"; }
emit_rail() { printf 'RAIL\t%s\t%s\t%s\n' "$1" "$2" "$3"; }

emit_meta schema_version muxi.inventory.v1
emit_meta pod "$POD"
emit_meta host "$(hostname 2>/dev/null || true)"
emit_meta pod_ip "$(hostname -i 2>/dev/null | awk '{print $1}')"
emit_meta collected_at "$(date -Iseconds)"

for tool in lldpctl lldpd ip ethtool traceroute; do
  if command -v "$tool" >/dev/null 2>&1; then
    emit_meta "tool_${tool}" available_not_invoked
  else
    emit_meta "tool_${tool}" unavailable
  fi
done

for dev in xscale_0 xscale_1 xscale_2 xscale_3; do
  root="/sys/class/infiniband/$dev"
  if [[ ! -d "$root" ]]; then
    emit_rail "$dev" present false
    required_missing=1
    continue
  fi
  emit_rail "$dev" present true
  device_path="$(readlink -f "$root/device" 2>/dev/null || true)"
  emit_rail "$dev" pci_path "$device_path"
  emit_rail "$dev" pci_bdf "$(basename "$device_path")"
  for field in state phys_state rate active_mtu; do
    value="$(cat "$root/ports/1/$field" 2>/dev/null || true)"
    source=sysfs
    if [[ "$field" == "active_mtu" && -z "$value" ]] &&
       command -v ibv_devinfo >/dev/null 2>&1; then
      value="$(ibv_devinfo -d "$dev" 2>/dev/null | awk -F: '/^[[:space:]]*active_mtu:/ {sub(/^[[:space:]]+/,"",$2); print $2; exit}')"
      source=ibv_devinfo
    fi
    emit_rail "$dev" "$field" "$value"
    [[ "$field" == "active_mtu" ]] && emit_rail "$dev" active_mtu_source "$source"
  done
  gid="$(cat "$root/ports/1/gids/5" 2>/dev/null || true)"
  emit_rail "$dev" gid_index5 "$gid"
  ndev_file="$root/ports/1/gid_attrs/ndevs/5"
  ndev="$(cat "$ndev_file" 2>/dev/null || true)"
  emit_rail "$dev" gid_index5_netdev "$ndev"
  netdevs=""
  if [[ -d "$root/device/net" ]]; then
    for netpath in "$root"/device/net/*; do
      [[ -e "$netpath" ]] || continue
      netdevs="${netdevs:+$netdevs,}$(basename "$netpath")"
    done
  fi
  emit_rail "$dev" device_netdevs "$netdevs"
done

echo 'SECTION	proc_net_dev	BEGIN'
if [[ -r /proc/net/dev ]]; then
  awk 'NR>2 {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); print}' /proc/net/dev
else
  echo UNAVAILABLE
fi
echo 'SECTION	proc_net_dev	END'

echo 'SECTION	mx_smi_topo_n	BEGIN'
if command -v mx-smi >/dev/null 2>&1; then
  mx-smi topo -n 2>&1
else
  echo UNAVAILABLE
  required_missing=1
fi
echo 'SECTION	mx_smi_topo_n	END'

echo 'SECTION	ibv_devinfo	BEGIN'
if command -v ibv_devinfo >/dev/null 2>&1; then
  ibv_devinfo 2>&1
else
  echo UNAVAILABLE
fi
echo 'SECTION	ibv_devinfo	END'

if [[ "$required_missing" -ne 0 ]]; then
  echo 'PROBE_STATUS	FAILED_REQUIRED_FIELD'
  exit 22
fi
echo 'PROBE_STATUS	OK'
