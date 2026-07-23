#!/usr/bin/env python3
"""从跳板用 kubectl 只读采集 pod→node 与 topology 相关 labels/annotations。

禁止输出 token/cert/kubeconfig 正文。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TOPOLOGY_KEY_RE = re.compile(
    r"(topology\.kubernetes\.io|rack|zone|hostname|region|failure-domain|"
    r"provider|datacenter|机房|network|sriov|multus|k8s\.v1\.cni|"
    r"v1\.multus|rdma|roce|nic|rail|switch|tor|leaf|spine|node-role|"
    r"kubernetes\.io/hostname|beta\.kubernetes\.io)",
    re.I,
)
SECRET_KEY_RE = re.compile(r"(token|cert|certificate|key|password|secret|kubeconfig)", re.I)


def run(cmd: list[str], env: dict[str, str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        err = proc.stderr.strip()
        if re.search(r"Forbidden|Unauthorized|permission denied|credentials|authentication", err, re.I):
            print(f"AUTH_ERROR: {err[:500]}", file=sys.stderr)
            raise SystemExit(13)
        raise RuntimeError(f"cmd_failed rc={proc.returncode}: {err[:800]}")
    return proc.stdout


def filter_kv(d: dict[str, Any] | None) -> dict[str, Any]:
    if not d:
        return {}
    out = {}
    for k, v in d.items():
        if SECRET_KEY_RE.search(k):
            out[k] = "<redacted>"
            continue
        if TOPOLOGY_KEY_RE.search(k) or TOPOLOGY_KEY_RE.search(str(v)[:200]):
            # 仍 redact 明显证书内容
            vs = str(v)
            if "BEGIN " in vs or "LS-AUTH" in vs or len(vs) > 2000:
                out[k] = f"<redacted len={len(vs)}>"
            else:
                out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="yinjinrun-cs512-20260716-221823")
    parser.add_argument("--kubectl", default="/root/.cache/volcano/kubectl/kubectl")
    parser.add_argument("--kubeconfig", default=os.environ.get("KUBECONFIG", ""))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    env = os.environ.copy()
    if args.kubeconfig:
        env["KUBECONFIG"] = args.kubeconfig
    # 身份确认：只打印 user 名
    view = run(
        [args.kubectl, "config", "view", "--minify", "-o", "json"],
        env,
    )
    cfg = json.loads(view)
    user = None
    try:
        ctx_name = cfg.get("current-context")
        ctx = next(c for c in cfg.get("contexts", []) if c.get("name") == ctx_name)
        user = ctx.get("context", {}).get("user")
    except StopIteration:
        user = None
    print(f"KUBECTL_USER={user}")
    if user and "yinjinrun.p" not in user:
        print(f"IDENTITY_MISMATCH expected yinjinrun.p got {user}", file=sys.stderr)
        raise SystemExit(13)

    # 列出 job pods
    pods_json = run(
        [
            args.kubectl,
            "get",
            "pods",
            "-o",
            "json",
            "-l",
            f"volcano.sh/job-name={args.job}",
        ],
        env,
    )
    # 若 label 不对，回退到名称前缀
    pods_obj = json.loads(pods_json)
    items = pods_obj.get("items") or []
    if len(items) < 64:
        pods_json = run([args.kubectl, "get", "pods", "-o", "json"], env)
        pods_obj = json.loads(pods_json)
        items = [
            p
            for p in pods_obj.get("items") or []
            if str(p.get("metadata", {}).get("name", "")).startswith(args.job + "-")
        ]
    if len(items) != 64:
        print(f"POD_COUNT={len(items)} expected 64", file=sys.stderr)
        if len(items) == 0:
            raise SystemExit(21)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pod_rows = []
    node_names = set()
    for pod in items:
        md = pod.get("metadata") or {}
        spec = pod.get("spec") or {}
        status = pod.get("status") or {}
        name = md.get("name")
        node = spec.get("nodeName")
        node_names.add(node)
        ann = filter_kv(md.get("annotations"))
        labels = filter_kv(md.get("labels"))
        # network attachment 常见注解全量保留（已 filter）
        resources = {}
        for c in spec.get("containers") or []:
            resources[c.get("name") or "container"] = {
                "requests": c.get("resources", {}).get("requests"),
                "limits": c.get("resources", {}).get("limits"),
            }
        pod_rows.append(
            {
                "pod": name,
                "node": node,
                "hostNetwork": bool(spec.get("hostNetwork")),
                "podIP": status.get("podIP"),
                "phase": status.get("phase"),
                "labels_filtered": labels,
                "annotations_filtered": ann,
                "resources": resources,
                "dnsPolicy": spec.get("dnsPolicy"),
                "nodeSelector": spec.get("nodeSelector"),
                "tolerations_count": len(spec.get("tolerations") or []),
            }
        )

    node_rows = []
    for node in sorted(n for n in node_names if n):
        nj = run([args.kubectl, "get", "node", node, "-o", "json"], env)
        nobj = json.loads(nj)
        nmd = nobj.get("metadata") or {}
        nstatus = nobj.get("status") or {}
        addresses = nstatus.get("addresses") or []
        node_rows.append(
            {
                "node": node,
                "labels_filtered": filter_kv(nmd.get("labels")),
                "annotations_filtered": filter_kv(nmd.get("annotations")),
                "addresses": [
                    {"type": a.get("type"), "address": a.get("address")} for a in addresses
                ],
                "capacity_keys": sorted((nstatus.get("capacity") or {}).keys()),
                "allocatable_keys": sorted((nstatus.get("allocatable") or {}).keys()),
            }
        )

    # annotation key 汇总
    pod_ann_keys: Counter[str] = Counter()
    node_label_keys: Counter[str] = Counter()
    node_ann_keys: Counter[str] = Counter()
    for p in pod_rows:
        pod_ann_keys.update(p["annotations_filtered"].keys())
    for n in node_rows:
        node_label_keys.update(n["labels_filtered"].keys())
        node_ann_keys.update(n["annotations_filtered"].keys())

    out = {
        "schema_version": "muxi.control_plane.v1",
        "job": args.job,
        "identity_user": user,
        "kubectl_path": args.kubectl,
        "kubeconfig_path": args.kubeconfig,
        "pod_count": len(pod_rows),
        "node_count": len(node_rows),
        "pods": pod_rows,
        "nodes": node_rows,
        "pod_annotation_keys": dict(pod_ann_keys),
        "node_label_keys": dict(node_label_keys),
        "node_annotation_keys": dict(node_ann_keys),
    }
    (args.out_dir / "control_plane.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    )
    with (args.out_dir / "pod_node_map.jsonl").open("w") as f:
        for row in sorted(pod_rows, key=lambda x: x["pod"] or ""):
            f.write(
                json.dumps(
                    {
                        "pod": row["pod"],
                        "node": row["node"],
                        "podIP": row["podIP"],
                        "hostNetwork": row["hostNetwork"],
                        "phase": row["phase"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        json.dumps(
            {
                "pods": len(pod_rows),
                "nodes": len(node_rows),
                "pod_annotation_keys": sorted(pod_ann_keys),
                "node_label_keys": sorted(node_label_keys)[:40],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
