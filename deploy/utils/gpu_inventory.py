# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union

from kubernetes import client, config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def run_command(cmd: List[str], capture_output: bool = True, exit_on_error: bool = True):  # type: ignore
    try:
        return subprocess.run(cmd, capture_output=capture_output, text=True, check=True)
    except subprocess.CalledProcessError as e:  # pragma: no cover - passthrough
        if exit_on_error:
            logger.error(f"Command failed: {' '.join(cmd)}")
            if e.stdout:
                logger.error(e.stdout)
            if e.stderr:
                logger.error(e.stderr)
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
        raise


NVIDIA_PREFIX = "nvidia.com/"
LABEL_GPU_COUNT = f"{NVIDIA_PREFIX}gpu.count"
LABEL_GPU_PRODUCT = f"{NVIDIA_PREFIX}gpu.product"
LABEL_GPU_MEMORY = f"{NVIDIA_PREFIX}gpu.memory"  # MiB per GPU
LABEL_MIG_CAPABLE = f"{NVIDIA_PREFIX}mig.capable"

# AMD ROCm device plugin labels
AMD_PREFIX = "amd.com/"
LABEL_AMD_GPU_COUNT = f"{AMD_PREFIX}gpu.count"
LABEL_AMD_GPU_PRODUCT = f"{AMD_PREFIX}gpu.product"
LABEL_AMD_GPU_MEMORY = f"{AMD_PREFIX}gpu.memory"


@dataclass
class NodeGpuInventory:
    node_name: str
    gpu_vendor: Optional[str]  # "nvidia", "amd", or None
    gpu_count: Optional[int]
    gpu_product: Optional[str]
    gpu_memory_mib: Optional[int]
    mig_capable: Optional[bool]  # Only applicable for NVIDIA
    allocatable_gpu: Optional[int]
    mig_resources: Dict[str, str]  # Only applicable for NVIDIA

    def to_dict(self) -> Dict[str, Union[str, int, bool, Dict[str, str], None]]:
        return asdict(self)


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else None


def _bool_from_str(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _normalize_node(node: Union[client.V1Node, Dict]) -> Dict:
    # Convert V1Node to dict for uniform access
    if hasattr(node, "to_dict"):
        return node.to_dict()
    return node  # assume already dict


def _extract_inventory(node_obj: Dict) -> NodeGpuInventory:
    meta = node_obj.get("metadata", {})
    status = node_obj.get("status", {})
    labels = meta.get("labels", {}) or {}

    node_name = meta.get("name", "<unknown>")

    # Detect GPU vendor by checking allocatable resources
    gpu_vendor = None
    gpu_count = None
    gpu_product = None
    gpu_memory_mib = None
    mig_capable = None
    alloc_gpu = None
    mig_resources: Dict[str, str] = {}

    alloc = status.get("allocatable", {}) or {}

    # Check for NVIDIA GPUs first
    nvidia_gpu = _parse_int(alloc.get(f"{NVIDIA_PREFIX}gpu"))
    if nvidia_gpu and nvidia_gpu > 0:
        gpu_vendor = "nvidia"
        gpu_product = labels.get(LABEL_GPU_PRODUCT)
        gpu_memory_mib = _parse_int(labels.get(LABEL_GPU_MEMORY))
        mig_capable = _bool_from_str(labels.get(LABEL_MIG_CAPABLE))
        gpu_count = _parse_int(labels.get(LABEL_GPU_COUNT))
        alloc_gpu = nvidia_gpu
        if gpu_count is None:
            gpu_count = alloc_gpu
        # Collect MIG resources
        mig_resources = {
            k: str(v)
            for k, v in alloc.items()
            if isinstance(k, str)
            and k.startswith(f"{NVIDIA_PREFIX}mig-")
            and _parse_int(str(v))
        }

    # Check for AMD GPUs
    amd_gpu = _parse_int(alloc.get(f"{AMD_PREFIX}gpu"))
    if amd_gpu and amd_gpu > 0 and gpu_vendor is None:
        gpu_vendor = "amd"
        gpu_product = labels.get(LABEL_AMD_GPU_PRODUCT)
        gpu_memory_mib = _parse_int(labels.get(LABEL_AMD_GPU_MEMORY))
        gpu_count = _parse_int(labels.get(LABEL_AMD_GPU_COUNT))
        alloc_gpu = amd_gpu
        if gpu_count is None:
            gpu_count = alloc_gpu
        # AMD doesn't have MIG
        mig_capable = False
        mig_resources = {}

    return NodeGpuInventory(
        node_name=node_name,
        gpu_vendor=gpu_vendor,
        gpu_count=gpu_count,
        gpu_product=gpu_product,
        gpu_memory_mib=gpu_memory_mib,
        mig_capable=mig_capable,
        allocatable_gpu=alloc_gpu,
        mig_resources=mig_resources,
    )


def _list_nodes_via_client() -> List[Dict]:
    # Assume running inside a Kubernetes pod with service account
    try:
        config.load_incluster_config()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load in-cluster Kubernetes config. Ensure this runs in a pod with a service account. Error: {e}"
        )

    v1 = client.CoreV1Api()
    items = v1.list_node().items  # type: ignore[attr-defined]
    return [_normalize_node(n) for n in items]


def _list_nodes_via_kubectl() -> List[Dict]:
    if not shutil.which("kubectl"):
        raise RuntimeError("kubectl not found in PATH for fallback")
    result = run_command(["kubectl", "get", "nodes", "-o", "json"], capture_output=True)
    data = json.loads(result.stdout)
    return data.get("items", [])


def collect_gpu_inventory(
    prefer_client: bool = True,
) -> Tuple[List[NodeGpuInventory], str]:
    sources_tried: List[str] = []
    errors: List[str] = []

    def _via_client() -> List[NodeGpuInventory]:
        items = _list_nodes_via_client()
        return [_extract_inventory(n) for n in items]

    def _via_kubectl() -> List[NodeGpuInventory]:
        items = _list_nodes_via_kubectl()
        return [_extract_inventory(n) for n in items]

    if prefer_client:
        try:
            sources_tried.append("kubernetes-client")
            return _via_client(), ",".join(sources_tried)
        except Exception as e:
            errors.append(str(e))
            try:
                sources_tried.append("kubectl-json")
                return _via_kubectl(), ",".join(sources_tried)
            except Exception as e2:
                errors.append(str(e2))
                raise RuntimeError("Failed to list nodes: " + " | ".join(errors))
    else:
        try:
            sources_tried.append("kubectl-json")
            return _via_kubectl(), ",".join(sources_tried)
        except Exception as e:
            errors.append(str(e))
            try:
                sources_tried.append("kubernetes-client")
                return _via_client(), ",".join(sources_tried)
            except Exception as e2:
                errors.append(str(e2))
                raise RuntimeError("Failed to list nodes: " + " | ".join(errors))


def _format_gib(mib: Optional[int]) -> str:
    if mib is None:
        return ""
    return f"{mib/1024:.1f} GiB"


def print_table(rows: List[NodeGpuInventory], show_mig: bool = False) -> None:
    headers = ["NODE", "VENDOR", "GPUS", "MODEL", "VRAM/GPU", "MIG"]
    table: List[List[str]] = []
    for r in rows:
        mig_str = ""
        if r.gpu_vendor == "nvidia":
            if r.mig_capable is True:
                if r.mig_resources:
                    mig_str = ",".join(
                        f"{k.split('/')[-1]}={v}"
                        for k, v in sorted(r.mig_resources.items())
                    )
                else:
                    mig_str = "capable"
            elif r.mig_capable is False:
                mig_str = "no"
        else:
            mig_str = "n/a"  # AMD doesn't have MIG

        table.append(
            [
                r.node_name,
                r.gpu_vendor or "",
                "" if r.gpu_count is None else str(r.gpu_count),
                r.gpu_product or "",
                _format_gib(r.gpu_memory_mib),
                mig_str if show_mig else ("yes" if r.mig_capable else ""),
            ]
        )

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(row: List[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    logger.info(_fmt_row(headers))
    logger.info(_fmt_row(["-" * w for w in widths]))
    for row in table:
        logger.info(_fmt_row(row))


def aggregate_valued_rows(
    rows: List[NodeGpuInventory],
) -> Tuple[Optional[NodeGpuInventory], int]:
    """Aggregate rows that have meaningful GPU metadata.

    Preference order when multiple distinct values exist:
    1) Larger GPUs per node (gpu_count)
    2) Larger VRAM per GPU (gpu_memory_mib)
    Returns (selected_row_like, distinct_count).
    """
    valued: List[NodeGpuInventory] = [
        r for r in rows if (r.gpu_product is not None or r.gpu_memory_mib is not None)
    ]
    if not valued:
        return None, 0

    # Group by (vendor, product, vram_mib)
    from collections import defaultdict

    groups: Dict[
        Tuple[Optional[str], Optional[str], Optional[int]],
        Dict[str, object],
    ] = defaultdict(lambda: {"max_gpu": 0, "rows": []})
    for r in valued:
        key = (r.gpu_vendor, r.gpu_product, r.gpu_memory_mib)
        meta = groups[key]
        meta["rows"].append(r)  # type: ignore[attr-defined, index]
        # Use known gpu_count if available for ranking
        if r.gpu_count is not None:
            meta["max_gpu"] = max(int(meta["max_gpu"]), int(r.gpu_count))  # type: ignore[arg-type, call-overload, index]

    def sort_key(
        item: Tuple[
            Tuple[Optional[str], Optional[str], Optional[int]],
            Dict[str, object],
        ],
    ):
        (vendor, prod, mem_mib), meta = item
        max_gpu = int(meta["max_gpu"])  # type: ignore[arg-type, call-overload, index]
        mem_val = mem_mib if mem_mib is not None else -1
        return (max_gpu, mem_val)

    selected_key, selected_meta = sorted(groups.items(), key=sort_key, reverse=True)[0]
    sel_vendor, sel_prod, sel_mem_mib = selected_key
    sel_gpu = int(selected_meta["max_gpu"])  # type: ignore[arg-type, call-overload, index]

    selected = NodeGpuInventory(
        node_name="<aggregate>",
        gpu_vendor=sel_vendor,
        gpu_count=sel_gpu if sel_gpu > 0 else None,
        gpu_product=sel_prod,
        gpu_memory_mib=sel_mem_mib,
        mig_capable=None,
        allocatable_gpu=None,
        mig_resources={},
    )

    return selected, len(groups)


def _get_current_namespace(default: str = "default") -> str:
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            return f.read().strip() or default
    except Exception:
        return default


def enrich_with_smi(
    rows: List[NodeGpuInventory],
    namespace: Optional[str] = None,
    timeout_seconds: int = 180,
) -> None:
    """For nodes missing product/memory labels, schedule a short-lived pod on each node
    that requests 1 GPU and runs nvidia-smi to capture model and memory.

    Requires permissions: create/get/delete pods and get pods/log in the namespace.
    """
    ns = namespace or _get_current_namespace()
    try:
        config.load_incluster_config()
    except Exception:
        pass

    v1 = client.CoreV1Api()

    for inv in rows:
        if not inv.gpu_count or (
            inv.gpu_product is not None and inv.gpu_memory_mib is not None
        ):
            continue

        pod_name = f"gpu-inv-smi-{uuid.uuid4().hex[:6]}"
        container = client.V1Container(
            name="smi",
            image="nvidia/cuda:12.3.2-base-ubuntu22.04",
            command=["bash", "-lc"],
            args=[
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits"
            ],
            resources=client.V1ResourceRequirements(
                limits={"nvidia.com/gpu": "1", "cpu": "100m", "memory": "128Mi"},
                requests={"nvidia.com/gpu": "1", "cpu": "50m", "memory": "64Mi"},
            ),
        )

        pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=client.V1ObjectMeta(name=pod_name, namespace=ns),
            spec=client.V1PodSpec(
                restart_policy="Never",
                node_name=inv.node_name,
                containers=[container],
            ),
        )

        logs = ""
        try:
            v1.create_namespaced_pod(namespace=ns, body=pod)
            start = time.time()
            while time.time() - start < timeout_seconds:
                p = v1.read_namespaced_pod(name=pod_name, namespace=ns)
                phase = (p.status.phase or "").lower()
                if phase in ("succeeded", "failed"):
                    break
                time.sleep(2)
            try:
                logs = v1.read_namespaced_pod_log(name=pod_name, namespace=ns)
            except Exception:
                logs = ""
        finally:
            try:
                v1.delete_namespaced_pod(
                    name=pod_name, namespace=ns, body=client.V1DeleteOptions()
                )
            except Exception:
                pass

        for line in logs.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2 and parts[0]:
                inv.gpu_product = inv.gpu_product or parts[0]
                mem_match = re.search(r"\d+", parts[1])
                if mem_match:
                    inv.gpu_memory_mib = inv.gpu_memory_mib or int(mem_match.group(0))
                break


def enrich_with_rocm_smi(
    rows: List[NodeGpuInventory],
    namespace: Optional[str] = None,
    timeout_seconds: int = 180,
) -> None:
    """For AMD GPU nodes missing product/memory labels, schedule a short-lived pod
    that runs rocm-smi to capture model and memory.

    Requires permissions: create/get/delete pods and get pods/log in the namespace.
    """
    ns = namespace or _get_current_namespace()
    try:
        config.load_incluster_config()
    except Exception:
        pass

    v1 = client.CoreV1Api()

    for inv in rows:
        # Only process AMD GPU nodes missing metadata
        if inv.gpu_vendor != "amd":
            continue
        if not inv.gpu_count or (
            inv.gpu_product is not None and inv.gpu_memory_mib is not None
        ):
            continue

        pod_name = f"gpu-inv-rocm-{uuid.uuid4().hex[:6]}"
        container = client.V1Container(
            name="rocm-smi",
            image="rocm/rocm-terminal:latest",
            command=["bash", "-lc"],
            args=[
                # rocm-smi --showproductname shows GPU model
                # rocm-smi --showmeminfo vram shows VRAM
                "rocm-smi --showproductname --csv | tail -1 && rocm-smi --showmeminfo vram --csv | grep -v 'device\\|GPU' | head -1"
            ],
            resources=client.V1ResourceRequirements(
                limits={"amd.com/gpu": "1", "cpu": "100m", "memory": "128Mi"},
                requests={"amd.com/gpu": "1", "cpu": "50m", "memory": "64Mi"},
            ),
            security_context=client.V1SecurityContext(
                capabilities=client.V1Capabilities(add=["SYS_PTRACE"])
            ),
        )

        pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=client.V1ObjectMeta(name=pod_name, namespace=ns),
            spec=client.V1PodSpec(
                restart_policy="Never",
                node_name=inv.node_name,
                containers=[container],
            ),
        )

        logs = ""
        try:
            v1.create_namespaced_pod(namespace=ns, body=pod)
            start = time.time()
            while time.time() - start < timeout_seconds:
                p = v1.read_namespaced_pod(name=pod_name, namespace=ns)
                phase = (p.status.phase or "").lower()
                if phase in ("succeeded", "failed"):
                    break
                time.sleep(2)
            try:
                logs = v1.read_namespaced_pod_log(name=pod_name, namespace=ns)
            except Exception:
                logs = ""
        finally:
            try:
                v1.delete_namespaced_pod(
                    name=pod_name, namespace=ns, body=client.V1DeleteOptions()
                )
            except Exception:
                pass

        # Parse rocm-smi output
        # Example output format varies, but typically:
        # Card series: AMD Instinct MI300X
        # VRAM Total: 196608 MB (or similar)
        for line in logs.splitlines():
            line = line.strip()
            # Try to extract GPU model
            if "instinct" in line.lower() or "radeon" in line.lower():
                inv.gpu_product = inv.gpu_product or line.strip()
            # Try to extract VRAM (in MB)
            if "vram" in line.lower() or re.search(
                r"\d+\s*(MB|MiB|GB|GiB)", line, re.I
            ):
                mem_match = re.search(r"(\d+)\s*(MB|MiB|GB|GiB)?", line, re.I)
                if mem_match:
                    mem_val = int(mem_match.group(1))
                    unit = (mem_match.group(2) or "MB").upper()
                    if "GB" in unit or "GIB" in unit:
                        mem_val = mem_val * 1024
                    inv.gpu_memory_mib = inv.gpu_memory_mib or mem_val


def enrich_gpu_inventory(
    rows: List[NodeGpuInventory],
    namespace: Optional[str] = None,
    timeout_seconds: int = 180,
) -> None:
    """Enrich GPU inventory for both NVIDIA and AMD GPUs."""
    nvidia_rows = [r for r in rows if r.gpu_vendor == "nvidia"]
    amd_rows = [r for r in rows if r.gpu_vendor == "amd"]

    if nvidia_rows:
        enrich_with_smi(nvidia_rows, namespace, timeout_seconds)
    if amd_rows:
        enrich_with_rocm_smi(amd_rows, namespace, timeout_seconds)


def get_gpu_summary(
    prefer_client: bool = True, enrich_smi: bool = True
) -> Dict[str, object]:
    """Return an aggregate GPU summary for the cluster.

    Selection policy when multiple values exist: prefer higher GPUs per node,
    then higher VRAM/GPU. Returns dict with keys: gpus_per_node, model, vram, vendor.
    If model/VRAM unavailable anywhere, returns {"gpus_per_node": max_gpus, "model": "", "vram": 0, "vendor": ""}.
    """
    # TODO: use proper tools (i.e., DCGM) to get GPU inventory
    rows, _ = collect_gpu_inventory(prefer_client=prefer_client)
    if enrich_smi:
        enrich_gpu_inventory(rows)

    agg, _distinct = aggregate_valued_rows(rows)
    if agg is None:
        # Fallback to max GPUs only
        max_gpus = 0
        for r in rows:
            if r.gpu_count is not None:
                max_gpus = max(max_gpus, int(r.gpu_count))
        return {"gpus_per_node": max_gpus, "model": "", "vram": 0, "vendor": ""}

    gpus_val = int(agg.gpu_count) if agg.gpu_count is not None else 0
    model_val = agg.gpu_product or ""
    vram_val = int(agg.gpu_memory_mib) if agg.gpu_memory_mib is not None else 0
    vendor_val = agg.gpu_vendor or ""
    return {
        "gpus_per_node": gpus_val,
        "model": model_val,
        "vram": vram_val,
        "vendor": vendor_val,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report GPU inventory per Kubernetes node (count, SKU, VRAM)."
    )
    parser.add_argument(
        "--format",
        "-o",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )
    parser.add_argument(
        "--prefer",
        choices=["client", "kubectl"],
        default="client",
        help="Prefer Kubernetes Python client or kubectl JSON fallback",
    )
    parser.add_argument(
        "--show-mig",
        action="store_true",
        help="In table output, show MIG resource types and counts",
    )
    parser.add_argument(
        "--enrich-smi",
        action="store_true",
        help="Schedule short-lived pods per node to fetch model/VRAM via nvidia-smi (NVIDIA) or rocm-smi (AMD)",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Print a single representative (GPUs per node, MODEL, VRAM/GPU). Warn if multiple values exist",
    )

    args = parser.parse_args()

    prefer_client = args.prefer == "client"
    rows, source = collect_gpu_inventory(prefer_client=prefer_client)

    if args.enrich_smi:
        enrich_gpu_inventory(rows)

    if args.format == "json":
        payload = {
            "source": source,
            "items": [r.to_dict() for r in rows],
        }
        logger.info(json.dumps(payload, indent=2))
        return

    # Table output
    print_table(rows, show_mig=args.show_mig)

    if args.aggregate:
        agg, distinct = aggregate_valued_rows(rows)
        if agg is None:
            logger.warning("No nodes expose MODEL/VRAM; cannot aggregate")
            return
        if distinct > 1:
            logger.warning(
                f"Multiple distinct GPU model/VRAM pairs detected across nodes: {distinct}. Showing highest GPUs per node, then highest VRAM/GPU."
            )
        # Print concise aggregate line
        model = agg.gpu_product or ""
        vram = _format_gib(agg.gpu_memory_mib)
        gpus = agg.gpu_count if agg.gpu_count is not None else ""
        logger.info(f"Aggregate => GPUS: {gpus}  MODEL: {model}  VRAM/GPU: {vram}")


if __name__ == "__main__":
    main()
