# AMD (ROCm) enablement plan: vLLM-first

## Goals (Phase 0/1)
- **Target**: vLLM-only backend on AMD GPUs (ROCm), Dynamo frontend + runtime + vLLM engine.
- **Include RDMA**: enable RCCL-based RDMA paths for KV/embeddings transfer where supported.
- **Non-goals** (initial): TensorRT-LLM support, CUDA-specific tooling, multimodal encode workers on GPU.
- **Deliverables**:
  - ROCm-based container build for `vllm-runtime`.
  - Deployment scripts/manifests supporting AMD GPU scheduling.
  - Clear operator runbook for validation commands and known limitations.

## Current NVIDIA/CUDA coupling to replace
- **Container pipeline** uses NVIDIA NGC base images and CUDA toolchains.
- **Runtime scripts** assume `--runtime nvidia` and `--gpus` flags.
- **Deployment scripts** use `nvidia.com/*` GPU labels/resources and `nvidia-smi`.
- **Device selection in handlers** assumes `cuda` devices.

## Phase 1: vLLM-only AMD support (ROCm)

### 1) Container build (ROCm + RDMA)
**New artifacts**
- `container/Dockerfile.vllm.rocm` (new)
- `container/build.sh` additions to build a ROCm vLLM image
- `container/run.sh` additions to run with ROCm devices

**Base image choice**
- Use AMD ROCm base images (e.g., `rocm/pytorch` or `rocm/dev-ubuntu` + pip vLLM ROCm wheels).

**Key changes**
- Remove CUDA toolchain stages (nvcc, cuda libs) from ROCm Dockerfile.
- Install ROCm-enabled PyTorch + vLLM build (ROCm wheels or source build).
- Install RDMA stack with RCCL-enabled support, matching the current CUDA image behavior where possible.
- Ensure `/dev/kfd` and `/dev/dri` access inside container.
- Ensure RDMA device access (`/dev/infiniband/*`) and required capabilities when running the container.

**Example build targets**
- `dynamo:rocm-vllm-runtime`

### 2) Runtime entrypoint / run script
**Updates**
- Extend `container/run.sh` to support ROCm:
  - Default `RUNTIME=rocm` option or `--runtime rocm`.
  - Inject devices: `--device=/dev/kfd --device=/dev/dri`.
  - Add group permissions (video/render) where needed.
- Add RDMA device mounts when `--enable-rdma` (or equivalent flag) is set:
  - `--device=/dev/infiniband/rdma_cm --device=/dev/infiniband/uverbs0` (plus any required verbs devices).

**Example run**
```bash
./container/run.sh --image dynamo:rocm-vllm-runtime --runtime rocm \
  --gpus all -v $HOME/.cache:/home/dynamo/.cache
```

### 3) vLLM runtime configuration
**Backend wiring**
- Continue using `python -m dynamo.vllm` with ROCm-compatible vLLM.
- Avoid TensorRT-LLM references in ROCm images (skip TRTLLM layers).

**Device selection**
- Ensure vLLM config honors ROCm device visibility (typically `HIP_VISIBLE_DEVICES`).

**RCCL/RDMA configuration**
- Set RCCL transport discovery variables at runtime (environment or config):
  - Example: `RCCL_SOCKET_IFNAME=^lo,docker0` to select RDMA-capable NICs.
  - Example: `RCCL_NET_GDR_LEVEL=SYS` (or cluster-validated value) to enable GPU-direct RDMA.
- Ensure timeouts and channel configuration are tuned for multi-node runs (cluster-specific).

### 4) Deployment scripts for AMD + RDMA
**New/updated scripts**
- Add AMD-aware GPU inventory:
  - Replace `nvidia.com/*` label queries with AMD GPU labels (e.g., `amd.com/gpu` if using AMD device plugin).
  - Replace `nvidia-smi` probing with `rocm-smi`.
- Add RDMA capability detection and pod spec updates:
  - Ensure RDMA device plugin / SR-IOV resources are requested if required.
  - Update manifests to mount `/dev/infiniband` and allow required IPC settings.
  - Validate RCCL can discover RDMA-capable NICs and uses the correct transport.

**Kubernetes**
- Node labels and GPU resources must align with your AMD device plugin.
- Update sample manifests to request `amd.com/gpu: 1` (or your plugin’s resource name).
- Add environment variable wiring for RCCL (via pod env vars or config map).
- Prefer `hostNetwork: true` only if required by the RDMA/NIC topology and validated in cluster policy.

### 5) Validation & smoke tests (with RDMA)
**Host checks**
- `rocm-smi` on host
- `python -c "import torch; print(torch.version.hip); print(torch.cuda.is_available())"`
- RDMA connectivity checks (per cluster tooling): `ibv_devinfo`, `ibstat`, or vendor-provided diagnostics
- RCCL diagnostics (if available): `rccl-tests` or vendor test suite to validate RDMA transport

**Container checks**
- `rocminfo` (if available in image)
- `python -m dynamo.vllm --help`
- Small model load: `python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager`
- RDMA/RCCL check (if applicable): verify RCCL can discover NICs and RDMA transport is enabled

---

## Phase 2: Replace CUDA-specific utility paths
- **Multimodal encode workers**: replace hardcoded `cuda` device strings; detect HIP/ROCm.
- **RCCL/descriptor paths**: support ROCm device descriptors and RDMA paths (no CPU-only fallback for RDMA-enabled mode).
- **Sanity check**: add ROCm section; gate NVIDIA checks on presence of `nvidia-smi`.

---

## Phase 3: Expand backend support
- Evaluate SGLang ROCm
- Keep TensorRT-LLM disabled on AMD

---

## Proposed file changes (initial patch list)
1. `container/Dockerfile.vllm.rocm` (new)
2. `container/build.sh` (add `ROCM` framework target + build flags)
3. `container/run.sh` (add ROCm runtime options)
4. `deploy/utils/gpu_inventory.py` (AMD GPU detection)
5. `deploy/sanity_check.py` (ROCm info path)
6. `deploy` sample manifests (GPU resource name updates)
7. RDMA manifests/utilities (e.g., `deploy/utils` additions for RDMA plugins/resources and RCCL checks)

---

## What you can run with my outcome
Once we implement Phase 1, you’ll be able to:

### Build
```bash
./container/build.sh --framework VLLM --rocm
```

### Run (local)
```bash
./container/run.sh --image dynamo:rocm-vllm-runtime --runtime rocm \
  --gpus all -v $HOME/.cache:/home/dynamo/.cache
```

### Test vLLM
```bash
DYN_SYSTEM_PORT=8081 python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager
```

### Kubernetes deployment
```bash
# Example: request AMD GPU resource
resources:
  limits:
    amd.com/gpu: 1
  requests:
    amd.com/gpu: 1

# Example: RDMA resources (names depend on your RDMA device plugin)
resources:
  limits:
    amd.com/gpu: 1
    rdma/rdma_shared_device_a: 1
  requests:
    amd.com/gpu: 1
    rdma/rdma_shared_device_a: 1
```

---

## Open questions
- Which AMD device plugin resource name do you use? (common: `amd.com/gpu`)
- Which ROCm version is required on your cluster? (match ROCm PyTorch build)
- Which RDMA device plugin resources and naming conventions are standard in your cluster?
