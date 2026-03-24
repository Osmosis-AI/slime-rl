# Limes CI/CD Infrastructure

This directory contains the Kubernetes infrastructure for running GPU-accelerated CI on Together AI's cluster using GitHub Actions Runner Controller (ARC).

## Architecture Overview

```
GitHub PR / workflow_dispatch
        │
        ▼
GitHub Actions (pr-test.yml)
        │  job targets: limes-gpu-runners
        ▼
ARC Listener Pod (arc-systems namespace)
        │  detects job, scales runner set to 1
        ▼
Runner Pod (arc-runners namespace)
        │  2 containers: runner (ARC) + dind (Docker daemon)
        │  DinD mode: runner spawns slimerl/slime:latest as job container
        ▼
Job Container (slimerl/slime:latest)
        │  /root/models mounted from PVC
        │  downloads model on first run → cached in PVC
        │  runs pytest test file with full GPU access
        ▼
Results reported back to GitHub
```

## Files

| File | Purpose |
|------|---------|
| `pvc.yaml` | PersistentVolumeClaim for model storage (1Ti, `local` storage class) |
| `arc-runners-values.yaml` | Helm values for the ARC runner scale set |
| `setup.sh` | One-time cluster setup script |

## One-Time Setup

Run this once from a machine with `kubectl` access pointed at the Together cluster:

```bash
export GITHUB_TOKEN=<fine-grained PAT with Actions + Administration read/write on Osmosis-AI/slime-rl>
bash infra/k8s/setup.sh
```

This will:
1. Install the ARC controller in `arc-systems` namespace
2. Create the `arc-runners` namespace and GitHub token secret
3. Create the `limes-ci-pvc` PVC for model caching
4. Deploy the `limes-gpu-runners` scale set

Verify runners registered at: https://github.com/Osmosis-AI/slime-rl/settings/actions/runners

## How CI Works

### Triggering a job
Jobs run when a PR has one of these labels applied:

| Label | Jobs triggered |
|-------|---------------|
| `run-ci-short` | Small 0.5B model smoke tests (4 GPUs) |
| `run-ci-megatron` | Full Megatron training tests including Qwen3.5-35B-A3B (8 GPUs) |
| `run-ci-sglang-config` | SGLang config tests (8 GPUs) |
| `run-ci-precision` | Precision/parallel correctness tests (8 GPUs) |
| `run-ci-ckpt` | Checkpoint save/load tests (8 GPUs) |
| `run-ci-changed` | Auto-detects and runs only changed test files |
| `run-ci-plugin-contracts` | Plugin contract tests (CPU only, always runs on PRs) |

You can also trigger any job manually via **Actions → PR Test → Run workflow**.

### Model caching
The PVC (`limes-ci-pvc`, `local` storage class, 1Ti) is mounted at `/root/models` in the job container via `-v /root/models:/root/models` in the workflow's container options. The first CI run that needs a model will download it there (~30 min for Qwen3.5-35B-A3B at 70GB). All subsequent runs reuse the cached model — no re-download needed.

### Runner lifecycle
- The runner pod is **ephemeral** — it spins up when a job is queued, runs the job, then terminates
- `minRunners: 0` means no idle pods when there's no work
- `maxRunners: 1` — only 1 job runs at a time (matches 1 available node)

## Updating the Runner Configuration

If you need to change GPU count, image, memory, or mounts:

1. Edit `arc-runners-values.yaml`
2. Apply with Helm upgrade:

```bash
helm upgrade limes-gpu-runners \
  --namespace arc-runners \
  -f infra/k8s/arc-runners-values.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set
```

## Adding a New CI Test

1. Create a test file in `tests/` following the existing pattern (e.g. `test_qwen3_30B_A3B.py`)
2. Set `NUM_GPUS` at the top of the file — the `run-ci-changed` job reads this automatically
3. Add it to the appropriate job in `.github/workflows/pr-test.yml.j2`:

```python
'e2e-test-megatron': {
    'label': 'run-ci-megatron',
    'tests': [
        # add your test here
        {'test_file': 'test_your_new_test.py', 'num_gpus': 8},
    ],
},
```

4. Regenerate the workflow yaml:

```bash
pip install jinja2
python .github/workflows/generate_github_workflows.py
```

5. Commit both `pr-test.yml.j2` and `pr-test.yml`.

## Debugging

### Check runner pod status
```bash
kubectl get pods -n arc-runners
kubectl describe pod <runner-pod-name> -n arc-runners
```

### Check ARC listener logs
```bash
kubectl get pods -n arc-systems
kubectl logs <listener-pod-name> -n arc-systems | tail -30
```

### Check PVC status
```bash
kubectl get pvc -n arc-runners
```

### Common issues

**Runner pod stuck in `ContainerCreating`**
- Check events: `kubectl describe pod <pod> -n arc-runners | grep -A20 Events:`
- Usually a volume mount issue or image pull problem

**`no PriorityClass with name X was found`**
- Remove `priorityClassName` from `arc-runners-values.yaml` or use a valid class
- Check available classes: `kubectl get priorityclasses`

**PVC stuck in `Pending`**
- Normal if using `WaitForFirstConsumer` (`local` storage class) — it binds when the first pod is scheduled
- Apply a pod that uses the PVC to trigger binding

**Listener pod running but no runner pod created**
- Check ephemeral runner status: `kubectl get ephemeralrunners -n arc-runners`
- Error message will be in the `MESSAGE` column

**Runner pod completes in ~2 seconds without running any job**
- This means ARC DinD config is conflicting with manually defined containers
- Keep `arc-runners-values.yaml` minimal — only set `containerMode.type: "dind"` and extra `volumes`
- Do NOT manually define `containers`, `initContainers`, or `DOCKER_HOST` — ARC injects these automatically
- Do NOT add duplicate entries for `dind`, `init-dind-externals`, `dind-sock`, `dind-externals` volumes

**Runner pod cycles but job never executes**
- Cancel all queued workflows on GitHub first, then trigger a fresh run
- Stale queued jobs cause runners to register and immediately deregister

## GitHub Actions Secrets Required

These must be set in the repo at **Settings → Secrets and variables → Actions**:

| Secret | Purpose |
|--------|---------|
| `WANDB_API_KEY` | Weights & Biases logging during training runs |
