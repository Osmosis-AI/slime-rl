#!/bin/bash
# One-time setup script for the limes CI/CD infrastructure on Together K8s
# Run this once from a machine with kubectl access to the cluster.
set -e

GITHUB_TOKEN=${GITHUB_TOKEN:?Set GITHUB_TOKEN to a fine-grained PAT with Actions+Administration read/write}

echo "==> Installing ARC controller..."
helm install arc \
  --namespace arc-systems \
  --create-namespace \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller

echo "==> Waiting for ARC controller to be ready..."
kubectl rollout status deployment/arc-gha-rs-controller -n arc-systems

echo "==> Creating namespaces and secrets..."
kubectl create namespace arc-runners --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic arc-github-secret \
  --namespace arc-runners \
  --from-literal=github_token="${GITHUB_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "==> Creating PVC for models and datasets..."
kubectl apply -f "$(dirname "$0")/pvc.yaml"

echo "==> Deploying limes-gpu-runners scale set..."
helm install limes-gpu-runners \
  --namespace arc-runners \
  -f "$(dirname "$0")/arc-runners-values.yaml" \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

echo ""
echo "==> Done! Verify runners registered at:"
echo "    https://github.com/Osmosis-AI/slime-rl/settings/actions/runners"
echo ""
echo "==> Next: pre-seed the PVC with models by running:"
echo "    kubectl apply -f infra/k8s/seed-models-job.yaml"
