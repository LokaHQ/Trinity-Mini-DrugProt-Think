<p align="center">
  <img src="assets/logo.png" alt="Trinity-Mini-DrugProt-Think" width="350" />
</p>

<p align="center">
  <strong>Trinity-Mini-DrugProt-Think</strong><br/>
  RLVR (GRPO) + LoRA post-training on Arcee Trinity Mini for DrugProt relation classification.
</p>

<p align="center">
  <a href="index.html">📝 <strong>Report</strong></a> &nbsp; | &nbsp;
  <a href="https://medium.com/@jakimovski_bojan/9e1c1c430ce9">
    <img
      src="https://www.sysgroup.com/wp-content/uploads/2025/02/Amazon_Web_Services-Logo.wine_.png"
      alt="AWS"
      height="18"
      style="vertical-align: middle;"
    />
    <strong>AWS deployment guide</strong>
  </a>
  &nbsp; | &nbsp;
  <a href="https://huggingface.co/lokahq/Trinity-Mini-DrugProt-Think">
    <img
      src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg"
      alt="Hugging Face"
      height="18"
      style="vertical-align: middle;"
    />
    <strong>Model</strong>
  </a>
</p>

This repo contains two distinct tracks:

1. **Experiments**: RL training configs and artifacts for DrugProt.
2. **Serving**: Deploy a Hugging Face base model + LoRA adapter to a SageMaker real-time endpoint (SageMaker SDK v3).

## Links

- [`uv`](https://docs.astral.sh/uv/) (Python project + tool runner)
- [SageMaker Python SDK v3](https://github.com/aws/sagemaker-python-sdk) (`ModelBuilder`, `InferenceSpec`; see [`v3-examples/`](https://github.com/aws/sagemaker-python-sdk/tree/master/v3-examples))
- [Prime Intellect](https://www.primeintellect.ai/) (Prime CLI + RL runs; docs: https://docs.primeintellect.ai/)
- [Weights & Biases](https://wandb.ai/site) (experiment tracking)
- [Hugging Face PEFT](https://github.com/huggingface/peft) (LoRA)

## Repo layout

- `experiments/configs/rl/`: RL experiment configs (`*.toml`).
- `index.html`: blog post (static HTML + Chart.js, serves from repo root).
- `data/`: exported metrics CSVs used by the blog post charts.
- `experiments/reports/`: supplementary writeups (deployment guide, etc.).
- `serving/lora_inference/`: SageMaker v3 `InferenceSpec` + container requirements.
- `serving/scripts/`: deploy/test/delete endpoint scripts.

## Experiments

### Prerequisites

1. Prime CLI installed (`prime --version`), e.g. `uv tool install prime`.
2. Logged in (`prime login`).
3. W&B API key available.

### Secrets

Configs use `env_files = ["secrets.env"]`, so put the secrets file next to the configs:

- `experiments/configs/rl/secrets.env` (gitignored)

```env
WANDB_API_KEY=your_key_here
```

### Run a baseline

- Baseline config: `experiments/configs/rl/w1_alpha16_baseline.toml`

```bash
prime rl run experiments/configs/rl/w1_alpha16_baseline.toml
```

### Monitor a run

```bash
prime rl progress <run_id>
prime rl logs <run_id> -f
prime rl metrics <run_id>
prime rl distributions <run_id> --type rewards
prime rl rollouts <run_id> --step <step> -n 50
```

## Serving (SageMaker real-time endpoint)

Deploy a Hugging Face base model with a LoRA adapter to a SageMaker real-time endpoint using SageMaker SDK v3 (`ModelBuilder` + `InferenceSpec`).

### What’s implemented

- `serving/lora_inference/spec.py`: loads base model + LoRA adapter and runs generation.
- `serving/lora_inference/requirements.txt`: container-time inference dependencies.
- `serving/scripts/deploy_lora_endpoint.py`: deploy/update endpoint.
- `serving/scripts/test_lora_endpoint.py`: invoke endpoint.
- `serving/scripts/delete_lora_endpoint.py`: delete endpoint.

### Prerequisites

- AWS credentials configured locally (`~/.aws/credentials` or env vars).
- SageMaker execution role ARN with model/endpoint permissions.
- Optional (private Hugging Face repos): `HF_TOKEN` env var.

### Install local dependencies

```bash
uv sync
```

### Deploy from a Hugging Face adapter repo

```bash
export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account-id>:role/<sagemaker-role>"
export HF_TOKEN="hf_xxx"   # optional for private repos

uv run python serving/scripts/deploy_lora_endpoint.py \
  --endpoint-name trinity-mini-drugprot-think \
  --adapter-id lokahq/Trinity-Mini-DrugProt-Think \
  --role-arn "$SAGEMAKER_ROLE_ARN" \
  --instance-type ml.g5.2xlarge
```

### Deploy from a local adapter directory

```bash
uv run python serving/scripts/deploy_lora_endpoint.py \
  --endpoint-name trinity-mini-drugprot-think-local \
  --adapter-id ./adapter \
  --role-arn "$SAGEMAKER_ROLE_ARN" \
  --instance-type ml.g5.2xlarge
```

If `./adapter/adapter_config.json` exists, the server will resolve the base model from the adapter metadata. You can always override with `--base-model-id <hf-model-id>`.

### Update an existing endpoint

```bash
uv run python serving/scripts/deploy_lora_endpoint.py \
  --endpoint-name trinity-mini-drugprot-think \
  --adapter-id lokahq/Trinity-Mini-DrugProt-Think \
  --role-arn "$SAGEMAKER_ROLE_ARN" \
  --update-endpoint
```

### Test invocation

```bash
uv run python serving/scripts/test_lora_endpoint.py \
  --endpoint-name trinity-mini-drugprot-think \
  --prompt "Give me one practical use-case of LoRA adapters in production." \
  --max-new-tokens 120 \
  --temperature 0.7 \
  --top-p 0.95
```

### Payload contract

Request JSON:

```json
{
  "inputs": "string prompt",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.95,
  "do_sample": true
}
```

Response JSON:

```json
{
  "generated_text": "...",
  "full_text": "...",
  "model_id": "<hf-base-model-id>",
  "adapter_id": "<adapter path or repo id>",
  "model_name": "Trinity-Mini-DrugProt-Think"
}
```

## Acknowledgements

- **Model:** [Arcee AI](https://www.arcee.ai) (with [Prime Intellect](https://www.primeintellect.ai) and [Datalogy](https://datalogy.ai)) for releasing the Trinity family.
- **Training:** [Prime Intellect](https://www.primeintellect.ai) for hosted training infrastructure.
- **Environment:** [OpenMed](https://huggingface.co/datasets/OpenMed/drugprot-parquet) for DrugProt dataset packaging.
- **Deployment:** [AWS](https://aws.amazon.com/) for deployment and hosting.
