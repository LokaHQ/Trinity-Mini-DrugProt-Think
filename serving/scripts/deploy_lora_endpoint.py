"""Deploy a base model + LoRA adapter using SageMaker SDK v3 ModelBuilder."""

import argparse
import logging
import os
import shutil
import site
import sys
import tempfile
from pathlib import Path

from sagemaker.core.training.configs import SourceCode
from sagemaker.core.helper.session_helper import Session as SageMakerSession
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer


LOGGER = logging.getLogger(__name__)


def _find_repo_root(start: Path) -> Path:
    """
    Find repository root by walking upward for `pyproject.toml` (preferred) or `.git`.

    Args:
        start: Starting file/directory path.

    Returns:
        Path: Repo root directory.

    Raises:
        RuntimeError: If no repo root markers are found.
    """
    start_dir = start if start.is_dir() else start.parent
    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Unable to locate repo root from: {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
SERVING_ROOT = REPO_ROOT / "serving"
if str(SERVING_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVING_ROOT))

from lora_inference.spec import ArceeLoraInferenceSpec  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Deploy base model + LoRA adapter with SDK v3.")
    parser.add_argument("--endpoint-name", required=True, help="SageMaker endpoint name.")
    parser.add_argument("--model-name", default=None, help="SageMaker model name. Defaults to <endpoint>-model.")
    parser.add_argument(
        "--adapter-id",
        default="adapter",
        help="LoRA adapter reference: local directory path or Hugging Face repo id.",
    )
    parser.add_argument("--base-model-id", default="arcee-ai/Trinity-Mini")
    parser.add_argument(
        "--model-display-name",
        default=None,
        help=(
            "Human-readable model name returned in responses as `model_name`. "
            "Defaults to $MODEL_NAME, else 'Trinity-Mini-DrugProt-Think'."
        ),
    )
    parser.add_argument(
        "--role-arn",
        default=os.getenv("SAGEMAKER_ROLE_ARN"),
        help="SageMaker execution role ARN. Defaults to $SAGEMAKER_ROLE_ARN.",
    )
    parser.add_argument("--region", default=None, help="AWS region override.")
    parser.add_argument(
        "--image-uri",
        default=(
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
            "huggingface-pytorch-inference:2.6.0-transformers4.51.3-gpu-py312-cu124-ubuntu22.04"
        ),
        help="Serving image URI. requirements.txt upgrades transformers at runtime.",
    )
    parser.add_argument("--instance-type", default="ml.p4d.24xlarge")
    parser.add_argument("--initial-instance-count", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--model-server-timeout", type=int, default=900)
    parser.add_argument("--wait", action="store_true", default=True)
    parser.add_argument("--no-wait", dest="wait", action="store_false")
    parser.add_argument("--update-endpoint", action="store_true")
    return parser


def _prepare_model_path_and_adapter_ref(adapter_id: str) -> tuple[str, str, str | None]:
    """Prepare model path and adapter env reference.

    For local adapters, copy contents into model artifacts under `adapter/`.
    For remote adapters, keep model artifacts empty and pass repo id directly.

    Args:
        adapter_id: Local path or Hugging Face adapter repo id.

    Returns:
        tuple[str, str, str | None]: (model_path, adapter_env_ref, cleanup_dir)
    """
    candidate = Path(adapter_id).expanduser()
    if candidate.exists():
        src = candidate.resolve()
        if not src.is_dir():
            raise ValueError(f"Local adapter path exists but is not a directory: {src}")

        staging_root = Path(tempfile.mkdtemp(prefix="sagemaker-lora-model-"))
        dst = staging_root / "adapter"
        shutil.copytree(src, dst)
        LOGGER.info("Staged local adapter from %s to %s", src, dst)
        return str(staging_root), "adapter", str(staging_root)

    # Remote adapter path (e.g. Hugging Face repo id)
    staging_root = Path(tempfile.mkdtemp(prefix="sagemaker-model-empty-"))
    LOGGER.info("Using remote adapter reference: %s", adapter_id)
    return str(staging_root), adapter_id, str(staging_root)


def _build_env(args: argparse.Namespace, adapter_env_ref: str) -> dict[str, str]:
    """Build environment variables for serving container."""
    model_display_name = args.model_display_name or os.getenv("MODEL_NAME") or "Trinity-Mini-DrugProt-Think"
    env = {
        "ADAPTER_ID": adapter_env_ref,
        "BASE_MODEL_ID": args.base_model_id,
        "MODEL_NAME": model_display_name,
        "HF_TASK": "text-generation",
        "DTYPE": args.dtype,
        "DEVICE_MAP": args.device_map,
        "MAX_NEW_TOKENS_DEFAULT": "256",
        "TEMPERATURE_DEFAULT": "0.7",
        "TOP_P_DEFAULT": "0.95",
        "DO_SAMPLE_DEFAULT": "true",
        "SAGEMAKER_MODEL_SERVER_TIMEOUT": str(args.model_server_timeout),
        "MMS_DEFAULT_RESPONSE_TIMEOUT": str(args.model_server_timeout),
        "MMS_DEFAULT_MODEL_LOADING_TIMEOUT": str(args.model_server_timeout),
    }
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token
    return env


def _install_local_module_for_build() -> None:
    """Install local lora_inference module into site-packages for SDK subprocesses."""
    src_pkg = SERVING_ROOT / "lora_inference"
    if not src_pkg.exists():
        raise FileNotFoundError(f"Expected package at {src_pkg}")
    site_pkg = Path(site.getsitepackages()[0]) / "lora_inference"
    shutil.copytree(src_pkg, site_pkg, dirs_exist_ok=True)


def _apply_modelbuilder_repack_workaround(model_builder: ModelBuilder) -> None:
    """Patch SDK v3 repack path bug for local model_path + source_code."""
    original_upload_code = model_builder._upload_code

    def _patched_upload_code(key_prefix: str, repack: bool = False) -> None:
        s3_model_data_url = getattr(model_builder, "s3_model_data_url", None)
        s3_upload_path = getattr(model_builder, "s3_upload_path", None)
        if (
            repack
            and isinstance(s3_model_data_url, str)
            and s3_model_data_url.endswith("/")
            and isinstance(s3_upload_path, str)
            and s3_upload_path.endswith(".tar.gz")
        ):
            model_builder.s3_model_data_url = s3_upload_path
        return original_upload_code(key_prefix, repack)

    model_builder._upload_code = _patched_upload_code


def main() -> None:
    """Run deployment."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args()

    if not args.role_arn:
        raise SystemExit("Missing --role-arn (or set SAGEMAKER_ROLE_ARN).")

    if args.region:
        os.environ["AWS_DEFAULT_REGION"] = args.region
        os.environ["AWS_REGION"] = args.region

    cleanup_dir: str | None = None
    previous_env: dict[str, str | None] = {}

    try:
        model_path, adapter_env_ref, cleanup_dir = _prepare_model_path_and_adapter_ref(args.adapter_id)
        env_vars = _build_env(args, adapter_env_ref)
        _install_local_module_for_build()

        # Keep build-time behavior aligned with runtime for inference_spec resolution.
        for key, value in env_vars.items():
            previous_env[key] = os.environ.get(key)
            os.environ[key] = value

        source_code = SourceCode(
            source_dir=str(INFERENCE_SRC := SERVING_ROOT / "lora_inference"),
            entry_script="spec.py",
            requirements="requirements.txt",
        )

        session = SageMakerSession()
        model_builder = ModelBuilder(
            inference_spec=ArceeLoraInferenceSpec(),
            source_code=source_code,
            role_arn=args.role_arn,
            sagemaker_session=session,
            image_uri=args.image_uri,
            model_server=ModelServer.TORCHSERVE,
            instance_type=args.instance_type,
            model_path=model_path,
            env_vars=env_vars,
            dependencies=[],
            content_type="application/json",
            accept_type="application/json",
        )
        _apply_modelbuilder_repack_workaround(model_builder)

        model_name = args.model_name or f"{args.endpoint_name}-model"
        LOGGER.info("Building model: %s", model_name)
        model_builder.build(model_name=model_name)

        LOGGER.info("Deploying endpoint: %s", args.endpoint_name)
        endpoint = model_builder.deploy(
            endpoint_name=args.endpoint_name,
            initial_instance_count=args.initial_instance_count,
            instance_type=args.instance_type,
            wait=args.wait,
            update_endpoint=args.update_endpoint,
            container_timeout_in_seconds=args.model_server_timeout,
        )

        endpoint_name = getattr(endpoint, "endpoint_name", args.endpoint_name)
        print("Deployment request submitted successfully.")
        print(f"Endpoint name: {endpoint_name}")
        print("Example test command:")
        print(
            "uv run python serving/scripts/test_lora_endpoint.py "
            f"--endpoint-name {endpoint_name} --prompt \"Explain LoRA in one sentence.\""
        )
    finally:
        for key, old in previous_env.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
