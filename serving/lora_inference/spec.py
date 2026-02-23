"""Inference spec for serving Arcee Trinity Mini with a LoRA adapter on SageMaker."""

import json
import logging
import os
from pathlib import Path
from typing import Any

try:
    from sagemaker.serve.spec.inference_spec import InferenceSpec
except ImportError:
    class InferenceSpec:  # type: ignore[override]
        """Runtime fallback when SageMaker SDK is unavailable in the container."""

        def load(self, model_dir: str) -> Any:
            raise NotImplementedError

        def invoke(self, input_object: Any, model: Any) -> Any:
            raise NotImplementedError


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ArceeLoraInferenceSpec(InferenceSpec):
    """SageMaker v3 inference spec for a CausalLM base model with LoRA adapter.

    The loader resolves the base model from PEFT adapter metadata by default,
    with an explicit `BASE_MODEL_ID` override available through environment
    variables.
    """

    @staticmethod
    def _parse_bool(value: Any, default: bool) -> bool:
        """Parse a string-ish value into bool.

        Args:
            value: Raw value from payload or environment.
            default: Fallback value when parsing is not possible.

        Returns:
            bool: Parsed boolean.
        """
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        return default

    @staticmethod
    def _resolve_dtype(dtype_value: str, torch_module: Any) -> Any:
        """Resolve an environment dtype string to a torch dtype.

        Args:
            dtype_value: String such as `bfloat16`, `float16`, or `float32`.
            torch_module: Imported torch module.

        Returns:
            torch.dtype: Matching dtype for model loading.

        Raises:
            ValueError: If dtype string is unsupported.
        """
        normalized = dtype_value.strip().lower()
        dtype_map = {
            "bfloat16": torch_module.bfloat16,
            "bf16": torch_module.bfloat16,
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
            "auto": "auto",
        }
        if normalized not in dtype_map:
            raise ValueError(
                f"Unsupported DTYPE='{dtype_value}'. Supported values: {sorted(dtype_map)}"
            )
        return dtype_map[normalized]

    @staticmethod
    def _parse_payload(input_object: Any) -> dict[str, Any]:
        """Parse invoke payload into a JSON dictionary.

        Args:
            input_object: Payload from SageMaker runtime.

        Returns:
            dict[str, Any]: JSON payload dictionary.

        Raises:
            ValueError: If payload cannot be parsed into a dictionary.
        """
        if isinstance(input_object, dict):
            return input_object

        if isinstance(input_object, (bytes, bytearray)):
            try:
                decoded = input_object.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("Request body is not valid UTF-8.") from exc
            try:
                parsed = json.loads(decoded)
            except json.JSONDecodeError as exc:
                raise ValueError("Request body is not valid JSON.") from exc
            if not isinstance(parsed, dict):
                raise ValueError("JSON payload must be an object.")
            return parsed

        if isinstance(input_object, str):
            try:
                parsed = json.loads(input_object)
            except json.JSONDecodeError:
                return {"inputs": input_object}
            if not isinstance(parsed, dict):
                raise ValueError("JSON payload must be an object.")
            return parsed

        raise ValueError(
            f"Unsupported request type: {type(input_object)!r}. Expected dict, str, or bytes."
        )

    @staticmethod
    def _resolve_adapter_ref(model_dir: str, adapter_id: str) -> str:
        """Resolve adapter reference from model artifacts or Hugging Face.

        Args:
            model_dir: SageMaker model artifact directory.
            adapter_id: Adapter reference from environment.

        Returns:
            str: Resolved adapter path or HF repo id.
        """
        candidate = Path(adapter_id)
        if not candidate.is_absolute():
            in_model_dir = Path(model_dir) / adapter_id
            if in_model_dir.exists():
                return str(in_model_dir)

        if candidate.exists():
            return str(candidate.resolve())

        return adapter_id

    def get_model(self) -> str | None:
        """Return base model id for SDK image auto-detection.

        Returns:
            str | None: Hugging Face base model id when resolvable.
        """
        base_model_id = os.getenv("BASE_MODEL_ID")
        if base_model_id:
            return base_model_id

        adapter_id = os.getenv("ADAPTER_ID")
        if not adapter_id:
            return None

        candidate = Path(adapter_id)
        config_candidates = []
        if candidate.is_dir():
            config_candidates.append(candidate / "adapter_config.json")
        config_candidates.append(Path("/opt/ml/model") / adapter_id / "adapter_config.json")

        for config_path in config_candidates:
            if config_path.exists():
                try:
                    data = json.loads(config_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                resolved = data.get("base_model_name_or_path")
                if isinstance(resolved, str) and resolved.strip():
                    return resolved
        return None

    def load(self, model_dir: str) -> dict[str, Any]:
        """Load base CausalLM and attach LoRA adapter.

        Args:
            model_dir: Path to model artifacts directory provided by SageMaker.

        Returns:
            dict[str, Any]: Loaded model bundle with tokenizer, model, and defaults.

        Raises:
            ValueError: On invalid or missing configuration.
            RuntimeError: On import/loading failures.
        """
        adapter_id = os.getenv("ADAPTER_ID")
        if not adapter_id:
            raise ValueError("ADAPTER_ID environment variable is required.")

        adapter_ref = self._resolve_adapter_ref(model_dir, adapter_id)
        base_model_override = os.getenv("BASE_MODEL_ID")
        hf_token = os.getenv("HF_TOKEN")
        device_map = os.getenv("DEVICE_MAP", "auto")
        dtype_env = os.getenv("DTYPE", "bfloat16")

        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import required inference dependencies. "
                "Ensure requirements.txt contains torch/transformers/peft."
            ) from exc

        try:
            torch_dtype = self._resolve_dtype(dtype_env, torch)
            peft_config = PeftConfig.from_pretrained(adapter_ref, token=hf_token)
            resolved_base_model = base_model_override or peft_config.base_model_name_or_path
            if not resolved_base_model:
                raise ValueError(
                    "Unable to resolve base model from adapter config. "
                    "Set BASE_MODEL_ID explicitly."
                )

            LOGGER.info(
                "Loading tokenizer/model. base_model=%s adapter=%s device_map=%s dtype=%s",
                resolved_base_model,
                adapter_ref,
                device_map,
                dtype_env,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                resolved_base_model,
                token=hf_token,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                resolved_base_model,
                token=hf_token,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            model = PeftModel.from_pretrained(base_model, adapter_ref, token=hf_token)
            model.eval()

            return {
                "model": model,
                "tokenizer": tokenizer,
                "adapter_id": adapter_ref,
                "base_model_id": resolved_base_model,
                "defaults": {
                    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS_DEFAULT", "256")),
                    "temperature": float(os.getenv("TEMPERATURE_DEFAULT", "0.7")),
                    "top_p": float(os.getenv("TOP_P_DEFAULT", "0.95")),
                    "do_sample": self._parse_bool(os.getenv("DO_SAMPLE_DEFAULT"), True),
                },
                "torch": torch,
            }
        except Exception:
            LOGGER.exception("Failed to initialize model from adapter '%s'.", adapter_ref)
            raise

    def invoke(self, input_object: Any, model: dict[str, Any]) -> dict[str, Any]:
        """Run text generation with the LoRA-attached model.

        Args:
            input_object: Incoming payload from SageMaker runtime.
            model: Loaded model bundle returned by `load`.

        Returns:
            dict[str, Any]: Generation output and metadata.

        Raises:
            ValueError: If request payload is invalid.
            RuntimeError: If model invocation fails.
        """
        payload = self._parse_payload(input_object)
        prompt = payload.get("inputs")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Payload must include non-empty string field 'inputs'.")

        defaults = model["defaults"]
        max_new_tokens = int(payload.get("max_new_tokens", defaults["max_new_tokens"]))
        temperature = float(payload.get("temperature", defaults["temperature"]))
        top_p = float(payload.get("top_p", defaults["top_p"]))
        do_sample = self._parse_bool(payload.get("do_sample"), defaults["do_sample"])

        try:
            torch = model["torch"]
            runtime_model = model["model"]
            tokenizer = model["tokenizer"]

            tokenized = tokenizer(prompt, return_tensors="pt")
            device = next(runtime_model.parameters()).device
            tokenized = {key: value.to(device) for key, value in tokenized.items()}

            with torch.inference_mode():
                outputs = runtime_model.generate(
                    **tokenized,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_tokens = outputs[0][tokenized["input_ids"].shape[-1] :]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {
                "generated_text": generated_text,
                "full_text": full_text,
                "model_id": model["base_model_id"],
                "adapter_id": model["adapter_id"],
                "model_name": os.getenv("MODEL_NAME", "Trinity-Mini-DrugProt-Think"),
            }
        except Exception as exc:
            LOGGER.exception("Inference failure for adapter '%s'.", model.get("adapter_id"))
            raise RuntimeError(f"Inference failed: {exc}") from exc


_SPEC = ArceeLoraInferenceSpec()


def model_fn(model_dir: str) -> dict[str, Any]:
    """
    SageMaker Hugging Face handler hook to load the model once per worker.

    Args:
        model_dir: Path to `/opt/ml/model`.

    Returns:
        dict[str, Any]: Loaded runtime bundle.
    """
    return _SPEC.load(model_dir)


def input_fn(request_body: Any, request_content_type: str) -> dict[str, Any]:
    """
    Parse incoming request body into model input payload.

    Args:
        request_body: Raw request payload from MMS.
        request_content_type: Content type header value.

    Returns:
        dict[str, Any]: Parsed generation payload.

    Raises:
        ValueError: If content type is unsupported.
    """
    content_type = (request_content_type or "").split(";")[0].strip().lower()
    if content_type in {"application/json", "application/jsonlines", ""}:
        return _SPEC._parse_payload(request_body)
    if content_type in {"text/plain"}:
        if isinstance(request_body, (bytes, bytearray)):
            return {"inputs": request_body.decode("utf-8")}
        return {"inputs": str(request_body)}
    raise ValueError(f"Unsupported content type: {request_content_type!r}")


def predict_fn(input_data: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """
    Run prediction using parsed request and loaded model bundle.

    Args:
        input_data: Request payload parsed by `input_fn`.
        model: Loaded bundle from `model_fn`.

    Returns:
        dict[str, Any]: Generation response.
    """
    return _SPEC.invoke(input_data, model)


def output_fn(prediction: dict[str, Any], accept: str) -> tuple[str, str]:
    """
    Serialize prediction response for SageMaker runtime.

    Args:
        prediction: Prediction payload produced by `predict_fn`.
        accept: Requested response MIME type.

    Returns:
        tuple[str, str]: Serialized body and response content type.
    """
    _ = accept
    return json.dumps(prediction), "application/json"
