"""Invoke a SageMaker endpoint hosting Arcee Trinity Mini + LoRA adapter."""

import argparse
import json
import os
import time
from typing import Any

from sagemaker.core.deserializers import JSONDeserializer
from sagemaker.core.resources import Endpoint
from sagemaker.core.serializers import JSONSerializer


def _build_parser() -> argparse.ArgumentParser:
    """
    Create CLI parser for endpoint invocation.

    Returns:
        argparse.ArgumentParser: CLI parser.
    """
    parser = argparse.ArgumentParser(
        description="Invoke a SageMaker endpoint with a text prompt."
    )
    parser.add_argument(
        "--endpoint-name", required=True, help="SageMaker endpoint name."
    )
    parser.add_argument(
        "--prompt", required=True, help="Text prompt to generate from."
    )
    parser.add_argument(
        "--region", default=None, help="AWS region override."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None
    )
    parser.add_argument(
        "--temperature", type=float, default=None
    )
    parser.add_argument(
        "--top-p", type=float, default=None
    )
    parser.add_argument(
        "--do-sample", action="store_true", default=None
    )
    parser.add_argument(
        "--no-do-sample", action="store_false", dest="do_sample"
    )
    return parser


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    """
    Build invocation payload from CLI args.

    Args:
        args: Parsed CLI args.

    Returns:
        dict[str, Any]: JSON payload for endpoint invoke.
    """
    payload: dict[str, Any] = {"inputs": args.prompt}
    if args.max_new_tokens is not None:
        payload["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.do_sample is not None:
        payload["do_sample"] = args.do_sample
    return payload


def main() -> None:
    """Invoke endpoint and print response with latency."""
    args = _build_parser().parse_args()
    payload = _build_payload(args)
    if args.region:
        os.environ["AWS_DEFAULT_REGION"] = args.region
        os.environ["AWS_REGION"] = args.region

    endpoint = Endpoint.get(
        endpoint_name=args.endpoint_name,
        region=args.region,
    )
    endpoint.serializer = JSONSerializer()
    endpoint.deserializer = JSONDeserializer()

    try:
        started_at = time.perf_counter()
        response = endpoint.invoke(
            body=payload,
            content_type="application/json",
            accept="application/json",
            region=args.region,
        )
        latency_ms = (time.perf_counter() - started_at) * 1000.0

        body = response.body
        normalized: Any = body
        if isinstance(body, list) and body and isinstance(body[0], str):
            try:
                normalized = json.loads(body[0])
            except json.JSONDecodeError:
                normalized = body
        elif isinstance(body, str):
            try:
                normalized = json.loads(body)
            except json.JSONDecodeError:
                normalized = body

        print(json.dumps(normalized, indent=2, ensure_ascii=True))

        print(f"LatencyMs: {latency_ms:.2f}")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Invoke failed for endpoint {args.endpoint_name}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
