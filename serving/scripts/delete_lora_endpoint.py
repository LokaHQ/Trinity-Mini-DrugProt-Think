"""Delete a SageMaker endpoint using SageMaker SDK v3 resources."""

import argparse
import os
import time

from botocore.exceptions import ClientError
from sagemaker.core.resources import Endpoint


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(description="Delete a SageMaker endpoint.")
    parser.add_argument(
        "--endpoint-name", required=True, help="SageMaker endpoint name."
    )
    parser.add_argument("--region", default=None, help="AWS region override.")
    parser.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="Wait until deletion completes.",
    )
    parser.add_argument(
        "--no-wait", dest="wait", action="store_false", help="Do not wait for deletion."
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Maximum wait time in seconds.",
    )
    parser.add_argument(
        "--poll-seconds", type=int, default=15, help="Polling interval while waiting."
    )
    return parser


def _is_not_found(exc: ClientError) -> bool:
    """Check whether a boto-style error indicates endpoint does not exist.

    Args:
        exc: ClientError raised by AWS API.

    Returns:
        bool: True when endpoint is not found.
    """
    code = exc.response.get("Error", {}).get("Code", "")
    message = exc.response.get("Error", {}).get("Message", "")
    return code == "ValidationException" and "Could not find endpoint" in message


def _wait_for_delete(
    endpoint_name: str, region: str | None, timeout_seconds: int, poll_seconds: int
) -> None:
    """
    Wait until endpoint no longer exists.

    Args:
        endpoint_name: Endpoint name.
        region: AWS region.
        timeout_seconds: Maximum wait time.
        poll_seconds: Poll interval in seconds.

    Raises:
        TimeoutError: If endpoint is not deleted within timeout.
        ClientError: If an unexpected AWS error occurs.
    """
    started = time.monotonic()
    while True:
        try:
            endpoint = Endpoint.get(endpoint_name=endpoint_name, region=region)
            print(f"Current status: {endpoint.endpoint_status}")
        except ClientError as exc:
            if _is_not_found(exc):
                print(f"Endpoint deleted: {endpoint_name}")
                return
            raise

        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for endpoint deletion after {timeout_seconds}s: {endpoint_name}"
            )
        time.sleep(poll_seconds)


def main() -> None:
    """
    Delete an endpoint and optionally wait for completion.

    Raises:
        RuntimeError: If delete request fails.
        TimeoutError: If wait is enabled and deletion does not complete in time.
    """
    args = _build_parser().parse_args()
    if args.region:
        os.environ["AWS_DEFAULT_REGION"] = args.region
        os.environ["AWS_REGION"] = args.region

    try:
        endpoint = Endpoint.get(endpoint_name=args.endpoint_name, region=args.region)
    except ClientError as exc:
        if _is_not_found(exc):
            print(f"Endpoint not found, nothing to delete: {args.endpoint_name}")
            return
        raise RuntimeError(
            f"Failed to describe endpoint {args.endpoint_name}: {exc}"
        ) from exc

    print(
        f"Deleting endpoint: {args.endpoint_name} (current status: {endpoint.endpoint_status})"
    )
    try:
        endpoint.delete()
    except ClientError as exc:
        if _is_not_found(exc):
            print(f"Endpoint already deleted: {args.endpoint_name}")
            return
        raise RuntimeError(
            f"Failed to delete endpoint {args.endpoint_name}: {exc}"
        ) from exc

    if args.wait:
        _wait_for_delete(
            endpoint_name=args.endpoint_name,
            region=args.region,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    else:
        print("Delete request submitted.")


if __name__ == "__main__":
    main()
