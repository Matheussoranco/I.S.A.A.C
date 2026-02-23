"""Tests for the SecurityPolicy dataclass."""

from __future__ import annotations

from isaac.sandbox.security import SecurityPolicy


class TestSecurityPolicy:
    def test_defaults(self) -> None:
        policy = SecurityPolicy()
        assert policy.network_mode == "none"
        assert policy.memory_limit == "256m"
        assert policy.cpu_limit == 1.0
        assert policy.pids_limit == 64
        assert policy.user == "65534:65534"
        assert policy.cap_drop == ["ALL"]
        assert policy.read_only_rootfs is True

    def test_to_container_kwargs(self) -> None:
        policy = SecurityPolicy()
        kwargs = policy.to_container_kwargs()
        assert kwargs["network_mode"] == "none"
        assert kwargs["mem_limit"] == "256m"
        assert kwargs["nano_cpus"] == int(1e9)
        assert kwargs["pids_limit"] == 64
        assert kwargs["user"] == "65534:65534"
        assert kwargs["cap_drop"] == ["ALL"]
        assert kwargs["security_opt"] == ["no-new-privileges"]
        assert kwargs["read_only"] is True

    def test_custom_policy(self) -> None:
        policy = SecurityPolicy(
            memory_limit="512m",
            cpu_limit=2.0,
            timeout_seconds=60,
        )
        assert policy.memory_limit == "512m"
        assert policy.timeout_seconds == 60
        kwargs = policy.to_container_kwargs()
        assert kwargs["nano_cpus"] == int(2e9)

    def test_frozen(self) -> None:
        policy = SecurityPolicy()
        try:
            policy.memory_limit = "1g"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass  # Expected — dataclass is frozen
