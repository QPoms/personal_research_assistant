from backend.config import Settings
from backend.runtime.model_retry import (
    ModelRetryPolicy,
    is_retryable_model_error,
    retry_policy_from_settings,
)


class ProviderError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_retry_policy_is_five_retries_with_bounded_exponential_backoff():
    policy = ModelRetryPolicy(jitter=False)

    assert policy.max_retries == 5
    assert [policy.delay_seconds(index) for index in range(5)] == [2, 4, 8, 16, 30]


def test_only_transient_status_codes_are_retryable():
    for status_code in (408, 409, 429, 500, 502, 503):
        assert is_retryable_model_error(ProviderError(status_code)) is True
    for status_code in (400, 401, 403, 404, 422):
        assert is_retryable_model_error(ProviderError(status_code)) is False


def test_settings_build_the_shared_retry_policy():
    policy = retry_policy_from_settings(
        Settings(
            model_max_retries=5,
            model_retry_initial_delay_seconds=3,
            model_retry_backoff_factor=2,
            model_retry_max_delay_seconds=20,
            model_retry_jitter=False,
        )
    )

    assert policy == ModelRetryPolicy(
        max_retries=5,
        initial_delay_seconds=3,
        backoff_factor=2,
        max_delay_seconds=20,
        jitter=False,
    )
