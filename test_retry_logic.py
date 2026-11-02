from types import SimpleNamespace

def get_wait_time(exc, attempt):
    response_obj = getattr(exc, "response", None)
    headers = response_obj.headers if response_obj is not None else {}
    retry_after = headers.get("retry-after")

    try:
        if retry_after is not None:
            return float(retry_after) + 1
        return 10 * (attempt + 1)
    except (ValueError, TypeError):
        return 10 * (attempt + 1)


# Test 1: Groq provides retry-after
error = SimpleNamespace(
    response=SimpleNamespace(headers={"retry-after": "5"})
)
assert get_wait_time(error, 0) == 6

# Test 2: Header is missing
error = SimpleNamespace(response=SimpleNamespace(headers={}))
assert get_wait_time(error, 1) == 20

# Test 3: Header is invalid
error = SimpleNamespace(
    response=SimpleNamespace(headers={"retry-after": "invalid"})
)
assert get_wait_time(error, 0) == 10

print("All retry timing tests passed.")