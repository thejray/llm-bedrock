import llm
from llm.plugins import pm
import os
import pytest

API_KEY = os.environ.get("PYTEST_BEDROCK_API_KEY") or "fake-key"


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"]}


@pytest.mark.vcr
def test_prompt():
    model = llm.get_model("us.amazon.nova-micro-v1:0")
    model.key = API_KEY
    response = model.prompt("What is the capital of France?", system="One word answer")
    assert response.text() == "Paris"
