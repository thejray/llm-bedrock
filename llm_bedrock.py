import boto3
import llm

MODELS = (
    "us.amazon.nova-micro-v1:0",
    "us.amazon.nova-lite-v1:0",
    "us.amazon.nova-pro-v1:0",
)
AWS_REGION = "us-west-2"


@llm.hookimpl
def register_models(register):
    for model_id in MODELS:
        register(BedrockModel(model_id))


class BedrockModel(llm.Model):
    needs_key = "bedrock-runtime"
    can_stream = True

    def __init__(self, model_id):
        self.model_id = model_id

    def execute(self, prompt, stream, response, conversation):
        key = self.get_key()
        # It's actually access key: secret key
        access_key, _, secret_key = key.partition(":")
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        bedrock = session.client("bedrock-runtime", region_name=AWS_REGION)
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt.prompt},
                ],
            }
        ]
        params = {"messages": messages, "modelId": self.model_id}
        chunks = []
        usage = {}
        if stream:
            bedrock_response = bedrock.converse_stream(**params)
            for event in bedrock_response["stream"]:
                chunks.append(event)
                ((event_type, event_content),) = event.items()
                if event_type == "contentBlockDelta":
                    yield event_content["delta"]["text"]
                if "metadata" in event and "usage" in event["metadata"]:
                    usage = event["metadata"]["usage"]
        else:
            bedrock_response = bedrock.converse(**params)
            chunk = bedrock_response["output"]["message"]["content"][-1]["text"]
            yield chunk
            usage = bedrock_response["usage"]

        if usage:
            response.set_usage(input=usage["inputTokens"], output=usage["outputTokens"])

    def __str__(self):
        return "Bedrock: {}".format(self.model_id)
