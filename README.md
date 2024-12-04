# llm-bedrock

[![PyPI](https://img.shields.io/pypi/v/llm-bedrock.svg)](https://pypi.org/project/llm-bedrock/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-bedrock?include_prereleases&label=changelog)](https://github.com/simonw/llm-bedrock/releases)
[![Tests](https://github.com/simonw/llm-bedrock/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/llm-bedrock/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-bedrock/blob/main/LICENSE)

Run prompts against models hosted on AWS Bedrock

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-bedrock
```
## Usage

You'll need an access key and a secret key to use this plugin, with permission granted to access the Bedrock models. These [step by step instructions](https://ndurner.github.io/amazon-nova) can help you obtain those credentials.

Combine those into a `access_key:secret_key` format (joined by a colon) and paste that into:

```bash
llm keys set bedrock
# paste access_key:secret_key here
```

Run `llm models` to see the list of models. Run a prompt like this:

```bash
llm -m us.amazon.nova-pro-v1:0 'a happy poem about a pelican with a secret'
```
## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-bedrock
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```
