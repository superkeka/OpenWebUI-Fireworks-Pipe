"""
title: Fireworks Manifold Pipe
author: superkeka
author_url: https://github.com/superkeka
funding_url: https://github.com/open-webui
version: 0.1.3
"""

from pydantic import BaseModel, Field
from typing import List, Union, Generator, Iterator
import os
import requests
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Pipe:
    class Valves(BaseModel):
        FIREWORKS_API_KEY: str = Field(default="")

    def __init__(self):
        self.type = "manifold"
        self.id = "fireworks"
        self.name = "fireworks/"
        self.valves = self.Valves(
            **{"FIREWORKS_API_KEY": os.getenv("FIREWORKS_API_KEY", "")}
        )

    def get_fireworks_models(self):
        return [
            {
                "id": "accounts/fireworks/models/llama-v3p1-405b-instruct",
                "name": "llama-v3p1-405b-instruct",
            },
            {
                "id": "accounts/fireworks/models/llama-v3p1-70b-instruct",
                "name": "llama-v3p1-70b-instruct",
            },
            {
                "id": "accounts/fireworks/models/llama-v3p1-8b-instruct",
                "name": "llama-v3p1-8b-instruct",
            },
            {
                "id": "accounts/fireworks/models/llama-v3p2-1b-instruct",
                "name": "llama-v3p2-1b-instruct",
            },
            {
                "id": "accounts/fireworks/models/llama-v3p2-3b-instruct",
                "name": "llama-v3p2-3b-instruct",
            },
            {
                "id": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
                "name": "llama-v3p2-11b-vision-instruct",
            },
            {
                "id": "accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
                "name": "llama-v3p2-90b-vision-instruct",
            },
        ]

    def pipes(self) -> List[dict]:
        return self.get_fireworks_models()

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        logger.debug(f"Requested model: {body['model']}")

        models = self.get_fireworks_models()

        model_id = next(
            (
                model["id"]
                for model in models
                if model["id"] == body["model"].replace("fireworks_pipe.", "")
            ),
            None,
        )
        if model_id is None:
            logger.error(f"Model '{body['model']}' not found")
            return f"Error: Model '{body['model']}' not found"

        payload = {
            "model": model_id,
            "messages": body.get("messages", []),
            "max_tokens": body.get("max_tokens", 16384),
            "temperature": body.get("temperature", 0.6),
            "top_k": body.get("top_k", 40),
            "top_p": body.get("top_p", 1),
            "presence_penalty": body.get("presence_penalty", 0),
            "frequency_penalty": body.get("frequency_penalty", 0),
            "stream": body.get("stream", False),
        }

        logger.debug(f"Request payload: {payload}")

        headers = {
            "Authorization": f"Bearer {self.valves.FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        logger.debug(f"Request headers: {headers}")

        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        logger.debug(f"Request URL: {url}")

        try:
            if body.get("stream", False):
                return self.stream_response(url, headers, payload)
            else:
                return self.non_stream_response(url, headers, payload)
        except Exception as e:
            logger.error(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(self, url, headers, payload):
        logger.debug(f"Sending stream request to {url}")
        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            if response.status_code != 200:
                logger.error(f"HTTP Error {response.status_code}: {response.text}")
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if "choices" in data and len(data["choices"]) > 0:
                                content = data["choices"][0]["delta"].get("content")
                                if content:
                                    logger.debug(f"Yielding content: {content}")
                                    yield content
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON: {line}")
                        except KeyError as e:
                            logger.error(f"Unexpected data structure: {e}")
                            logger.error(f"Full data: {data}")

    def non_stream_response(self, url, headers, payload):
        logger.debug(f"Sending non-stream request to {url}")
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            logger.error(f"HTTP Error {response.status_code}: {response.text}")
            raise Exception(f"HTTP Error {response.status_code}: {response.text}")

        res = response.json()
        logger.debug(f"Received response: {res}")
        return (
            res["choices"][0]["message"]["content"]
            if "choices" in res and res["choices"]
            else ""
        )
