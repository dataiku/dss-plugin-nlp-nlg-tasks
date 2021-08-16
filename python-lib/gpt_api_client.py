# -*- coding: utf-8 -*-
"""Module with client calling the GPT endpoint"""

import json
from typing import List
from typing import Tuple

import requests
from requests.models import Response
import openai

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (requests.HTTPError,)
OPENEDAI_URL = "https://gpt-text-generation.p.rapidapi.com/completions"
OPENEDAI_HOST = "gpt-text-generation.p.rapidapi.com"

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class GPTClient:
    def __init__(self, engine, api_key) -> None:
        self.engine = engine
        self.api_key = api_key

    def format_prompt(
        self,
        task: str = "",
        text: str = "",
        input_desc: str = "",
        output_desc: str = "",
        examples: List[Tuple[str, str]] = [("", "")],
    ) -> str:
        """
        Returns prompt of form:

        Correct grammar mistakes.

        Original: Where do you went?
        Standard American English: Where did you go?
        Original: Where is you?
        Standard American English:

        Args:
            task: The task for GPT, e.g. Correct grammar mistakes.
            input_desc: Description of input column
            output_desc: Description of output column
            example_in: Example of an input text
            example_out: Example of desired output text
        Returns:
            prompt: Formatted prompt
        """

        prompt = ""

        ### Task ###
        if task:
            prompt += f"{task}\n\n"

        ### Examples ###
        for ex_inp, ex_out in examples:
            if ex_inp:
                if input_desc:
                    prompt += f"{input_desc}: "
                prompt += f"{ex_inp}\n"

            # One could also provide examples without descriptions, e.g.
            # elephant
            # giraffe
            # cat
            if ex_out:
                if output_desc:
                    prompt += f"{output_desc}: "
                prompt += f"{ex_out}\n"

        ### Final prompt ###
        if text:
            if input_desc:
                prompt += f"{input_desc}: "
            prompt += f"{text}\n"

        if output_desc:
            prompt += f"{output_desc}:"

        return prompt

    def generate(
        self,
        task: str = "",
        text: str = "",
        input_desc: str = "",
        output_desc: str = "",
        examples: List[Tuple[str, str]] = [("", "")],
        temperature: float = 0.8,
    ) -> str:
        """
        Generates Text.
        """
        prompt = self.format_prompt(task, text, input_desc, output_desc, examples)

        if self.engine == "openedai":
            response = self.request_openedai(prompt, temperature)
        else:
            response = self.request_openai(prompt, temperature)

        return response

    def request_openai(self, prompt, temperature):

        response = openai.Completion.create(
            model=self.engine, prompt=prompt, stop="\n", temperature=temperature, max_tokens=100
        )

        return response

    def request_openedai(self, prompt, temperature):

        print("Sending prompt:")
        print(prompt)
        response = requests.post(
            url=OPENEDAI_URL,
            data=json.dumps({"prompt": prompt, "temperature": temperature}),
            headers={
                "content-type": "application/json",
                "x-rapidapi-key": self.api_key,
                "x-rapidapi-host": OPENEDAI_HOST,
            },
        )
        if "generation" in response.text:
            # Returns text from the response object which is a json string, so no need to dump it into json anymore
            return response.text
        else:
            # Extract & send error related information
            user_message = (
                "Encountered the following error while sending an API request to OpenedAI:"
                + f" Error Code: {response.status_code}"
                + f" Error message: {response.text}"
            )

            raise requests.HTTPError(user_message)
