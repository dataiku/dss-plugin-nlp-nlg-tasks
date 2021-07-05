# -*- coding: utf-8 -*-
"""Module with utility functions to call the GPT translation API"""

import json

import requests

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (requests.HTTPError,)

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class GPTClient:
    def __init__(self, api_key) -> None:
        self.api_key = api_key
        self.url = "https://gpt-text-generation.p.rapidapi.com/complete"
        self.host = "gpt-text-generation.p.rapidapi.com"

    def format_prompt(self, text, task, input_desc, output_desc, example_in=None, example_out=None):
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
        ### Preprocess
        text = text.replace("\n", "")

        ### Put all together
        task_prompt = f"{task}\n\n"

        if example_in:
            example_prompt = f"{input_desc}: {example_in}\n{output_desc}: {example_out}\n\n"
        else:
            example_prompt = ""

        final_prompt = f"{input_desc}: {text}\n{output_desc}:"

        full_prompt = task_prompt + example_prompt + final_prompt

        return full_prompt

    def generate(
        self,
        text,
        task,
        input_desc,
        output_desc,
        example_in=None,
        example_out=None,
        temperature=0.8,
    ):
        """
        Generates Text.
        """

        prompt = self.format_prompt(text, task, input_desc, output_desc, example_in, example_out)

        print("Sending prompt:")
        print(prompt)
        response = requests.post(
            url=self.url,
            data=json.dumps({"prompt": prompt, "temperature": temperature}),
            headers={
                "content-type": "application/json",
                "x-rapidapi-key": self.api_key,
                "x-rapidapi-host": self.host,
            },
        )
        if "generation" in response.text:
            # Returns text from the response object which is a json string, so no need to dump it into json anymore
            return response.text
        else:
            # Extract & send error related information
            user_message = (
                "Encountered the following error while sending an API request:"
                + f" Error Code: {response.status_code}"
                + f" Error message: {response.text}"
            )

            raise requests.HTTPError(user_message)
