# -*- coding: utf-8 -*-
import json
from typing import Dict
from typing import List
from typing import Tuple

import dataiku
from dataiku.customrecipe import get_input_names_for_role
from dataiku.customrecipe import get_output_names_for_role
from dataiku.customrecipe import get_recipe_config

from dkulib.dku_io_utils import set_column_descriptions
from dkulib.parallelizer import DataFrameParallelizer
from gpt_api_client import API_EXCEPTIONS
from gpt_api_client import GPTClient
from gpt_api_formatting import GPTAPIFormatter
import pandas as pd
from plugin_io_utils import ErrorHandlingEnum
from plugin_io_utils import validate_column_input
from retry import retry

# ==============================================================================
# SETUP
# ==============================================================================

recipe_config = get_recipe_config()

api_configuration_preset = recipe_config.get("api_configuration_preset")
if api_configuration_preset is None or api_configuration_preset == {}:
    raise ValueError("Please specify an API configuration preset")

output_mode = recipe_config.get("output_mode", False)
if output_mode:
    examples = recipe_config.get("output_examples", "")
    examples = [("", v) for v in examples]
    # Explicity set to empty strings as DSS may cache previous settings
    input_desc = ""
    text_column = ""
else:
    examples = recipe_config.get("examples")
    examples = [(k, v) for k, v in examples.items()]
    input_desc = recipe_config.get("input_desc", "")
    # If none specificed, will trigger <class 'ValueError'>: You must specify a valid column name
    text_column = recipe_config.get("text_column")

task = recipe_config.get("task", "")
output_desc = recipe_config.get("output_desc", "")
temperature = recipe_config.get("temperature", 0.7)
max_tokens = recipe_config.get("max_tokens", 64)

# Create a fitting name for the output column
if output_desc:
    output_column_name = output_desc.lower().replace(" ", "_")
else:
    output_column_name = "generation"

# Params for parallelization
column_prefix = "gpt"
parallel_workers = api_configuration_preset.get("parallel_workers")
error_handling = (
    ErrorHandlingEnum.FAIL if get_recipe_config().get("fail_on_error") else ErrorHandlingEnum.LOG
)

# Create client
client = GPTClient(api_configuration_preset.get("engine"), api_configuration_preset.get("api_key"))
max_attempts = api_configuration_preset.get("max_attempts")
wait_interval = api_configuration_preset.get("wait_interval")


# ==============================================================================
# DEFINITIONS
# ==============================================================================

if output_mode:
    input_dataset = None
    # Simulate an empty input dataframe if none is specified
    input_df = pd.DataFrame([""] * recipe_config.get("num_outputs"), columns=[output_column_name])
else:
    input_dataset_names = get_input_names_for_role("input_dataset")
    if not input_dataset_names:
        raise ValueError(
            "Cannot find input dataset. Use Output-only mode to generate without input dataset."
        )
    input_dataset = dataiku.Dataset(input_dataset_names[0])
    validate_column_input(text_column, [col["name"] for col in input_dataset.read_schema()])
    input_df = input_dataset.get_dataframe()

output_dataset = dataiku.Dataset(get_output_names_for_role("output_dataset")[0])


@retry((API_EXCEPTIONS), delay=wait_interval, tries=max_attempts)
def call_gpt_api(
    row: Dict,
    text_column: str = "",
    task: str = "",
    input_desc: str = "",
    output_desc: str = "",
    examples: List[Tuple[str, str]] = [("", "")],
    temperature: float = 0.7,
    max_tokens: int = 64,
) -> str:
    """
    Calls GPT Text Generation API.
    """
    if text_column:
        text = row[text_column]
    else:
        text = ""

    # Recipe UI will show an error when selecting a non-string input column
    if not isinstance(text, str):
        return json.dumps({})
    else:
        response = client.generate(
            task=task,
            text=text,
            input_desc=input_desc,
            output_desc=output_desc,
            examples=examples,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response


formatter = GPTAPIFormatter(
    input_df=input_df,
    output_column=output_column_name,
    input_column=text_column,
    column_prefix=column_prefix,
    output_mode=output_mode,
    error_handling=error_handling,
)

# ==============================================================================
# RUN
# ==============================================================================

df_parallelizer = DataFrameParallelizer(
    function=call_gpt_api,
    error_handling=error_handling,
    exceptions_to_catch=API_EXCEPTIONS,
    parallel_workers=parallel_workers,
    output_column_prefix=column_prefix,
)

df = df_parallelizer.run(
    input_df,
    text_column=text_column,
    task=task,
    input_desc=input_desc,
    output_desc=output_desc,
    examples=examples,
    temperature=temperature,
    max_tokens=max_tokens,
)

output_df = formatter.format_df(df)
output_dataset.write_with_schema(output_df)

set_column_descriptions(
    input_dataset=input_dataset,
    output_dataset=output_dataset,
    column_descriptions=formatter.column_description_dict,
)
