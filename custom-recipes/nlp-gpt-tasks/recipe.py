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

api_configuration_preset = get_recipe_config().get("api_configuration_preset")
if api_configuration_preset is None or api_configuration_preset == {}:
    raise ValueError("Please specify an API configuration preset")

# Recipe parameters
# If there is not input dataset
output_mode = get_recipe_config().get("output_mode", False)
if output_mode:
    examples = get_recipe_config().get("output_examples", "")
    examples = [("", v) for v in examples]
    # Explicity set to empty strings as DSS may cache previous settings
    input_desc = ""
    text_column = ""
else:
    examples = get_recipe_config().get("examples")
    examples = [(k, v) for k, v in examples.items()]
    input_desc = get_recipe_config().get("input_desc", "")
    text_column = get_recipe_config().get("text_column")


print("EXAMPLES:", examples)

task = get_recipe_config().get("task", "")
output_desc = get_recipe_config().get("output_desc", "")
temperature = get_recipe_config().get("temperature", 0.8)

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
if api_configuration_preset.get("engine") == "openedai":
    response_column = "generation"
else:
    response_column = "text"


# ==============================================================================
# DEFINITIONS
# ==============================================================================

if get_recipe_config().get("output_mode"):
    # Simulate an empty input dataframe if none is specified
    input_dataset = None
    input_df = pd.DataFrame(
        [""] * get_recipe_config().get("num_outputs"), columns=[output_column_name]
    )
else:
    input_dataset = dataiku.Dataset(get_input_names_for_role("input_dataset")[0])
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
    temperature: float = 0.8,
) -> str:
    """
    Calls GPT Text Generation API.
    """
    if text_column:
        text = row[text_column]
    else:
        text = ""

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
        )
        return response


formatter = GPTAPIFormatter(
    input_df=input_df,
    output_column=output_column_name,
    input_column=text_column,
    column_prefix=column_prefix,
    output_mode=output_mode,
    error_handling=error_handling,
    response_column=response_column,
)

# ==============================================================================
# RUN
# ==============================================================================

df_parallelizer = DataFrameParallelizer(
    function=call_gpt_api,
    error_handling=error_handling,
    exceptions_to_catch=API_EXCEPTIONS,
    parallel_workers=parallel_workers,
    batch_size=1,
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
)

output_df = formatter.format_df(df)
output_dataset.write_with_schema(output_df)

set_column_descriptions(
    input_dataset=input_dataset,
    output_dataset=output_dataset,
    column_descriptions=formatter.column_description_dict,
)
