# -*- coding: utf-8 -*-
import json
from typing import AnyStr
from typing import Dict

import dataiku
from dataiku.customrecipe import get_input_names_for_role
from dataiku.customrecipe import get_output_names_for_role
from dataiku.customrecipe import get_recipe_config

from gpt_api_client import API_EXCEPTIONS
from gpt_api_client import GPTClient
from gpt_api_formatting import GPTAPIFormatter
from dku_io_utils import set_column_description
from plugin_io_utils import ErrorHandlingEnum
from plugin_io_utils import validate_column_input
from retry import retry

from parallelizer import DataFrameParallelizer

# ==============================================================================
# SETUP
# ==============================================================================

api_configuration_preset = get_recipe_config().get("api_configuration_preset")
if api_configuration_preset is None or api_configuration_preset == {}:
    raise ValueError("Please specify an API configuration preset")

# Recipe parameters
text_column = get_recipe_config().get("text_column")
task = get_recipe_config().get("task", "")
input_desc = get_recipe_config().get("input_desc", "")
output_desc = get_recipe_config().get("output_desc", "")
example_in = get_recipe_config().get("example_in", "")
example_out = get_recipe_config().get("example_out", "")
temperature = get_recipe_config().get("temperature", 0.8)

# Params for parallelization
column_prefix = "gpt_api"
parallel_workers = api_configuration_preset.get("parallel_workers")
error_handling = (
    ErrorHandlingEnum.FAIL if get_recipe_config().get("fail_on_error") else ErrorHandlingEnum.LOG
)

# Params for translation
client = GPTClient(api_configuration_preset.get("rapidapi_key"))
max_attempts = api_configuration_preset.get("max_attempts")
wait_interval = api_configuration_preset.get("wait_interval")


# ==============================================================================
# DEFINITIONS
# ==============================================================================

input_dataset = dataiku.Dataset(get_input_names_for_role("input_dataset")[0])
output_dataset = dataiku.Dataset(get_output_names_for_role("output_dataset")[0])
validate_column_input(text_column, [col["name"] for col in input_dataset.read_schema()])
input_df = input_dataset.get_dataframe()


@retry((API_EXCEPTIONS), delay=wait_interval, tries=max_attempts)
def call_gpt_api(
    row: Dict,
    text_column: AnyStr,
    task: AnyStr,
    input_desc: AnyStr,
    output_desc: AnyStr,
    example_in: AnyStr = None,
    example_out: AnyStr = None,
    temperature: float = 0.8,
) -> AnyStr:
    """
    Calls GPT Text Generation API.
    """
    text = row[text_column]
    if not isinstance(text, str) or str(text).strip() == "":
        return json.dumps({})
    else:
        response = client.generate(
            text=text,
            task=task,
            input_desc=input_desc,
            output_desc=output_desc,
            example_in=example_in,
            example_out=example_out,
            temperature=temperature,
        )
        return response


formatter = GPTAPIFormatter(
    input_df=input_df,
    input_column=text_column,
    column_prefix=column_prefix,
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
    example_in=example_in,
    example_out=example_out,
    temperature=temperature,
)

output_df = formatter.format_df(df)
output_dataset.write_with_schema(output_df)

set_column_description(
    input_dataset=input_dataset,
    output_dataset=output_dataset,
    column_description_dict=formatter.column_description_dict,
)
