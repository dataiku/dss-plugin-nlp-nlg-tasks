# -*- coding: utf-8 -*-
"""Module with classes to format results from the DeepL Translation API"""

import logging
from typing import AnyStr
from typing import Dict

import pandas as pd

from plugin_io_utils import (
    API_COLUMN_NAMES_DESCRIPTION_DICT,
    ErrorHandlingEnum,
    build_unique_column_names,
    generate_unique,
    safe_json_loads,
    move_api_columns_to_end,
)

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class GenericAPIFormatter:
    """
    Generic Formatter class for API responses:
    - initialize with generic parameters
    - compute generic column descriptions
    - apply format_row to dataframe
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        column_prefix: AnyStr = "api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        self.input_df = input_df
        self.column_prefix = column_prefix
        self.error_handling = error_handling
        self.api_column_names = build_unique_column_names(input_df, column_prefix)
        self.column_description_dict = {
            v: API_COLUMN_NAMES_DESCRIPTION_DICT[k]
            for k, v in self.api_column_names._asdict().items()
        }

    def format_row(self, row: Dict) -> Dict:
        return row

    def format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Formatting API results...")
        df = df.apply(func=self.format_row, axis=1)
        df = move_api_columns_to_end(df, self.api_column_names, self.error_handling)
        logging.info("Formatting API results: Done.")
        return df


class GPTAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for GPT API responses for the OpenedAI GPT API.
    Make sure the response is a valid JSON.
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        input_column: AnyStr = "",
        output_column: AnyStr = "generation",
        column_prefix: AnyStr = "gpt",
        output_mode: bool = False,
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)

        if output_mode:
            self.generated_text_column_name = output_column
        else:
            self.generated_text_column_name = generate_unique(
                f"{output_column}", input_df.columns, prefix=None
            )
        self.output_mode = output_mode
        self.input_column = input_column
        self.input_df_columns = input_df.columns
        self._compute_column_description()

    def _compute_column_description(self):
        if self.output_mode:
            self.column_description_dict[self.generated_text_column_name] = "Generated text."
        else:
            self.column_description_dict[
                self.generated_text_column_name
            ] = f"Generation based on '{self.input_column}' column."

    def format_row(self, row: Dict) -> Dict:
        """
        Formats raw row with response into final dataframe row.

        Args:
            row: Dict of a single dataframe row with a column corresponding to the response.

        Returns:
            row: Dict of a single formatted dataframe row
        """
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        # Only take the first line
        row[self.generated_text_column_name] = response.get("generation", "").split("\n")[0]
        return row
