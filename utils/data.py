import pandas as pd
import streamlit as st
import io
import json
import openpyxl
from typing import Dict, Optional, Tuple, List, Any, Union
import os


class DataManager:
    """
    A helper class to manage data loading, processing, and state management
    for different file formats including CSV, Excel, and JSON.
    """

    def __init__(self):
        """Initialize the DataManager with empty state."""
        self.reset()

    def reset(self):
        """Reset all data state variables."""
        self.df = None
        self.df_info = None
        self.loaded = False
        self.original_file_name = None
        self.file_name = None
        self.file_type = None
        self.excel_sheets = []
        self.selected_sheet = None

    def get_dataframe_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate information about the dataframe columns and types.

        Args:
            df: The pandas DataFrame to analyze

        Returns:
            A DataFrame containing column information
        """
        # Create a DataFrame with column information
        info_data = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            non_null = df[col].count()
            null_count = df[col].isna().sum()
            unique_count = df[col].nunique()

            # Get sample values (up to 3)
            sample_values = df[col].dropna().head(3).tolist()
            sample_str = ", ".join([str(val) for val in sample_values])
            if len(sample_str) > 50:
                sample_str = sample_str[:47] + "..."

            info_data.append(
                {
                    "Column": str(col),
                    "Data Type": dtype_str,
                    "Non-Null Count": f"{non_null} / {len(df)}",
                    "Null Count": null_count,
                    "Unique Values": unique_count,
                    "Sample Values": sample_str,
                }
            )

        df_info = pd.DataFrame(info_data)
        return df_info

    def detect_file_type(self, file_name: str) -> str:
        """
        Detect the file type from filename.

        Args:
            file_name: Name of the file

        Returns:
            String indicating file type ('csv', 'excel', 'json', or 'unknown')
        """
        lower_name = file_name.lower()
        if lower_name.endswith(".csv"):
            return "csv"
        elif lower_name.endswith((".xls", ".xlsx", ".xlsm")):
            return "excel"
        elif lower_name.endswith(".json"):
            return "json"
        else:
            return "unknown"

    def get_excel_sheet_names(self, file) -> List[str]:
        """
        Get all sheet names from an Excel file.

        Args:
            file: The uploaded Excel file object

        Returns:
            List of sheet names
        """
        # Save the uploaded file to a temporary file
        with io.BytesIO(file.read()) as buffer:
            # Reset file pointer
            file.seek(0)
            # Load Excel workbook
            wb = openpyxl.load_workbook(buffer, read_only=True, data_only=True)
            return wb.sheetnames

    def process_file(
        self, file, sheet_name: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Process the uploaded file and convert to DataFrame.

        Args:
            file: The uploaded file object
            sheet_name: Sheet name for Excel files

        Returns:
            Tuple of (success, error_message)
        """
        print("[DataManager]: processing file into dataframe.")
        try:
            self.file_name = file.name
            self.original_file_name = file.name
            self.file_type = self.detect_file_type(file.name)

            if self.file_type == "csv":
                self.df = pd.read_csv(file)
                self.loaded = True
                self.df_info = self.get_dataframe_info(self.df)
                return True, None

            elif self.file_type == "excel":
                if sheet_name is None:
                    # Just get the sheet names, don't load the data yet
                    self.excel_sheets = self.get_excel_sheet_names(file)
                    return False, "Please select an Excel sheet"
                else:
                    # Load the selected sheet
                    self.df = pd.read_excel(file, sheet_name=sheet_name)
                    self.selected_sheet = sheet_name
                    self.loaded = True
                    self.df_info = self.get_dataframe_info(self.df)
                    return True, None

            elif self.file_type == "json":
                try:
                    # Try to load as a record-oriented JSON
                    self.df = pd.read_json(file)
                    self.loaded = True
                    self.df_info = self.get_dataframe_info(self.df)
                    return True, None
                except:
                    # If that fails, load JSON and try to normalize it
                    file.seek(0)
                    json_data = json.load(file)

                    # If it's a dict, convert to DataFrame
                    if isinstance(json_data, dict):
                        self.df = pd.json_normalize(json_data)
                    # If it's a list, convert to DataFrame
                    elif isinstance(json_data, list):
                        self.df = pd.json_normalize(json_data)
                    else:
                        return False, "Unsupported JSON structure"

                    self.loaded = True
                    self.df_info = self.get_dataframe_info(self.df)
                    return True, None
            else:
                return False, f"Unsupported file format: {self.file_type}"

        except Exception as e:
            return False, f"Error processing file: {str(e)}"

    def convert_dtypes(self):
        """Attempt to convert string columns to appropriate data types."""
        if self.df is not None:
            for col in self.df.columns:
                # Try to convert to numeric
                try:
                    self.df[col] = pd.to_numeric(self.df[col])
                except:
                    # If numeric conversion fails, try datetime
                    try:
                        self.df[col] = pd.to_datetime(self.df[col])
                    except:
                        # Keep as string if both conversions fail
                        pass

            # Update dataframe info after type conversion
            self.df_info = self.get_dataframe_info(self.df)

    def set_dataframe(self, df):
        """Set a dataframe as the current working dataframe."""
        self.df = df.copy(deep=True)
        if self.original_file_name:
            self.file_name = f"Generated DataFrame ({self.original_file_name})"
        else:
            self.file_name = "Generated DataFrame"
        self.loaded = True
        self.df_info = self.get_dataframe_info(self.df)
        self.last_modified = pd.Timestamp.now()
        return self

    def reset(self):
        """Reset all data state variables."""
        self.df = None
        self.df_info = None
        self.loaded = False
        self.original_file_name = None
        self.file_name = None
        self.file_type = None
        self.excel_sheets = []
        self.selected_sheet = None