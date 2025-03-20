import warnings
warnings.filterwarnings("ignore")

import openai
openai.api_key = ""

import random
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import asyncio
from streamlit_extras.colored_header import colored_header
from streamlit_rich_message_history import *
from utils import *

st.set_page_config(layout="wide")

user_avatar = "😈"
bot_avatar = "🤖"
error_avatar = "⚠️"

phrases = [
    "🤔 Pondering",
    "💡 Coming up with ideas",
    "🧠 Brainstorming",
    "🔄 Processing",
    "⏳ Patience, please",
    "🔍 Investigating",
    "🛠️ Crafting a response",
    "🚀 Working on it",
    "⚙️ Fine-tuning",
    "🌐 Gathering information"
]

def init():
    if st.session_state.get("init", None): return
    st.session_state["message_history"] = MessageHistory()
    st.session_state["data_manager"] = DataManager()
    st.session_state["init"] = True

def sidebar():
    with st.sidebar:
        st.title("Data Explorer 📊")
        
        # Data upload section
        st.header("Upload Data")
        
        # Get data manager instance
        data_manager = st.session_state["data_manager"]
        
        # File uploader
        file = st.file_uploader(
            label="Upload a Dataset (CSV, Excel, or JSON)", 
            type=["csv", "xlsx", "xls", "xlsm", "json"]
        )

        
        if file and data_manager.df is None:
            # Process the uploaded file
            success, error_msg = data_manager.process_file(file)
            
            if not success and error_msg == "Please select an Excel sheet":
                # Show sheet selection for Excel files
                sheet_name = st.selectbox(
                    "Select a sheet:", 
                    options=data_manager.excel_sheets,
                    index=0 if data_manager.excel_sheets else None
                )
                
                if st.button("Load Selected Sheet"):
                    # Process Excel file with selected sheet
                    file.seek(0)  # Reset file pointer
                    success, error_msg = data_manager.process_file(file, sheet_name)
                    if success:
                        st.success(f"Successfully loaded sheet: {sheet_name}")
                        st.session_state["data_loaded"] = True
                        st.rerun()
                    else:
                        st.error(error_msg)
            
            elif not success:
                st.error(error_msg)
            
            elif success and not st.session_state.get("data_loaded"):
                st.success(f"Successfully loaded {file.name}")
                st.session_state["data_loaded"] = True
                st.rerun()
        
        if file == None and data_manager.df is not None:
            st.session_state["message_history"].clear()
            st.session_state["data_manager"].reset()
            st.session_state["data_loaded"] = False
            st.rerun()

        if data_manager.loaded:
            if st.button("Clear Message History"):
                st.session_state["message_history"].clear()
                st.rerun()

        # Data type conversion option
        if data_manager.loaded:
            if st.button("Attempt Auto Type"):
                data_manager.convert_dtypes()
                st.success("Data types converted where possible")
                st.rerun()
        
        # Show dataset information if data is loaded
        if data_manager.loaded:
            st.header("Dataset Information", divider="rainbow")
            
            # File information
            file_info_col1, file_info_col2 = st.columns(2)
            with file_info_col1:
                st.info(f"File: {data_manager.file_name}")
            with file_info_col2:
                st.info(f"Type: {data_manager.file_type.upper()}")
            
            if data_manager.file_type == 'excel' and data_manager.selected_sheet:
                st.info(f"Sheet: {data_manager.selected_sheet}")
            
            # Display rows and columns count
            if data_manager.df is not None:
                rows, cols = data_manager.df.shape
                st.metric("Rows", rows)
                st.metric("Columns", cols)
            
            # Display DataFrame info
            st.subheader("Column Information")
            st.dataframe(data_manager.df_info, use_container_width=True)
            
            # Option to view sample data
            if st.checkbox("View Sample Data"):
                st.subheader("Sample Data")
                st.dataframe(data_manager.df.head(10), use_container_width=True)
            
            # Option to reset/clear data
            if st.button("Clear Data", type="primary"):
                data_manager.reset()
                st.session_state["data_loaded"] = False
                st.rerun()

def messages():
    """Render all messages in the message history."""
    st.session_state["message_history"].render_all()

def extract_dataframe_from_result(result_content):
    """Extract a DataFrame from various result types (dict, list, DataFrame)."""
    # If it's already a DataFrame, return it
    if isinstance(result_content, pd.DataFrame):
        return result_content, None
    
    # If it's a dictionary, check if it contains a DataFrame
    if isinstance(result_content, dict):
        title = result_content.get('title', None)
        # Check for DataFrame under 'data' key
        if 'data' in result_content and isinstance(result_content['data'], pd.DataFrame):
            return result_content['data'], title
        # Check if any value is a DataFrame
        for key, value in result_content.items():
            if isinstance(value, pd.DataFrame):
                return value, title or f"Data from '{key}'"
    
    # If it's a list, check if any element is a DataFrame
    if isinstance(result_content, list):
        for i, item in enumerate(result_content):
            if isinstance(item, pd.DataFrame):
                return item, f"Data from list element {i}"
    
    # If we couldn't find a DataFrame, return None
    return None, None


async def query():
    """Handle user queries and process responses."""
    data_manager = st.session_state["data_manager"]
    user_input = st.chat_input(
        placeholder="What do you want to explore?", 
        disabled=not data_manager.loaded
    )
    
    if user_input:
        # Add user message to history
        user_message = UserMessage(avatar=user_avatar, text=user_input)
        st.session_state["message_history"].add_user_message(user_message)
        
        # Display user message
        with st.chat_message(name="user", avatar=user_avatar):
            st.markdown(user_input)
        
        # Process with assistant
        assistant_message = AssistantMessage(avatar=bot_avatar)
        with st.chat_message(name="assistant", avatar=bot_avatar) as message_container:
            df = data_manager.df
            try:
                async for result in df.ask.__aiter__(user_input, provider_type="ollama", model="gemma3:12b"):
                    if result.kind == ResultKind.CODE_BLOCK:
                        assistant_message.add_code(code=result.content, language="python", title="Here is the code I generated.")
                    elif result.kind == ResultKind.DESCRIPTION:
                        assistant_message.add(content=result.content, title="Here is an overview of the code.")
                    elif result.kind == ResultKind.RESULT:
                        # Check if result contains an error
                        if isinstance(result.content, dict) and "error" in result.content:
                            assistant_message.add_error(
                                error_text=f"I encountered an error while processing your request:\n\n{result.content['error']}", 
                                title="Error in Analysis"
                            )
                        else:
                            # Extract DataFrame and title if result contains a DataFrame in different formats
                            df_content, df_title = extract_dataframe_from_result(result.content)
                            
                            if df_content is not None:
                                # Use the extracted title if available, or use default
                                display_title = df_title or "Results Preview (First 5 Rows)"
                                assistant_message.add_dataframe_preview(df_content, title=display_title)
                            else:
                                # Just display the original result if no DataFrame found
                                assistant_message.add(content=result.content)
                    
                    if (result.kind not in (ResultKind.START, ResultKind.END)):
                        # Render the component after adding it
                        assistant_message.components[-1].render()
            except Exception as e:
                import traceback
                error_message = f"An unexpected error occurred: {str(e)}"
                st.error(error_message)
                # Add error to message history
                assistant_message.add(content=error_message, title="Error")
                assistant_message.components[-1].render()
        
        st.session_state["message_history"].add_assistant_message(assistant_message)

def main():
    init()
    sidebar()
    messages()
    asyncio.run(query())

if __name__ == "__main__":
    main()