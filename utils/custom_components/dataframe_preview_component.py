import streamlit as st
import pandas as pd
from utils import DataManager
from streamlit_rich_message_history import MessageHistory


DATAFRAME_PREVIEW_TYPE = MessageHistory.register_component_type("dataframe_preview")


def dataframe_preview_renderer(content, kwargs):
    df = content
    component_id = id(df)

    if st.session_state.get(f"df_set_{component_id}", False):
        st.write("✅ **This dataframe has been set as the working dataframe**")
        return

    preview_df = df.head(5)
    st.write(kwargs.get("title", "Dataframe Preview"))
    st.dataframe(preview_df, use_container_width=True)

    if st.button("Set as Working Dataframe", key=f"set_df_{component_id}"):
        try:
            if "data_manager" not in st.session_state:
                st.error("data_manager not found in session state!")
                return
            
            st.session_state["data_manager"] = st.session_state["data_manager"].set_dataframe(df.copy(deep=True))
            
            st.session_state["df_just_updated"] = True

            st.session_state[f"df_set_{component_id}"] = True
            
            st.rerun()
        except Exception as e:
            st.error(f"Failed to set dataframe: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


MessageHistory.register_component_renderer(
    DATAFRAME_PREVIEW_TYPE, dataframe_preview_renderer
)

MessageHistory.register_component_method(
    "add_dataframe_preview", DATAFRAME_PREVIEW_TYPE
)
