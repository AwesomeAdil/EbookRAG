import streamlit as st

def display_reader(reading_chunks):
    """Display current chunk as an "eBook page" with navigation."""
    
    st.subheader(f"📄 Page {st.session_state.current_chunk + 1} of {len(reading_chunks)}")
    
    # Format the text for better readability
    page_text = reading_chunks[st.session_state.current_chunk]
    page_text = page_text.replace("\n\n", "\n\n\n")
    st.markdown(page_text)
    
    # Navigation Buttons
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    with nav_col1:
        if st.button("⬅️ Previous") and st.session_state.current_chunk > 0:
            st.session_state.current_chunk -= 1
            _update_history()
            st.rerun()
    
    with nav_col2:
        if st.button("Next ➡️") and st.session_state.current_chunk < len(reading_chunks) - 1:
            st.session_state.current_chunk += 1
            _update_history()
            st.rerun()

    with nav_col3:
        jump_page = st.number_input("Jump to page:", min_value=1, max_value=len(reading_chunks), value=st.session_state.current_chunk + 1)
        if st.button("Go"):
            st.session_state.current_chunk = jump_page - 1
            _update_history()
            st.rerun()

def _update_history():
    """Update reading history with current chunk."""
    if len(st.session_state.history) == 0 or st.session_state.history[-1] != st.session_state.current_chunk:
        st.session_state.history.append(st.session_state.current_chunk)
