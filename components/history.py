import streamlit as st

def display_history(reading_chunks):
    """Display reading history with pagination."""
    
    st.subheader("📚 Reading History")
    if len(st.session_state.history) > 0:
        # Show only the 3 most recent pages by default
        history_limit = 3
        
        # Create two columns for recent history and "Show More" button
        hist_col1, hist_col2 = st.columns([3, 1])
        
        with hist_col1:
            # Display the most recent history items
            for i, page_idx in enumerate(reversed(st.session_state.history[-history_limit:])):
                if st.button(f"Page {page_idx + 1}", key=f"history_{i}"):
                    st.session_state.current_chunk = page_idx
                    st.rerun()
        
        with hist_col2:
            # Add a "Show More" button if there's more history
            if len(st.session_state.history) > history_limit:
                if "show_full_history" not in st.session_state:
                    st.session_state.show_full_history = False
                    
                if st.button("Show More" if not st.session_state.show_full_history else "Show Less"):
                    st.session_state.show_full_history = not st.session_state.show_full_history
                    st.rerun()
        
        # Show expanded history if requested
        if "show_full_history" in st.session_state and st.session_state.show_full_history:
            _display_full_history(history_limit)
    else:
        st.write("No history yet")

def _display_full_history(history_limit):
    """Display the full reading history."""
    st.write("---")
    st.write("Full Reading History:")
    
    # Create a grid layout for history items
    num_cols = 3  # Number of buttons per row
    history_items = list(reversed(st.session_state.history))
    
    # Skip the first few items as they're already shown above
    history_items = history_items[history_limit:]
    
    # Create rows of history items
    for i in range(0, len(history_items), num_cols):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            if i + j < len(history_items):
                page_idx = history_items[i + j]
                with cols[j]:
                    if st.button(f"Page {page_idx + 1}", key=f"full_history_{i+j}"):
                        st.session_state.current_chunk = page_idx
                        st.rerun()
