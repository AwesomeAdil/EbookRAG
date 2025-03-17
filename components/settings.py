import streamlit as st
import pickle
import pandas as pd
from utils.diagnostics import run_diagnostics, fix_feedback_data

def display_settings(vector_db, reading_chunks, chunk_mapping, feedback_data, reranker):
    """Display settings and debug functionality."""
    
    st.subheader("⚙️ Settings & Debug")
    settings_tabs = st.tabs(["Basic", "Training", "Diagnostics", "Training Data", "Feedback Data"])
    
    # Tab 1: Basic Settings
    with settings_tabs[0]:
        _display_basic_settings()
        
        if st.checkbox("Show Chunk Mapping", key="show_chunk_mapping"):
            st.write("Sample of chunk mapping:")
            sample_size = min(10, len(chunk_mapping))
            sample_mapping = {k: chunk_mapping[k] for k in list(chunk_mapping.keys())[:sample_size]}
            st.json(sample_mapping)
    
    # Tab 2: Model Training
    with settings_tabs[1]:
        _display_training_settings(feedback_data, reranker)
    
    # Tab 3: Diagnostics
    with settings_tabs[2]:
        _display_diagnostics(vector_db, feedback_data, reranker)
    
    # Tab 4: Training Data
    with settings_tabs[3]:
        _display_training_data(feedback_data, reranker)
    
    # Tab 5: Feedback Data
    with settings_tabs[4]:
        st.subheader("Feedback Data")
        st.json(feedback_data)

def _display_basic_settings():
    """Display basic settings like clearing history."""
    if st.button("Clear Reading History", key="clear_history_btn"):
        st.session_state.history = []
        st.success("History cleared!")
        st.rerun()

def _display_training_settings(feedback_data, reranker):
    """Display model training settings and controls."""
    st.subheader("Model Retraining")
    
    # Option to bypass weekly training limit
    bypass_weekly_limit = st.checkbox("Bypass weekly training limit", value=False, 
                                     help="If checked, will attempt to retrain regardless of when the model was last trained",
                                     key="bypass_weekly_limit")
    
    # Display last training time
    if reranker.last_training_time:
        st.info(f"Model last trained on: {reranker.last_training_time.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.info("Model hasn't been trained yet")
    
    # Display feedback entries count
    feedback_count = sum(len(results) for results in feedback_data.values())
    st.info(f"Available feedback entries: {feedback_count}/{reranker.min_feedback_entries} required")
    
    # Retrain button
    if st.button("Retrain Model", key="retrain_model_btn"):
        # Override the should_retrain logic if bypass is checked
        if bypass_weekly_limit:
            # Still respect minimum entries requirement
            if feedback_count >= reranker.min_feedback_entries:
                with st.spinner("Training model..."):
                    reranker.train_model(force=True)
                st.success("Model trained successfully!")
            else:
                st.error(f"Not enough feedback data to train. Need at least {reranker.min_feedback_entries} entries.")
        else:
            # Use the normal should_retrain logic
            with st.spinner("Checking if retraining is needed..."):
                if reranker.should_retrain():
                    reranker.train_model()
                    st.success("Model trained successfully!")
                else:
                    st.warning("Retraining not needed yet based on time or feedback volume criteria.")

def _display_diagnostics(vector_db, feedback_data, reranker):
    """Display system diagnostics and troubleshooting tools."""
    st.subheader("System Diagnostics")
    
    # Run diagnostics on the system
    diagnostics_results = run_diagnostics(vector_db, feedback_data, reranker)
    
    # Display results in expandable sections
    with st.expander("Vector DB Stats", expanded=False):
        st.write(f"Total documents: {diagnostics_results['doc_count']}")
        st.write("Sample document IDs:", diagnostics_results['sample_ids'])
        
        # Show a few document examples
        for doc_id in diagnostics_results['sample_ids']:
            if doc_id in reranker.vector_db.docstore._dict:
                doc = reranker.vector_db.docstore._dict[doc_id]
                st.text_area(f"Document {doc_id}", doc.page_content[:200], height=100, key=f"doc_{doc_id}")
    
    with st.expander("Feedback-Vector DB Consistency", expanded=False):
        if diagnostics_results['missing_docs']:
            st.error(f"Found {len(diagnostics_results['missing_docs'])} document IDs in feedback data that don't exist in vector DB!")
            st.write("Examples:")
            for query, idx in diagnostics_results['missing_docs'][:5]:
                st.write(f"Query: '{query}', Missing doc ID: {idx}")
            
            # Add a fix button
            if st.button("Remove Invalid Entries", key="fix_feedback"):
                fix_feedback_data(feedback_data, reranker.vector_db, "delete")
                st.success("Feedback data cleaned! Reload the app to see changes.")
        else:
            st.success("All feedback entries refer to valid documents!")
    
    with st.expander("Document ID Analysis", expanded=False):
        st.write("### Feedback Data Document IDs:")
        if diagnostics_results['feedback_ids']:
            st.write(f"Sample IDs from feedback data: {diagnostics_results['feedback_ids'][:5]}")
            st.write(f"ID type: {diagnostics_results['feedback_id_type']}")
        else:
            st.write("No feedback data available yet.")
        
        st.write("### Vector DB Document IDs:")
        if diagnostics_results['vector_ids']:
            st.write(f"Sample IDs from vector DB: {diagnostics_results['vector_ids'][:5]}")
            st.write(f"ID type: {diagnostics_results['vector_id_type']}")
        else:
            st.write("No documents in vector DB.")
    
    # Add feedback data fixer
    st.write("### Fix Feedback Data")
    fix_strategy = st.radio(
        "Select fixing strategy:",
        ["Map to actual IDs", "Delete invalid entries"],
        key="fix_strategy"
    )
    
    if st.button("Fix Feedback Data", key="fix_feedback_btn"):
        fixed_count = fix_feedback_data(feedback_data, reranker.vector_db, fix_strategy.lower())
        st.success(f"Fixed {fixed_count} feedback entries. Restart the app to see changes.")

def _display_training_data(feedback_data, reranker):
    """Display training data for model tuning."""
    st.subheader("Training Data Preview")
    
    # Get the training data
    training_data = _prepare_training_data(feedback_data, reranker)

    # Display the data
    if training_data:
        # Create tabs for different views
        view_tab1, view_tab2 = st.tabs(["Tabular View", "Text Examples"])
        
        with view_tab1:
            # Show as a table
            df_data = [{
                "Query": item["query"],
                "Text Preview": item["doc_content"][:100] + "...",
                "Feedback": item["feedback"],
                "Similarity": item["similarity_score"],
                "Length": item["doc_length"]
            } for item in training_data]
            
            st.dataframe(pd.DataFrame(df_data))
        
        with view_tab2:
            # Show text examples with helpful/not helpful classification
            # Use a selectbox instead of expanders to avoid nesting issues
            example_selector = st.selectbox(
                "Select example to view:",
                options=range(len(training_data)),
                format_func=lambda i: f"Example {i+1}: {training_data[i]['query'][:50]}"
            )
            
            # Display the selected example
            item = training_data[example_selector]
            st.markdown(f"**Query:** {item['query']}")
            st.markdown(f"**Feedback:** {'👍 Helpful' if item['is_helpful'] else '👎 Not Helpful'}")
            st.markdown(f"**Similarity Score:** {item['similarity_score']}")
            st.markdown(f"**Document Length:** {item['doc_length']} words")
            st.markdown("**Content:**")
            st.text_area("", value=item['doc_content'], height=150, key=f"ta_example")
    else:
        st.info("No training data available yet.")

def _prepare_training_data(feedback_data, reranker):
    """Prepare training data from feedback entries."""
    import numpy as np
    
    training_data = []
    
    for query, results in feedback_data.items():
        for result_idx, feedback in results.items():
            try:
                # Check if result_idx is in the docstore
                if result_idx not in reranker.vector_db.docstore._dict:
                    continue
                
                # Get document content
                doc = reranker.vector_db.docstore._dict[result_idx]
                doc_content = doc.page_content
                
                # Get features
                query_embedding = reranker.vector_db.embedding_function.embed_query(query)
                doc_embedding = reranker.vector_db.embedding_function.embed_documents([doc_content])[0]
                similarity = np.dot(query_embedding, doc_embedding)
                doc_length = len(doc_content.split())
                
                # Add to training data
                training_data.append({
                    "query": query,
                    "doc_content": doc_content[:200] + "..." if len(doc_content) > 200 else doc_content,
                    "feedback": feedback,
                    "similarity_score": round(similarity, 3),
                    "doc_length": doc_length,
                    "is_helpful": feedback == "helpful"
                })
            except Exception as e:
                pass
    
    return training_data
