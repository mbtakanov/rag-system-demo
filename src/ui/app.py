import streamlit as st
import httpx
import json

from config import API_URL

st.title("RAG System")

tabs = st.tabs(["Ask", "Search", "Health", "Metadata", "Root"])

# Ask Tab
with tabs[0]:

    def submit_query():
        st.session_state.ask_submitted = True

    query = st.text_input(
        "Ask a question:", key="ask_query", on_change=submit_query, max_chars=500
    )
    col1, col2 = st.columns(2)
    with col1:
        k = st.slider("Number of results", 1, 10, 5, key="ask_k")
    with col2:
        alpha = st.slider("Vector weight", 0.0, 1.0, 0.5, key="ask_alpha")

    expand_query = st.checkbox("Enable query expansion", key="ask_expand")

    if st.button("Ask", key="ask_btn"):
        st.session_state.ask_submitted = True

    if st.session_state.get("ask_submitted", False) and query.strip():
        st.session_state.ask_submitted = False

        with st.spinner("Answering..."):
            try:
                with httpx.stream(
                    "GET",
                    f"{API_URL}/ask",
                    params={
                        "query": query,
                        "k": k,
                        "alpha": alpha,
                        "expand_query": expand_query,
                    },
                    timeout=30.0,
                ) as response:
                    answer_placeholder = st.empty()
                    full_answer = ""
                    metadata = None

                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            if data["type"] == "content":
                                full_answer += data["data"]
                                answer_placeholder.write(full_answer)
                            elif data["type"] == "metadata":
                                metadata = data["data"]
                            elif data["type"] == "error":
                                st.error(data["data"])

                    if metadata:
                        st.subheader("Retrieval Metadata")
                        st.json(metadata)
            except Exception as e:
                st.error(f"Error: {e}")

# Search Tab
with tabs[1]:
    st.info("**Search for relevant document chunks without generating an answer.**")

    def submit_search():
        st.session_state.search_submitted = True

    search_query = st.text_input(
        "Search query:", key="search_query", on_change=submit_search
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        search_k = st.slider("Number of results", 1, 10, 5, key="search_k")
    with col2:
        search_alpha = st.slider("Vector weight", 0.0, 1.0, 0.5, key="search_alpha")
    with col3:
        # TODO: Add pdf and docx once we fix the metadata extraction.
        file_type = st.selectbox("File type", ["All", "md"], key="file_type")

    search_expand = st.checkbox("Enable query expansion", key="search_expand")

    if st.button("Search", key="search_btn"):
        st.session_state.search_submitted = True

    if st.session_state.get("search_submitted", False) and search_query.strip():
        st.session_state.search_submitted = False

        with st.spinner("Searching..."):
            try:
                params = {
                    "query": search_query,
                    "k": search_k,
                    "alpha": search_alpha,
                    "expand_query": search_expand,
                }
                if file_type != "All":
                    params["file_type"] = file_type

                response = httpx.get(f"{API_URL}/search", params=params, timeout=30.0)

                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Found {data['num_results']} results")

                    if data.get("expanded"):
                        st.info("Query expansion was used")

                    for i, result in enumerate(data["results"], 1):
                        with st.expander(f"Result {i} (Score: {result['score']})"):
                            st.write(result["content"])
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")

# Health Tab
with tabs[2]:
    st.markdown("Check the health status of the RAG system.")

    if st.button("Check Health", key="health_btn"):
        with st.spinner("Checking health..."):
            try:
                response = httpx.get(f"{API_URL}/health", timeout=10.0)

                if response.status_code == 200:
                    data = response.json()

                    if data["status"] == "healthy":
                        st.success("System is healthy")
                    else:
                        st.error(
                            f"System is unhealthy: {data.get('error', 'Unknown error')}"
                        )

                    st.json(data)
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")

# Metadata Tab
with tabs[3]:
    st.markdown("Get metadata from a random document in the system.")

    if st.button("Get Random Metadata", key="metadata_btn"):
        with st.spinner("Fetching metadata..."):
            try:
                response = httpx.get(f"{API_URL}/random-metadata", timeout=10.0)

                if response.status_code == 200:
                    data = response.json()

                    st.subheader("Document Metadata")
                    st.json(data["metadata"])

                    st.subheader("Content Preview")
                    st.text(data["content_preview"])
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")

# Root Tab
with tabs[4]:
    st.markdown("Get API information and available endpoints.")

    if st.button("Get API Info", key="root_btn"):
        with st.spinner("Fetching API info..."):
            try:
                response = httpx.get(f"{API_URL}/", timeout=10.0)

                if response.status_code == 200:
                    data = response.json()

                    st.subheader(data["message"])
                    st.markdown(f"**Version:** {data['version']}")

                    st.subheader("Available Endpoints")
                    for endpoint, description in data["endpoints"].items():
                        st.markdown(f"- **{endpoint}**: {description}")

                    st.subheader("Example Usage")
                    st.json(data["examples"])
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")
