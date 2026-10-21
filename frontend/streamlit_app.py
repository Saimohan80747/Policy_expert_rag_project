# from pathlib import Path

# import requests
# import streamlit as st


# BASE_URL = "http://127.0.0.1:8000"

# st.set_page_config(page_title="RAG Chat", page_icon="🤖", layout="wide")

# if "session_id" not in st.session_state:
#     st.session_state.session_id = ""
# if "chat_items" not in st.session_state:
#     st.session_state.chat_items = []


# def api_get(path: str):
#     return requests.get(f"{BASE_URL}{path}", timeout=60)


# def api_post(path: str, **kwargs):
#     return requests.post(f"{BASE_URL}{path}", timeout=180, **kwargs)


# def api_delete(path: str):
#     return requests.delete(f"{BASE_URL}{path}", timeout=60)


# def load_session(session_id: str) -> None:
#     response = api_get(f"/sessions/{session_id}")
#     if response.status_code != 200:
#         st.error(response.text)
#         return

#     st.session_state.session_id = session_id
#     st.session_state.chat_items = response.json().get("session", [])


# def download_source(file_name: str | None, file_url: str | None) -> None:
#     if not file_url:
#         return

#     try:
#         response = requests.get(
#             f"http://127.0.0.1:8000/download/{file_name}"
#         )
        
#         if response.status_code == 200:

#             st.download_button(
#                 "📥 Download",
#                 data=response.content,
#                 file_name=file_name,
#                 mime="application/pdf"
#             )
#     except Exception:
#         pass

# def preview_source(file_name: str | None, file_url: str | None) -> None:
#     if not file_url:
#         return

#     try:
#         response = requests.get(
#             f"http://127.0.0.1:8000/preview/{file_name}"
#         )
#         if response.status_code == 200:
#             st.text_area(
#                 "Preview",
#                 response.json()["text"],
#                 height=300
#             )
#     except Exception:
#         pass

# with st.sidebar:
#     st.title("Document")

#     uploaded_file = st.file_uploader("Upload File", type=["pdf", "docx", "txt", "csv", "xlsx"])

#     if st.button("Upload Document"):
#         if uploaded_file is None:
#             st.warning("Select a file first.")
#         else:
#             files = {
#                 "file": (
#                     uploaded_file.name,
#                     uploaded_file,
#                     uploaded_file.type or "application/octet-stream",
#                 )
#             }
#             response = api_post("/pipeline", files=files)
#             if response.status_code == 200:
#                 st.success("Document uploaded successfully.")
#             else:
#                 st.error(response.text)

#     st.divider()
#     st.subheader("Chat History")
#     history = api_get("/history")

#     if history.status_code == 200:
#         sessions = history.json().get("sessions", [])
#         if not sessions:
#             st.caption("No previous chats.")

#         for session in sessions:
#             col1, col2 = st.columns([5, 1])
#             with col1:
#                 if st.button(session.get("title", "Untitled"), key=session["id"]):
#                     load_session(session["id"])
#                     st.rerun()
#             with col2:
#                 if st.button("🗑", key=f"delete_{session['id']}"):
#                     api_delete(f"/sessions/{session['id']}")
#                     if st.session_state.session_id == session["id"]:
#                         st.session_state.session_id = ""
#                         st.session_state.chat_items = []
#                     st.rerun()

# st.title("RAG Chat Assistant")

# for item in st.session_state.chat_items:
#     with st.chat_message("user"):
#         st.markdown(item.get("question", ""))
#     with st.chat_message("assistant"):
#         st.markdown(item.get("answer", ""))
#         if item.get("file_name"):
#             with st.expander("📄 Source Preview"):
#                 if st.button("Load Preview", key=f"preview_{item['file_name']}"):
#                     preview_source(
#                         item["file_name"],
#                         item["file_url"]
#                     )

#                 download_source(
#                     item["file_name"],
#                     item["file_url"]
#                 )

# question = st.chat_input("Ask your question...")

# if question:
#     with st.chat_message("user"):
#         st.markdown(question)

#     response = api_post(
#         "/prompt",
#         json={"question": question, "session_id": st.session_state.session_id},
#     )

#     if response.status_code == 200:
#         result = response.json()
#         st.session_state.session_id = result.get("session_id", "")
#         st.session_state.chat_items.append(
#             {
#                 "question": question,
#                 "answer": result.get("answer", ""),
#                 "file_name": result.get("file_name"),
#                 "file_url": result.get("file_url"),
#             }
#         )

#         with st.chat_message("assistant"):
#             st.markdown(result.get("answer", ""))
#             if result.get("file_name"):
#                 preview_source(result.get("file_name"), result.get("file_url"))
#                 download_source(result.get("file_name"), result.get("file_url"))
#     else:
#         st.error(response.text)