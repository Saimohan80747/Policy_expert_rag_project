"""Streamlit editor for creating knowledge-graph nodes and relationships."""

import os
import csv
import io
from typing import Any

import requests
import streamlit as st


DEFAULT_API_URL = "http://127.0.0.1:8000"

NODE_TYPES: dict[str, dict[str, Any]] = {
    "Employee": {
        "endpoint": "/nodes/employee",
        "fields": [
            ("emp_id", "Employee ID"),
            ("name", "Name"),
            ("email", "Email"),
            ("role", "Role"),
            ("department", "Department"),
            ("join_date", "Join date (YYYY-MM-DD)"),
            ("description", "Description"),
            ("resume_drive_link", "Resume link"),
            ("linkedin_url", "LinkedIn URL"),
        ],
    },
    "Department": {
        "endpoint": "/nodes/department",
        "fields": [
            ("dept_id", "Department ID"),
            ("name", "Name"),
            ("description", "Description"),
        ],
    },
    "Project": {
        "endpoint": "/nodes/project",
        "fields": [
            ("project_id", "Project ID"),
            ("name", "Name"),
            ("description", "Description"),
            ("start_date", "Start date (YYYY-MM-DD)"),
            ("end_date", "End date (YYYY-MM-DD)"),
        ],
    },
    "Intern": {
        "endpoint": "/nodes/intern",
        "fields": [
            ("id", "Intern ID"),
            ("name", "Name"),
            ("email", "Email"),
            ("role", "Role"),
            ("department", "Department"),
            ("join_date", "Join date (YYYY-MM-DD)"),
            ("description", "Description"),
            ("resume_drive_link", "Resume link"),
            ("linkedin_url", "LinkedIn URL"),
        ],
    },
    "SubDepartment": {
        "endpoint": "/nodes/subdepartment",
        "fields": [
            ("subdept_id", "Subdepartment ID"),
            ("name", "Name"),
            ("description", "Description"),
        ],
    },
    "Team": {
        "endpoint": "/nodes/team",
        "fields": [
            ("team_id", "Team ID"),
            ("name", "Name"),
            ("description", "Description"),
        ],
    },
    "Policy": {
        "endpoint": "/nodes/policy",
        "fields": [
            ("policy_id", "Policy ID"),
            ("name", "Name"),
            ("description", "Description"),
        ],
    },
}

RELATIONSHIP_TYPES: dict[str, dict[str, Any]] = {
    "Department has subdepartment": {
        "endpoint": "/relationships/department-subdepartment",
        "fields": [("department", "Department name"), ("subdepartment", "Subdepartment name")],
    },
    "Subdepartment has team": {
        "endpoint": "/relationships/subdepartment-team",
        "fields": [("subdepartment", "Subdepartment name"), ("team", "Team name")],
    },
    "Team has employee": {
        "endpoint": "/relationships/team-employee",
        "fields": [("team", "Team name"), ("employee", "Employee name")],
    },
    "Employee reports to employee": {
        "endpoint": "/relationships/employee-employee",
        "fields": [("employee1", "Reporting employee"), ("employee2", "Manager employee")],
    },
    "Intern reports to employee": {
        "endpoint": "/relationships/intern-employee",
        "fields": [("intern", "Intern name"), ("employee", "Employee name")],
    },
    "Policy prepared by employee": {
        "endpoint": "/relationships/policy-employee",
        "fields": [("policy", "Policy name"), ("employee", "Employee name")],
    },
    "Employees are teammates": {
        "endpoint": "/relationships/teammate",
        "fields": [("employee1", "First employee"), ("employee2", "Second employee")],
    },
    "Team works on project": {
        "endpoint": "/relationships/team-project",
        "fields": [("team", "Team name"), ("project", "Project name")],
    },
}


def _initialise_state() -> None:
    if "pending_nodes" not in st.session_state:
        st.session_state.pending_nodes = []
    if "pending_relationships" not in st.session_state:
        st.session_state.pending_relationships = []


def _records_to_csv(records: list[dict[str, str]]) -> bytes:
    if not records:
        raise ValueError("At least one record is required.")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode("utf-8")


def _submit_csv(api_url: str, endpoint: str, csv_content: bytes) -> bool:
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}{endpoint}",
            files={"file": ("records.csv", csv_content, "text/csv")},
            timeout=180,
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc}")
        return False

    if response.ok:
        try:
            result = response.json()
        except ValueError:
            result = {"message": response.text}
        st.success(result.get("message", "Records created successfully."))
        st.json(result)
        return True
    else:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        st.error(f"Backend rejected the request ({response.status_code}): {detail}")
        return False


def _uploaded_csv_rows(uploaded_file: Any, required_fields: list[str]) -> list[dict[str, str]]:
    try:
        text = uploaded_file.getvalue().decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text)))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"CSV could not be read: {exc}") from exc

    if not rows:
        raise ValueError("CSV must contain a header and at least one data row.")

    headers = set(rows[0])
    missing = [field for field in required_fields if field not in headers]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    if any(not all(str(row.get(field, "")).strip() for field in required_fields) for row in rows):
        raise ValueError("Every required field must have a value in every CSV row.")
    return rows


def _render_csv_input(api_url: str, config: dict[str, Any], key_prefix: str) -> None:
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key=f"{key_prefix}_csv",
        help="The first row must contain the field names shown below.",
    )
    st.caption("Required columns: " + ", ".join(field for field, _ in config["fields"]))

    if uploaded_file is None:
        return

    try:
        rows = _uploaded_csv_rows(
            uploaded_file,
            [field for field, _ in config["fields"]],
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    st.dataframe(rows, use_container_width=True, hide_index=True)
    if st.button("Send CSV to backend", key=f"{key_prefix}_submit", type="primary"):
        if _submit_csv(api_url, config["endpoint"], uploaded_file.getvalue()):
            st.success(f"Uploaded {len(rows)} row(s).")


def _render_node_editor(api_url: str) -> None:
    node_type = st.selectbox("Node type", list(NODE_TYPES), key="node_type")
    node_config = NODE_TYPES[node_type]
    input_mode = st.radio(
        "Input method",
        ["Values", "CSV file"],
        horizontal=True,
        key="node_input_mode",
    )

    if input_mode == "CSV file":
        _render_csv_input(api_url, node_config, "node")
        return

    with st.form("node_form", clear_on_submit=True):
        values: dict[str, str] = {}
        for field_name, label in node_config["fields"]:
            values[field_name] = st.text_input(label, key=f"node_{field_name}")
        add_node = st.form_submit_button("Add node to queue")

    if add_node:
        missing = [
            label
            for (field_name, label) in node_config["fields"]
            if not values[field_name].strip()
        ]
        if missing:
            st.warning(f"Complete these fields before queueing: {', '.join(missing)}")
        else:
            queued_endpoints = {
                item["endpoint"] for item in st.session_state.pending_nodes
            }
            if queued_endpoints and node_config["endpoint"] not in queued_endpoints:
                st.warning(
                    "Submit or clear the current queue before adding a different node type."
                )
            else:
                st.session_state.pending_nodes.append(
                    {
                        "type": node_type,
                        "endpoint": node_config["endpoint"],
                        "data": values,
                    }
                )
                st.success(f"{node_type} added to the queue.")

    pending = st.session_state.pending_nodes
    if pending:
        st.subheader("Queued nodes")
        st.dataframe(
            [{"type": item["type"], **item["data"]} for item in pending],
            use_container_width=True,
            hide_index=True,
        )
        selected_endpoint = pending[0]["endpoint"]
        if any(item["endpoint"] != selected_endpoint for item in pending):
            st.warning("Queue nodes of the same type together; the backend route is type-specific.")
        else:
            csv_content = _records_to_csv([item["data"] for item in pending])
            st.caption("CSV that will be sent to the backend:")
            st.code(csv_content.decode("utf-8"), language="csv")
            if st.button("Submit nodes", type="primary"):
                submitted = _submit_csv(
                    api_url,
                    selected_endpoint,
                    csv_content,
                )
                if submitted:
                    st.session_state.pending_nodes = []
                    st.rerun()
        if st.button("Remove all queued nodes"):
            st.session_state.pending_nodes = []
            st.rerun()


def _render_relationship_editor(api_url: str) -> None:
    relationship_type = st.selectbox(
        "Relationship type", list(RELATIONSHIP_TYPES), key="relationship_type"
    )
    relationship_config = RELATIONSHIP_TYPES[relationship_type]
    input_mode = st.radio(
        "Input method",
        ["Values", "CSV file"],
        horizontal=True,
        key="relationship_input_mode",
    )

    if input_mode == "CSV file":
        _render_csv_input(api_url, relationship_config, "relationship")
        return

    with st.form("relationship_form", clear_on_submit=True):
        values = {
            field_name: st.text_input(label, key=f"relationship_{field_name}")
            for field_name, label in relationship_config["fields"]
        }
        add_relationship = st.form_submit_button("Add relationship to queue")

    if add_relationship:
        missing = [
            label
            for field_name, label in relationship_config["fields"]
            if not values[field_name].strip()
        ]
        if missing:
            st.warning(f"Complete these fields before queueing: {', '.join(missing)}")
        else:
            queued_endpoints = {
                item["endpoint"] for item in st.session_state.pending_relationships
            }
            if (
                queued_endpoints
                and relationship_config["endpoint"] not in queued_endpoints
            ):
                st.warning(
                    "Submit or clear the current queue before adding a different relationship type."
                )
            else:
                st.session_state.pending_relationships.append(
                    {
                        "type": relationship_type,
                        "endpoint": relationship_config["endpoint"],
                        "data": values,
                    }
                )
                st.success("Relationship added to the queue.")

    pending = st.session_state.pending_relationships
    if pending:
        st.subheader("Queued relationships")
        st.dataframe(
            [{"type": item["type"], **item["data"]} for item in pending],
            use_container_width=True,
            hide_index=True,
        )
        selected_endpoint = pending[0]["endpoint"]
        if any(item["endpoint"] != selected_endpoint for item in pending):
            st.warning(
                "Queue relationships of the same type together; the backend route is type-specific."
            )
        else:
            csv_content = _records_to_csv([item["data"] for item in pending])
            st.caption("CSV that will be sent to the backend:")
            st.code(csv_content.decode("utf-8"), language="csv")
            if st.button("Submit relationships", type="primary"):
                submitted = _submit_csv(
                    api_url,
                    selected_endpoint,
                    csv_content,
                )
                if submitted:
                    st.session_state.pending_relationships = []
                    st.rerun()
        if st.button("Remove all queued relationships"):
            st.session_state.pending_relationships = []
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Knowledge Graph Editor",
        page_icon="🔗",
        layout="wide",
    )
    _initialise_state()

    st.title("Knowledge Graph Editor")
    st.caption("Create nodes and relationships through the existing graph API.")

    with st.sidebar:
        st.header("Connection")
        api_url = st.text_input(
            "Backend API URL",
            value=os.getenv("GRAPH_API_URL", DEFAULT_API_URL),
            help="Only the URL is configurable here; this frontend does not connect to Neo4j.",
        )
        st.info("Create nodes before adding relationships that reference them.")

    nodes_tab, relationships_tab = st.tabs(["Add nodes", "Add relationships"])
    with nodes_tab:
        _render_node_editor(api_url)
    with relationships_tab:
        _render_relationship_editor(api_url)


if __name__ == "__main__":
    main()
