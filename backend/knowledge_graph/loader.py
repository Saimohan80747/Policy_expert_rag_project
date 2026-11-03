import pandas as pd
from .db import create_driver
from sentence_transformers import SentenceTransformer



model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



# DATA_PATH = "backend/knowledge_graph/data/"

# # Node CSV files
# employee_data = pd.read_csv(DATA_PATH + "employee_data.csv")
# department_data = pd.read_csv(DATA_PATH + "department_data.csv")
# project_data = pd.read_csv(DATA_PATH + "project_data.csv")
# intern_data = pd.read_csv(DATA_PATH + "intern_data.csv")
# subdepartment_data = pd.read_csv(DATA_PATH + "subdepartment_data.csv")
# teams_data = pd.read_csv(DATA_PATH + "teams_data.csv")
# policy_data = pd.read_csv(DATA_PATH + "policy_data.csv")

# # Relationship CSV files
# department_subdepartment_data = pd.read_csv(DATA_PATH + "department_subdepartment_data.csv")
# subdepartment_team_data = pd.read_csv(DATA_PATH + "subdepartment_team_data.csv")
# team_employee_data = pd.read_csv(DATA_PATH + "team_employee_data.csv")
# employee_employee_data = pd.read_csv(DATA_PATH + "employee_employee_data.csv")
# intern_employee_data = pd.read_csv(DATA_PATH + "intern_employee_data.csv")
# policy_employee_data = pd.read_csv(DATA_PATH + "policy_employee_data.csv")
# teammate_data = pd.read_csv(DATA_PATH + "teammate_data.csv")
# team_project_data = pd.read_csv(DATA_PATH + "team_project_data.csv")




def load_constraints(driver):
    constraints = [
        "CREATE CONSTRAINT employee_name IF NOT EXISTS FOR (e:Employee) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT department_name IF NOT EXISTS FOR (d:Department) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT project_name IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT intern_name IF NOT EXISTS FOR (i:Intern) REQUIRE i.name IS UNIQUE",
        "CREATE CONSTRAINT subdepartment_name IF NOT EXISTS FOR (s:SubDepartment) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT team_name IF NOT EXISTS FOR (t:Team) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT policy_name IF NOT EXISTS FOR (p:Policy) REQUIRE p.name IS UNIQUE"
    ]

    for query in constraints:
        driver.execute_query(query)

    print("Constraints created successfully!")

def load_employee_data(driver, data):
    for _, row in data.iterrows():

        description = str(row["description"])
        embedding = model.encode(description).tolist()

        driver.execute_query(
            """
            MERGE (e:Employee {name: $name})
            SET e.emp_id = $emp_id,
                e.email = $email,
                e.role = $role,
                e.department = $department,
                e.join_date = date($join_date),
                e.description = $description,
                e.resume_drive_link = $resume_drive_link,
                e.linkedin_url = $linkedin_url,
                e.embedding = $embedding
            """,
            emp_id=row["emp_id"],
            name=row["name"],
            email=row["email"],
            role=row["role"],
            department=row["department"],
            join_date=row["join_date"],
            description=description,
            resume_drive_link=row["resume_drive_link"],
            linkedin_url=row["linkedin_url"],
            embedding=embedding
        )

    print("Employee data loaded successfully!")


def load_department_data(driver, data):
    for _, row in data.iterrows():

        description = str(row["description"])
        embedding = model.encode(description).tolist()

        driver.execute_query(
            """
            MERGE (d:Department {name: $name})
            SET d.dept_id = $dept_id,
                d.description = $description,
                d.embedding = $embedding
            """,
            dept_id=row["dept_id"],
            name=row["name"],
            description=description,
            embedding=embedding
        )

    print("Department data loaded successfully!")


def load_project_data(driver, data):
    for _, row in data.iterrows():

        description = str(row["description"])
        embedding = model.encode(description).tolist()

        driver.execute_query(
            """
            MERGE (p:Project {name: $name})
            SET p.project_id = $project_id,
                p.description = $description,
                p.start_date = date($start_date),
                p.end_date = date($end_date),
                p.embedding = $embedding
            """,
            project_id=row["project_id"],
            name=row["name"],
            description=description,
            start_date=row["start_date"],
            end_date=row["end_date"],
            embedding=embedding
        )

    print("Project data loaded successfully!")


def load_intern_data(driver, data):
    for _, row in data.iterrows():

        description = str(row["description"])
        embedding = model.encode(description).tolist()

        driver.execute_query(
            """
            MERGE (i:Intern {name: $name})
            SET i.id = $id,
                i.email = $email,
                i.role = $role,
                i.department = $department,
                i.join_date = date($join_date),
                i.description = $description,
                i.resume_drive_link = $resume_drive_link,
                i.linkedin_url = $linkedin_url,
                i.embedding = $embedding
            """,
            id=row["id"],
            name=row["name"],
            email=row["email"],
            role=row["role"],
            department=row["department"],
            join_date=row["join_date"],
            description=description,
            resume_drive_link=row["resume_drive_link"],
            linkedin_url=row["linkedin_url"],
            embedding=embedding
        )

    print("Intern data loaded successfully!")


def load_subdepartment_data(driver, data):
    for _, row in data.iterrows():

        description = str(row["description"])
        embedding = model.encode(description).tolist()

        driver.execute_query(
            """
            MERGE (s:SubDepartment {name: $name})
            SET s.subdept_id = $subdept_id,
                s.description = $description,
                s.embedding = $embedding
            """,
            subdept_id=row["subdept_id"],
            name=row["name"],
            description=description,
            embedding=embedding
        )

    print("SubDepartment data loaded successfully!")


def load_teams_data(driver, data):
    for _, row in data.iterrows():

        description = str(row["description"])
        embedding = model.encode(description).tolist()

        driver.execute_query(
            """
            MERGE (t:Team {name: $name})
            SET t.team_id = $team_id,
                t.description = $description,
                t.embedding = $embedding
            """,
            team_id=row["team_id"],
            name=row["name"],
            description=description,
            embedding=embedding
        )

    print("Teams data loaded successfully!")


def load_policy_data(driver, data):
    for _, row in data.iterrows():

        description = str(row["description"])
        embedding = model.encode(description).tolist()

        driver.execute_query(
            """
            MERGE (p:Policy {name: $name})
            SET p.policy_id = $policy_id,
                p.description = $description,
                p.embedding = $embedding
            """,
            policy_id=row["policy_id"],
            name=row["name"],
            description=description,
            embedding=embedding
        )

    print("Policy data loaded successfully!")



def load_department_subdepartment_relationship(driver, data):
    for _, row in data.iterrows():

        driver.execute_query(
            """
            MATCH (d:Department {name: $department})
            MATCH (s:SubDepartment {name: $subdepartment})
            MERGE (d)-[:HAS_SUBDEPARTMENT]->(s)
            """,
            department=row["department"],
            subdepartment=row["subdepartment"]
        )

    print("Department-SubDepartment relationships loaded!")


def load_subdepartment_team_relationship(driver, data):
    for _, row in data.iterrows():

        driver.execute_query(
            """
            MATCH (s:SubDepartment {name: $subdepartment})
            MATCH (t:Team {name: $team})
            MERGE (s)-[:HAS_TEAM]->(t)
            """,
            subdepartment=row["subdepartment"],
            team=row["team"]
        )

    print("SubDepartment-Team relationships loaded!")


def load_team_employee_relationship(driver, data):
    for _, row in data.iterrows():

        driver.execute_query(
            """
            MATCH (t:Team {name: $team})
            MATCH (e:Employee {name: $employee})
            MERGE (t)-[:HAS_EMPLOYEE]->(e)
            """,
            team=row["team"],
            employee=row["employee"]
        )

    print("Team-Employee relationships loaded!")


def load_employee_employee_relationship(driver, data):
    for _, row in data.iterrows():

        driver.execute_query(
            """
            MATCH (e1:Employee {name: $employee1})
            MATCH (e2:Employee {name: $employee2})
            MERGE (e1)-[:REPORTS_TO]->(e2)
            """,
            employee1=row["employee1"],
            employee2=row["employee2"]
        )

    print("Employee reporting relationships loaded!")


def load_intern_employee_relationship(driver, data):
    for _, row in data.iterrows():

        driver.execute_query(
            """
            MATCH (i:Intern {name: $intern})
            MATCH (e:Employee {name: $employee})
            MERGE (i)-[:REPORTS_TO]->(e)
            """,
            intern=row["intern"],
            employee=row["employee"]
        )

    print("Intern-Employee relationships loaded!")


def load_policy_employee_relationship(driver, data):
    for _, row in data.iterrows():

        driver.execute_query(
            """
            MATCH (p:Policy {name: $policy})
            MATCH (e:Employee {name: $employee})
            MERGE (p)-[:PREPARED_BY]->(e)
            """,
            policy=row["policy"],
            employee=row["employee"]
        )

    print("Policy-Employee relationships loaded!")


def load_teammate_relationship(driver, data):
    for _, row in data.iterrows():

        driver.execute_query(
            """
            MATCH (e1:Employee {name: $employee1})
            MATCH (e2:Employee {name: $employee2})
            MERGE (e1)-[:TEAMMATE_IS]->(e2)
            """,
            employee1=row["employee1"],
            employee2=row["employee2"]
        )

    print("Teammate relationships loaded!")


def load_team_project_relationship(driver, data):
    for _, row in data.iterrows():

        driver.execute_query(
            """
            MATCH (t:Team {name: $team})
            MATCH (p:Project {name: $project})
            MERGE (t)-[:WORKS_ON]->(p)
            """,
            team=row["team"],
            project=row["project"]
        )

    print("Team-Project relationships loaded!")


# def load_nodes(driver):
#     load_employee_data(driver, employee_data)
#     load_department_data(driver, department_data)
#     load_project_data(driver, project_data)
#     load_intern_data(driver, intern_data)
#     load_subdepartment_data(driver, subdepartment_data)
#     load_teams_data(driver, teams_data)
#     load_policy_data(driver, policy_data)




# def load_relationships(driver):
#     load_department_subdepartment_relationship(driver, department_subdepartment_data)
#     load_subdepartment_team_relationship(driver, subdepartment_team_data)
#     load_team_employee_relationship(driver, team_employee_data)
#     load_employee_employee_relationship(driver, employee_employee_data)
#     load_intern_employee_relationship(driver, intern_employee_data)
#     load_policy_employee_relationship(driver, policy_employee_data)
#     load_teammate_relationship(driver, teammate_data)
#     load_team_project_relationship(driver, team_project_data)



# def load_all(driver):
#     load_constraints(driver)

#     load_nodes(driver)

#     load_relationships(driver)

#     print("Entire knowledge graph loaded successfully!")



if __name__ == "__main__":

    driver = create_driver()

    try:
        load_all(driver)
    finally:
        driver.close()