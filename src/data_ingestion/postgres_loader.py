"""
PostgreSQL Data Loader

This module handles loading cleaned student performance data into PostgreSQL database.
Designed for use with Airflow DAG for data warehouse integration.

Author: ES25DE01 Project Team
"""

import pandas as pd
import os
from sqlalchemy import create_engine, text
from typing import Optional


def get_postgres_connection_string(
    host: str = "localhost",
    port: int = 5432,
    database: str = "student_performance",
    user: str = "postgres",
    password: str = "postgres"
) -> str:
    """
    Construct PostgreSQL connection string.

    Args:
        host: PostgreSQL server host
        port: PostgreSQL server port
        database: Database name
        user: Database user
        password: Database password

    Returns:
        SQLAlchemy connection string
    """
    # Allow environment variables to override defaults
    host = os.getenv('POSTGRES_HOST', host)
    port = int(os.getenv('POSTGRES_PORT', port))
    database = os.getenv('POSTGRES_DB', database)
    user = os.getenv('POSTGRES_USER', user)
    password = os.getenv('POSTGRES_PASSWORD', password)

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def create_tables(engine):
    """
    Create database tables if they don't exist.

    Args:
        engine: SQLAlchemy engine
    """
    create_table_sql = """
    -- Drop existing tables if they exist
    DROP TABLE IF EXISTS student_performance_cleaned CASCADE;
    DROP TABLE IF EXISTS student_performance_abt CASCADE;

    -- Create cleaned data table
    CREATE TABLE IF NOT EXISTS student_performance_cleaned (
        id SERIAL PRIMARY KEY,
        school VARCHAR(10),
        sex VARCHAR(1),
        age INTEGER,
        address VARCHAR(1),
        famsize VARCHAR(10),
        Pstatus VARCHAR(1),
        Medu INTEGER,
        Fedu INTEGER,
        Mjob VARCHAR(20),
        Fjob VARCHAR(20),
        reason VARCHAR(20),
        guardian VARCHAR(20),
        traveltime INTEGER,
        studytime INTEGER,
        failures INTEGER,
        schoolsup VARCHAR(10),
        famsup VARCHAR(10),
        paid VARCHAR(10),
        activities VARCHAR(10),
        nursery VARCHAR(10),
        higher VARCHAR(10),
        internet VARCHAR(10),
        romantic VARCHAR(10),
        famrel INTEGER,
        freetime INTEGER,
        goout INTEGER,
        Dalc INTEGER,
        Walc INTEGER,
        health INTEGER,
        absences INTEGER,
        G1 INTEGER,
        G2 INTEGER,
        G3 INTEGER,
        course VARCHAR(20),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create ABT (Analytical Base Table)
    CREATE TABLE IF NOT EXISTS student_performance_abt (
        id SERIAL PRIMARY KEY,
        school VARCHAR(10),
        sex VARCHAR(1),
        age INTEGER,
        address VARCHAR(1),
        famsize VARCHAR(10),
        Pstatus VARCHAR(1),
        Medu INTEGER,
        Fedu INTEGER,
        Mjob VARCHAR(20),
        Fjob VARCHAR(20),
        reason VARCHAR(20),
        guardian VARCHAR(20),
        traveltime INTEGER,
        studytime INTEGER,
        failures INTEGER,
        schoolsup VARCHAR(10),
        famsup VARCHAR(10),
        paid VARCHAR(10),
        activities VARCHAR(10),
        nursery VARCHAR(10),
        higher VARCHAR(10),
        internet VARCHAR(10),
        romantic VARCHAR(10),
        famrel INTEGER,
        freetime INTEGER,
        goout INTEGER,
        Dalc INTEGER,
        Walc INTEGER,
        health INTEGER,
        absences INTEGER,
        G1 INTEGER,
        G2 INTEGER,
        G3 INTEGER,
        course VARCHAR(20),
        avg_prev_grade FLOAT,
        grade_trend FLOAT,
        high_absence INTEGER,
        target_pass INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_cleaned_course ON student_performance_cleaned(course);
    CREATE INDEX IF NOT EXISTS idx_cleaned_target ON student_performance_cleaned(G3);
    CREATE INDEX IF NOT EXISTS idx_abt_course ON student_performance_abt(course);
    CREATE INDEX IF NOT EXISTS idx_abt_target_pass ON student_performance_abt(target_pass);
    """

    with engine.connect() as conn:
        # Execute each statement separately
        for statement in create_table_sql.split(';'):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
        conn.commit()

    print("✓ Database tables created successfully")


def load_cleaned_data_to_postgres(
    csv_path: str = "data/cleaned/student_performance_clean.csv",
    connection_string: Optional[str] = None,
    if_exists: str = 'replace'
) -> int:
    """
    Load cleaned student performance data into PostgreSQL.

    Args:
        csv_path: Path to cleaned CSV file
        connection_string: PostgreSQL connection string (uses defaults if None)
        if_exists: How to behave if table exists ('fail', 'replace', 'append')

    Returns:
        Number of rows loaded
    """
    print("="*70)
    print("Loading Cleaned Data to PostgreSQL")
    print("="*70)

    # Read cleaned data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cleaned data not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Create database connection
    if connection_string is None:
        connection_string = get_postgres_connection_string()

    engine = create_engine(connection_string)

    # Create tables
    create_tables(engine)

    # Load data to PostgreSQL
    table_name = 'student_performance_cleaned'
    df.to_sql(table_name, engine, if_exists=if_exists, index=False, method='multi')

    print(f"✓ Loaded {len(df)} rows into table '{table_name}'")

    # Verify data
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        count = result.scalar()
        print(f"✓ Verified: {count} rows in database")

    print("="*70)
    return len(df)


def load_abt_to_postgres(
    csv_path: str = "data/processed/abt_student_performance.csv",
    connection_string: Optional[str] = None,
    if_exists: str = 'replace'
) -> int:
    """
    Load ABT (Analytical Base Table) into PostgreSQL.

    Args:
        csv_path: Path to ABT CSV file
        connection_string: PostgreSQL connection string (uses defaults if None)
        if_exists: How to behave if table exists ('fail', 'replace', 'append')

    Returns:
        Number of rows loaded
    """
    print("="*70)
    print("Loading ABT to PostgreSQL")
    print("="*70)

    # Read ABT data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ABT data not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Create database connection
    if connection_string is None:
        connection_string = get_postgres_connection_string()

    engine = create_engine(connection_string)

    # Load data to PostgreSQL
    table_name = 'student_performance_abt'
    df.to_sql(table_name, engine, if_exists=if_exists, index=False, method='multi')

    print(f"✓ Loaded {len(df)} rows into table '{table_name}'")

    # Verify data
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        count = result.scalar()
        print(f"✓ Verified: {count} rows in database")

    # Print sample statistics
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT
                COUNT(*) as total_students,
                SUM(CASE WHEN target_pass = 1 THEN 1 ELSE 0 END) as passed,
                AVG(G3) as avg_final_grade,
                AVG(avg_prev_grade) as avg_previous_grades
            FROM {table_name}
        """))
        stats = result.fetchone()
        print(f"\nDatabase Statistics:")
        print(f"  Total students: {stats[0]}")
        print(f"  Passed: {stats[1]} ({stats[1]/stats[0]*100:.1f}%)")
        print(f"  Average final grade (G3): {stats[2]:.2f}")
        print(f"  Average previous grades: {stats[3]:.2f}")

    print("="*70)
    return len(df)


if __name__ == "__main__":
    """
    Standalone execution: Load both cleaned data and ABT to PostgreSQL.
    """
    print("\n" + "="*70)
    print("POSTGRESQL DATA LOADER - STANDALONE EXECUTION")
    print("="*70 + "\n")

    try:
        # Load cleaned data
        rows_cleaned = load_cleaned_data_to_postgres()

        print("\n")

        # Load ABT
        rows_abt = load_abt_to_postgres()

        print("\n" + "="*70)
        print("DATA LOADING COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"  - Cleaned data: {rows_cleaned} rows")
        print(f"  - ABT data: {rows_abt} rows")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
