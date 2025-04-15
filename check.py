import wrds
import pandas as pd
from datetime import datetime

# Connect to WRDS
db = wrds.Connection()


# Query to check the table schema
schema_query = """
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'crsp.msi';
"""

# Execute the query to inspect the schema
columns = db.raw_sql(schema_query)

# Print the columns to check available names
print(columns)
