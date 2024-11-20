import great_expectations as ge
import pandas as pd
import os
import json
from great_expectations.data_context import DataContext  # Correct import

# Set the current working directory
os.chdir(r'C:\Users\Lawrence\Downloads\GX')

# Load the avocado dataset
df = pd.read_csv('avocado.csv')

print("Dataset head:")
print(df.head())

# Set the path to your Great Expectations directory
context = DataContext(r'C:\Users\Lawrence\Downloads\GX\great_expectations')  # Adjust this path

# Create expectations
expectations = ge.dataset.PandasDataset(df)

# Create expectations
expectations.expect_column_to_exist("Date")
expectations.expect_column_values_to_be_of_type("Date", "object")
expectations.expect_column_to_exist("AveragePrice")
expectations.expect_column_values_to_be_of_type("AveragePrice", "float64")
expectations.expect_column_to_exist("Total Volume")
expectations.expect_column_values_to_be_of_type("Total Volume", "float64")

# Add more expectations
expectations.expect_column_values_to_not_be_null("Date")
expectations.expect_column_values_to_be_between("AveragePrice", min_value=0, max_value=10)
expectations.expect_column_values_to_be_greater_than("Total Volume", 0)

# Save expectations to a JSON file
with open("avocado_expectations.json", "w") as f:
    json.dump(expectations.get_expectation_suite().to_json_dict(), f, indent=2)

print("\nExpectations saved to 'avocado_expectations.json'")

# Validate the data
validation_results = expectations.validate()

# Print validation results
print("\nValidation results:")
print(json.dumps(validation_results, indent=2))

# Generate a simple HTML report
from great_expectations.render.view import DefaultJinjaPageView

document = DefaultJinjaPageView().render(validation_results)

with open("validation_report.html", "w") as f:
    f.write(document)

print("\nValidation report saved as 'validation_report.html'")
