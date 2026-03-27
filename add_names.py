import polars as pl
from faker import Faker

# Initialize Faker
fake = Faker()

# 1. Load your original dataset
print("Loading data...")
df = pl.read_csv("gym_data.csv")

# 2. Create a helper function to match names to the gender column
def generate_name(gender_value):
    # Standardizing to lowercase just in case the data is messy
    gender = str(gender_value).strip().lower()
    
    if gender in ['male', 'm']:
        return fake.name_male()
    elif gender in ['female', 'f']:
        return fake.name_female()
    else:
        return fake.name() # Generic random name as a fallback

# 3. Generate a list of names based on the 'Gender' column
print("Generating realistic names...")
# We convert the gender column to a Python list to process it quickly
gender_list = df["Gender"].to_list()
new_names = [generate_name(g) for g in gender_list]

# 4. Add the new 'name' column to your Polars dataframe
df = df.with_columns(pl.Series("name", new_names))

# 5. Save the updated file!
df.write_csv("gym_data_with_names.csv")
print("Done! Saved as gym_data_with_names.csv")