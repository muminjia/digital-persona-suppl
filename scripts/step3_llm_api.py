import os

from openai import OpenAI
import pandas as pd

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Load CSV
df = pd.read_csv("bg_20users/810494.0_avars.csv")

# Convert rows to prompt text
rows_text = ""
for _, row in df.iterrows():
    rows_text += f"""
question_id: {row['question_id']}
variable_label: {row['variable_label']}
answer: {row['answer']}
categories: {row['categories']}
"""

prompt = f"""
You are a data analyst generating a concise user profile from survey responses.

Each row contains:
- question_id
- variable_label
- answer
- categories (mapping if available)

Instructions:
1. Translate numeric answers using categories if available.
2. Ignore missing values.
3. Convert the information into natural language.
4. Produce a short user profile paragraph.

Dataset rows:
{rows_text}

Output:
User Profile:
"""

response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[
        {"role": "system", "content": "You summarize structured survey data into user profiles."},
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

print(response.choices[0].message.content)