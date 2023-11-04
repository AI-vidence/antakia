import pandas as pd

# create a sample DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# create a list of values for the new column
new_column_values = [7, 8, 9]

# insert the new column into the DataFrame
df.insert(loc=1, column='symbol', value=df.index)

# print the updated DataFrame
print(df)
