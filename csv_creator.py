import csv

# Define the header and data
header = ['Name', 'Family_Name', 'Email', 'Password']
data = [
    ['kasra', 'sehat', 'kasra.sehat@gmail.com', '12345'],
    ['ali', 'mmm', 'kasra.sehat@yahoo.com', '123456']
]

# Specify the filename
filename = '/root/kasra/projects/Chatbots/users.csv'

# Write to the CSV file
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header
    writer.writerows(data)   # Write the data rows

print(f"CSV file '{filename}' created successfully.")