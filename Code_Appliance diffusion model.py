#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display

# Global variables
global_yearly_connections = []
global_total_connections = None
global_early_adopters_by_year = []
global_late_adopters_by_year = []

# Input number of connections goal
print("Please select the number of connections goal by year 5:\n")

# Total connections widget
total_connections = widgets.IntText(description='Total connections', disabled=False, style={'description_width': '150px'})

# Button to trigger the function
submit_button1 = widgets.Button(description='Submit')

# Function to process the total connections and calculate yearly breakdown
def process_total_connections(b):
    global global_yearly_connections, global_total_connections  # Declare global variables

    # Access the input value
    global_total_connections = total_connections.value
    print(f'Total connections entered: {global_total_connections}')
    
    # Percentages for each year
    yearly_percentages = [0.28, 0.36, 0.18, 0.12, 0.06]
    yearly_connections = []

    # Calculate yearly connections based on the specified percentages
    cumulative_connections = 0
    for i, percentage in enumerate(yearly_percentages):
        # Calculate connections for the year
        yearly_value = round(percentage * global_total_connections)
        yearly_connections.append(yearly_value)
        cumulative_connections += yearly_value

    # Adjust final year to ensure total connections match the target
    difference = global_total_connections - cumulative_connections
    yearly_connections[-1] += difference  # Adjust the last year's connections value

    global_yearly_connections = yearly_connections  # Store the result in the global variable
    print(f'Yearly new customers calculated: {global_yearly_connections}')
    
    # Proceed to ask about early adopter percentage
    prompt_for_early_adopter_percentage()

# Connect button to function
submit_button1.on_click(process_total_connections)

# Display widgets
display(total_connections, submit_button1)

# Early Adopter Variables
default_early_adopter_percentage = 30
early_adopter_percentage = default_early_adopter_percentage  # Initial value

# Function to set or modify early adopter percentage
def set_early_adopter_percentage(new_percentage=None):
    global early_adopter_percentage
    early_adopter_percentage = new_percentage if new_percentage is not None else default_early_adopter_percentage
    print(f"Early Adopter percentage set to: {early_adopter_percentage}%")
    calculate_adopters_by_year()  # Calculate and display adopters based on the final percentage

# Function to calculate and display early and late adopters each year
def calculate_adopters_by_year():
    global global_early_adopters_by_year, global_late_adopters_by_year
    global_early_adopters_by_year = []
    global_late_adopters_by_year = []

    for yearly_connections in global_yearly_connections:
        # Calculate early and late adopters for the year
        early_adopters = round(early_adopter_percentage / 100 * yearly_connections)
        late_adopters = yearly_connections - early_adopters
        global_early_adopters_by_year.append(early_adopters)
        global_late_adopters_by_year.append(late_adopters)

    # Display results
    print("\nYearly Breakdown of Early and Late Adopters:")
    for year, (early, late) in enumerate(zip(global_early_adopters_by_year, global_late_adopters_by_year), 1):
        print(f"Year {year}: Early Adopters = {early}, Late Adopters = {late}")

# Function to prompt user for modifying early adopter percentage
def prompt_for_early_adopter_percentage():
    print('By default, Early Adopter users are considered to be 30% of the total connections. Would you like to modify that value?')
    modify_adopters_proportion = widgets.Dropdown(
        options=['Yes', 'No'],
        description='Modify adopters proportion?',
        style={'description_width': '250px'},
        layout=widgets.Layout(width='350px'),
        disabled=False
    )
    submit_button2 = widgets.Button(description='Submit')
    
    # Handle the selection from the dropdown
    def on_initial_button_click(b):
        response = modify_adopters_proportion.value
        if response == "Yes":
            # Input widget for custom early adopter percentage
            Early_Adopter = widgets.IntText(
                description='Early Adopter Users percentage',
                disabled=False,
                style={'description_width': '250px'}
            )
            submit_percentage_button = widgets.Button(description='Submit Percentage')

            # Define the function to process new percentage input
            def on_percentage_submit(b):
                set_early_adopter_percentage(Early_Adopter.value)

            # Connect button click to percentage submission
            submit_percentage_button.on_click(on_percentage_submit)
            
            # Display the IntText widget and the new submit button
            display(Early_Adopter, submit_percentage_button)
        else:
            set_early_adopter_percentage()  # Use default percentage and calculate adopters

    # Connect dropdown selection to function
    submit_button2.on_click(on_initial_button_click)
    
    # Display the dropdown and submit button
    display(modify_adopters_proportion, submit_button2)


# In[ ]:


import pandas as pd

# Define the tier distribution for Early Adopters and Late Adopters
distribution_EA = [
     [52.38, 16.67, 2.38, 23.81, 4.76],
[26.19, 11.90, 14.29, 42.86, 4.76],
[0.00, 9.52, 19.05, 66.67, 4.76],
[0.00, 0.00, 4.76, 78.57, 16.67],
[0.00, 0.00, 2.38, 57.14, 40.48],
[0.00, 0.00, 0.00, 38.12, 61.88],
[0.00, 0.00, 0.00, 24.06, 75.94],
[0.00, 0.00, 0.00, 15.63, 84.37],
[0.00, 0.00, 0.00, 10.69, 89.31],
[0.00, 0.00, 0.00, 7.83, 92.17],
[0.00, 0.00, 0.00, 6.19, 93.81],
[0.00, 0.00, 0.00, 5.25, 94.75],
[0.00, 0.00, 0.00, 4.72, 95.28],
[0.00, 0.00, 0.00, 4.41, 95.59],
[0.00, 0.00, 0.00, 4.24, 95.76],
[0.00, 0.00, 0.00, 4.14, 95.86],
[0.00, 0.00, 0.00, 4.08, 95.92],
[0.00, 0.00, 0.00, 4.05, 95.95],
[0.00, 0.00, 0.00, 4.03, 95.97],
[0.00, 0.00, 0.00, 4.02, 95.98],

]


# Late adopters scenarios: low, moderate, high
distribution_LA_low = [
    [98.02, 1.98, 0.00, 0.00, 0.00],
[97.03, 2.97, 0.00, 0.00, 0.00],
[94.06, 5.94, 0.00, 0.00, 0.00],
[77.23, 14.85, 3.96, 3.96, 0.00],
[45.54, 16.83, 16.83, 17.82, 2.97],
[39.79, 21.92, 18.42, 15.40, 4.47],
[33.55, 18.33, 18.54, 19.06, 10.53],
[27.60, 15.83, 18.42, 21.50, 16.66],
[22.53, 13.30, 18.42, 24.03, 21.72],
[18.30, 11.18, 18.42, 26.14, 25.95],
[14.90, 9.48, 18.42, 27.85, 29.36],
[12.23, 8.15, 18.42, 29.18, 32.02],
[10.19, 7.13, 18.42, 30.20, 34.07],
[8.66, 6.36, 18.42, 30.97, 35.60],
[7.52, 5.79, 18.42, 31.54, 36.74],
[6.68, 5.37, 18.42, 31.96, 37.58],
[6.06, 5.06, 18.42, 32.26, 38.19],
[5.62, 4.84, 18.42, 32.49, 38.64],
[5.30, 4.68, 18.42, 32.65, 38.96],
[5.06, 4.56, 18.42, 32.76, 39.19],

]


distribution_LA_moderate = [
      [98.02, 1.98, 0.00, 0.00, 0.00],
[97.03, 2.97, 0.00, 0.00, 0.00],
[94.06, 5.94, 0.00, 0.00, 0.00],
[77.23, 14.85, 3.96, 3.96, 0.00],
[45.54, 16.83, 16.83, 17.82, 2.97],
[39.07, 21.57, 18.40, 15.77, 5.19],
[31.98, 18.02, 18.42, 19.31, 12.27],
[25.26, 14.66, 18.42, 22.67, 19.00],
[19.17, 11.62, 18.42, 25.71, 25.09],
[13.89, 8.98, 18.42, 28.35, 30.37],
[9.48, 6.77, 18.42, 30.56, 34.77],
[5.91, 4.99, 18.42, 32.34, 38.34],
[3.10, 3.58, 18.42, 33.75, 41.16],
[0.92, 2.49, 18.42, 34.84, 43.34],
[0.00, 0.93, 18.03, 35.68, 45.36],
[0.00, 0.00, 16.33, 35.71, 47.96],
[0.00, 0.00, 13.91, 35.97, 50.12],
[0.00, 0.00, 12.22, 35.97, 51.81],
[0.00, 0.00, 11.03, 35.91, 53.06],
[0.00, 0.00, 10.04, 36.00, 53.95],
]


distribution_LA_high = [
         [98.02, 1.98, 0.00, 0.00, 0.00],
[97.03, 2.97, 0.00, 0.00, 0.00],
[94.06, 5.94, 0.00, 0.00, 0.00],
[77.23, 14.85, 3.96, 3.96, 0.00],
[45.54, 16.83, 16.83, 17.82, 2.97],
[38.53, 21.30, 18.39, 16.04, 5.75],
[30.87, 17.46, 18.42, 19.86, 13.39],
[23.37, 13.72, 18.42, 23.61, 20.88],
[16.37, 10.21, 18.42, 27.11, 27.89],
[10.10, 7.08, 18.42, 30.25, 34.16],
[4.70, 4.38, 18.42, 32.95, 39.56],
[0.19, 2.13, 18.42, 35.20, 44.06],
[0.00, 0.00, 12.66, 35.84, 51.50],
[0.00, 0.00, 5.49, 35.98, 58.53],
[0.00, 0.00, 0.00, 35.88, 64.12],
[0.00, 0.00, 0.00, 27.86, 72.14],
[0.00, 0.00, 0.00, 21.80, 78.20],
[0.00, 0.00, 0.00, 17.21, 82.79],
[0.00, 0.00, 0.00, 13.76, 86.24],
[0.00, 0.00, 0.00, 11.23, 88.77],
]


# Global variables to hold the selected scenario data and tables for later access
selected_early_adopters = []
selected_late_adopters = []
final_early_adopters_distribution = None
final_late_adopters_distribution = None

# Function to create and display tables for the selected senario
def display_selected_tables():
    global final_early_adopters_distribution, final_late_adopters_distribution
    
    df_EA = pd.DataFrame(selected_early_adopters, columns=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Tier 5'])
    df_EA.index = [f'Year {i+1}' for i in range(len(selected_early_adopters))]
    
    df_LA = pd.DataFrame(selected_late_adopters, columns=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Tier 5'])
    df_LA.index = [f'Year {i+1}' for i in range(len(selected_late_adopters))]
    
    print("Early Adopters Tier Distribution (%):")
    display(df_EA)
    print("\nLate Adopters Tier Distribution (%):")
    display(df_LA)
    
    final_early_adopters_distribution = df_EA
    final_late_adopters_distribution = df_LA

# Dropdown for scenario selection
scenario_dropdown = widgets.Dropdown(
    options=['low', 'moderate', 'high'],
    description='Select scenario:',
    style={'description_width': '150px'}
)

# Button to confirm the selection and display the data
confirm_button = widgets.Button(description="Confirm Scenario")

# Function to handle the selection and display results
def on_confirm_button_click(b):
    global selected_early_adopters, selected_late_adopters
    
    selected_early_adopters = distribution_EA
    
    # Assign selected_late_adopters based on dropdown selection
    if scenario_dropdown.value == 'low':
        selected_late_adopters = distribution_LA_low
    elif scenario_dropdown.value == 'moderate':
        selected_late_adopters = distribution_LA_moderate
    elif scenario_dropdown.value == 'high':
        selected_late_adopters = distribution_LA_high

    display_selected_tables()

# Link the button click event to the function
confirm_button.on_click(on_confirm_button_click)

# Display the dropdown and button
display(scenario_dropdown, confirm_button)


# In[ ]:


# Function to adjust each row to ensure the sum of customers in each cohort matches the specified total
def adjust_row_sum(df, total):
    adjusted_df = df.copy()

    for index in range(len(adjusted_df)):
        row_sum = adjusted_df.iloc[index].sum()
        if row_sum != total:
            difference = total - row_sum
            candidate_columns = [1, 2, 3]

            # Shuffle to randomize selection order
            random.shuffle(candidate_columns)
            for col in candidate_columns:
                if adjusted_df.iat[index, col] + difference >= 0:
                    adjusted_df.iat[index, col] += difference
                    break

    return adjusted_df


# In[ ]:


import random

#Tier Distribution for conections in  Year 1 (ha)

# Global variables to hold the final tables for later access
ha_EA_table = None
ha_LA_table = None



# Processing Early Adopters DataFrame for Year 1
ha_EA = (final_early_adopters_distribution / 100).multiply(global_early_adopters_by_year[0]).round().astype(int)
ha_EA = adjust_row_sum(ha_EA, global_early_adopters_by_year[0])

# Processing Late Adopters DataFrame for Year 1
ha_LA = (final_late_adopters_distribution / 100).multiply(global_late_adopters_by_year[0]).round().astype(int)
ha_LA = adjust_row_sum(ha_LA, global_late_adopters_by_year[0])

# Save tables to global variables for later access
ha_EA_table = ha_EA
ha_LA_table = ha_LA

# Optionally, save the tables as CSV files for persistent storage
ha_EA.to_csv("ha_EA.csv", index=True)  # Saves ha_EA with indices as row labels
ha_LA.to_csv("ha_LA.csv", index=True)  # Saves ha_LA with indices as row labels

# Display the adjusted tables
print("ha_EA (Early Adopters Tier Distribution for connections in Year 1):")
display(ha_EA_table)

print("\nha_LA (Late Adopters Tier Distribution for connections in Year 1):")
display(ha_LA_table)


# In[ ]:


#Tier Distribution for conections in  Year 2 (hb)

# Global variables to hold the final tables
hb_EA_table = None
hb_LA_table = None

# Processing Early Adopters DataFrame for Year 1
hb_EA = (final_early_adopters_distribution / 100).multiply(global_early_adopters_by_year[1]).round().astype(int)
hb_EA = adjust_row_sum(hb_EA, global_early_adopters_by_year[1]  )

# Processing Late Adopters DataFrame for Year 1
hb_LA = (final_late_adopters_distribution / 100).multiply(global_late_adopters_by_year[1]).round().astype(int)
hb_LA = adjust_row_sum(hb_LA, global_late_adopters_by_year[1])

# Fixing the shifting issue using pd.concat()
hb_EA_shifted = pd.concat([pd.DataFrame(0, index=[0], columns=hb_EA.columns), hb_EA.iloc[:-1]], ignore_index=True)
hb_LA_shifted = pd.concat([pd.DataFrame(0, index=[0], columns=hb_LA.columns), hb_LA.iloc[:-1]], ignore_index=True)

# Rename the index to "Year 1" to "Year 20"
hb_EA_shifted.index = [f"Year {i+1}" for i in range(len(hb_EA_shifted))]
hb_LA_shifted.index = [f"Year {i+1}" for i in range(len(hb_LA_shifted))]

# Save the adjusted tables to global variables
hb_EA_table = hb_EA_shifted
hb_LA_table = hb_LA_shifted

# Display the adjusted tables with proper row names
print("hb_EA (Early Adopters Tier Distribution for connections in Year 2):")
display(hb_EA_table)

print("\nhb_LA (Late Adopters Tier Distribution for connections in Year 2):")
display(hb_LA_table)


# In[ ]:


#Tier Distribution for conections in  Year 3 (hc)

# Global variables to hold the final tables for later access
hc_EA_table = None
hc_LA_table = None


# Processing Early Adopters DataFrame for Year 2
hc_EA = (final_early_adopters_distribution / 100).multiply(global_early_adopters_by_year[2]).round().astype(int)
hc_EA = adjust_row_sum(hc_EA, global_early_adopters_by_year[2])

# Processing Late Adopters DataFrame for Year 2
hc_LA = (final_late_adopters_distribution / 100).multiply(global_late_adopters_by_year[2]).round().astype(int)
hc_LA = adjust_row_sum(hc_LA, global_late_adopters_by_year[2])

# Shifting down each DataFrame by two rows, inserting rows of zeros at the top
hc_EA_shifted =  pd.concat([pd.DataFrame(0, index=[0, 1], columns=hc_EA.columns), hc_EA.iloc[:-2]], ignore_index=True)
hc_LA_shifted = pd.concat([pd.DataFrame(0, index=[0, 1], columns=hc_LA.columns), hc_LA.iloc[:-2]], ignore_index=True)



# Rename the index to "Year 1" to "Year 20"
hc_EA_shifted.index = [f"Year {i+1}" for i in range(20)]
hc_LA_shifted.index = [f"Year {i+1}" for i in range(20)]

# Save the adjusted tables to global variables
hc_EA_table = hc_EA_shifted
hc_LA_table = hc_LA_shifted

# Display the adjusted tables with proper row names
print("hc_EA (Early Adopters Tier Distribution for connections in Year 3):")
display(hc_EA_table)

print("\nhc_LA (Late Adopters Tier Distribution for connections in Year 3):")
display(hc_LA_table)


# In[ ]:


#Tier Distribution for conections in  Year 4 (hd)

# Global variables to hold the final tables for later access
hd_EA_table = None
hd_LA_table = None


# Processing Early Adopters DataFrame for Year 2
hd_EA = (final_early_adopters_distribution / 100).multiply(global_early_adopters_by_year[3]).round().astype(int)
hd_EA = adjust_row_sum(hd_EA, global_early_adopters_by_year[3])

# Processing Late Adopters DataFrame for Year 2
hd_LA = (final_late_adopters_distribution / 100).multiply(global_late_adopters_by_year[3]).round().astype(int)
hd_LA = adjust_row_sum(hd_LA, global_late_adopters_by_year[3])

# Shifting down each DataFrame by three rows, inserting rows of zeros at the top
hd_EA_shifted = pd.concat([pd.DataFrame(0, index=[0, 1, 2], columns=hd_EA.columns), hd_EA.iloc[:-3]], ignore_index=True)
hd_LA_shifted = pd.concat([pd.DataFrame(0, index=[0, 1, 2], columns=hd_LA.columns), hd_LA.iloc[:-3]], ignore_index=True)



# Rename the index to "Year 1" to "Year 20"
hd_EA_shifted.index = [f"Year {i+1}" for i in range(20)]
hd_LA_shifted.index = [f"Year {i+1}" for i in range(20)]

# Save the adjusted tables to global variables
hd_EA_table = hd_EA_shifted
hd_LA_table = hd_LA_shifted

# Display the adjusted tables with proper row names
print("hd_EA (Early Adopters Tier Distribution for connections in Year 4):")
display(hd_EA_table)

print("\nhd_LA (Late Adopters Tier Distribution for connections in Year 4):")
display(hd_LA_table)


# In[ ]:


#Tier Distribution for conections in  Year 5 (he)

# Global variables to hold the final tables for later access
he_EA_table = None
he_LA_table = None


# Processing Early Adopters DataFrame for Year 2
he_EA = (final_early_adopters_distribution / 100).multiply(global_early_adopters_by_year[4]).round().astype(int)
he_EA = adjust_row_sum(he_EA, global_early_adopters_by_year[4])

# Processing Late Adopters DataFrame for Year 2
he_LA = (final_late_adopters_distribution / 100).multiply(global_late_adopters_by_year[4]).round().astype(int)
he_LA = adjust_row_sum(he_LA, global_late_adopters_by_year[4])

# Shifting down each DataFrame by four rows, inserting rows of zeros at the top
he_EA_shifted = pd.concat([pd.DataFrame(0, index=[0, 1, 2, 3], columns=he_EA.columns), he_EA.iloc[:-4]], ignore_index=True)
he_LA_shifted = pd.concat([pd.DataFrame(0, index=[0, 1, 2, 3], columns=he_LA.columns), he_LA.iloc[:-4]], ignore_index=True)


# Rename the index to "Year 1" to "Year 20"
he_EA_shifted.index = [f"Year {i+1}" for i in range(20)]
he_LA_shifted.index = [f"Year {i+1}" for i in range(20)]

# Save the adjusted tables to global variables
he_EA_table = he_EA_shifted
he_LA_table = he_LA_shifted

# Display the adjusted tables with proper row names
print("he_EA (Late Adopters Tier Distribution for connections in Year 5):")
display(he_EA_table)

print("\nhe_LA (Late Adopters Tier Distribution for connections in Year 5):")
display(he_LA_table)


# In[ ]:


#Aggregation of customers in each tier 

# Initialize the summed table with zeros, matching the shape of one of the tables (e.g., ha_EA_table)
aggregation_table = pd.DataFrame(0, index=ha_EA_table.index, columns=ha_EA_table.columns)

# List of tables to sum, using the _table suffix
tables_to_sum = [
    ha_EA_table, ha_LA_table,
    hb_EA_table, hb_LA_table,
    hc_EA_table, hc_LA_table,
    hd_EA_table, hd_LA_table,
    he_EA_table, he_LA_table
]

# Sum all tables cell by cell
for table in tables_to_sum:
    aggregation_table += table

# Display the resulting summed table
print("Aggregation Table (from ha to he for both EA and LA):")
display(aggregation_table)


# In[ ]:


#Bar Plot

import matplotlib.pyplot as plt
import seaborn as sns

# Use the Muted color palette for a soft, professional look
colors = sns.color_palette("Blues", n_colors=len(aggregation_table.columns))

# Plot with customized colors
fig, ax = plt.subplots(figsize=(12, 8))
aggregation_table.plot(kind="bar", stacked=True, ax=ax, color=colors)

plt.title("Aggregation of customers", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of customers", fontsize=14)

#plt.yticks(range(0, 151, 10))  #adjust if needed

# Modify x-axis labels (1, 2, ..., 20)
ax.set_xticks(range(len(aggregation_table.index)))  # Set tick positions
ax.set_xticklabels(range(1, len(aggregation_table.index) + 1), fontsize=12)  # Set labels as numbers

plt.legend(title="Tiers", fontsize=11, title_fontsize=11)
plt.xticks(rotation=0)

plt.savefig("Aggregation of Connection by Tiers", dpi=300, bbox_inches="tight")
plt.show()


# In[ ]:


#Heatmap

plt.figure(figsize=(10, 8))
sns.heatmap(aggregation_table, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Cumulative Value'})
plt.title("Heatmap of Cumulative Values Across Years and Tiers")
plt.xlabel("Tiers")
plt.ylabel("Years")
plt.show()


# In[ ]:




