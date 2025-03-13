# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the lists of users for each activity
Chatting_user = ['HKY', 'LHJ', 'Guozhaonian', 'RenHongyu', 'Sunqiran', 'sulehan',
                 'LiaoJunjian', 'LHJ', 'TuYuzhao', 'yuchen', 'FengXuanqi',
                 'fupingyu', 'RenZhichen', 'WangYan', 'huangrenyi',
                 'ChenYongting', 'GuoYushan', 'liuxuya']

Pulling_trolley_user = ['TuYuzhao', 'Guozhaonian', 'fupingyu', 'FengXuanqi',
                        'WangYan', 'Sunqiran', 'LHJ', 'GuoYushan',
                        'ChenYongting', 'huangrenyi', 'sulehan', 'liuxuya',
                        'yuchen', 'LiaoJunjian', 'RenHongyu', 'RenZhichen', 'HKY']

Cleaning_whiteboard_user = ['RenHongyu', 'liuxuya', 'sulehan', 'GuoYushan',
                            'LHJ', 'RenZhichen', 'Guozhaonian', 'Sunqiran',
                            'fupingyu', 'yuchen', 'huangrenyi', 'WangYan',
                            'ChenYongting', 'HKY']

Sweep_user = ['sulehan', 'LHJ', 'TuYuzhao', 'Sunqiran', 'yuchen', 'FengXuanqi',
              'WangYan', 'huangrenyi', 'ChenYongting', 'LiaoJunjian', 'liuxuya',
              'RenZhichen', 'RenHongyu', 'Guozhaonian', 'fupingyu', 'GuoYushan', 'HKY']

Presenting_user = ['HKY', 'fupingyu', 'sulehan', 'yuchen', 'ChenYongting',
                   'WangYan', 'Sunqiran', 'GuoYushan', 'RenZhichen',
                   'liuxuya', 'huangrenyi', 'Guozhaonian']

News_interviewing_user = ['HKY', 'Guozhaonian', 'liuxuya', 'fupingyu',
                          'RenHongyu', 'sulehan', 'RenZhichen', 'huangrenyi',
                          'LiaoJunjian', 'GuoYushan', 'Sunqiran',
                          'ChenYongting', 'yuchen', 'WangYan']

# Remove duplicate entries in the Chatting_user list (e.g., 'LHJ' appears twice)
Chatting_user = list(set(Chatting_user))

# Compile a sorted list of all unique users
all_users = sorted(set(Chatting_user + Pulling_trolley_user + Cleaning_whiteboard_user +
                       Sweep_user + Presenting_user + News_interviewing_user))

# Initialize an empty list to store user participation data
data = []

# Populate the data list with dictionaries containing user participation
for user in all_users:
    data.append({
        'User': user,
        'Chatting': int(user in Chatting_user),
        'Pulling Trolley': int(user in Pulling_trolley_user),
        'Cleaning Whiteboard': int(user in Cleaning_whiteboard_user),
        'Sweep': int(user in Sweep_user),
        'Presenting': int(user in Presenting_user),
        'News Interviewing': int(user in News_interviewing_user)
    })

# Create a DataFrame from the data list
df = pd.DataFrame(data)

# Set 'User' as the index for better readability
df.set_index('User', inplace=True)

# Print the DataFrame to verify the data
print("User Participation Table:")
print(df)

# Visualize the DataFrame using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df, annot=True, cmap='YlGnBu', linewidths=0.5, linecolor='gray', cbar=False)
plt.title('User Participation Across Activities', fontsize=16)
plt.xlabel('Activities', fontsize=14)
plt.ylabel('Users', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('../result/user_participation_heatmap.png')
plt.show()


full_user = ['ChenYongting','GuoYushan','Guozhaonian','HKY','RenZhichen','Sunqiran','WangYan','fupingyu','huangrenyi','liuxuya','sulehan','yuchen']