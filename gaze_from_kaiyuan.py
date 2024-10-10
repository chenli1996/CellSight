import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# path = r"Sunqiran_presenting.txt"
user_behavior_file_path = '../point_cloud_data/FSVVD/ACM_MM23 User Behavior Dataset with Tools/User Movement/'
user_behavior_file_name = 'Sunqiran_presenting.txt'
path = user_behavior_file_path + user_behavior_file_name
data = []
with open(path, "r", encoding='utf-8') as fr:
    columns = fr.readline().strip().split(', ')
    for line in fr.readlines()[1:]:
        data.append(list(map(lambda x: float(x), line.split(' '))))
data = np.array(data)
print(data.shape)

df = pd.DataFrame(data, columns=columns)
df.set_index('#Frame', inplace=True)


gaze_directions = np.stack(df.apply(lambda row: [row['REyeRX'], row['REyeRY'], row['REyeRZ']], axis=1).values)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot head positions with Y as the vertical axis
ax.plot(df['HeadX'], df['HeadZ'], df['HeadY'], label='Head Movement Trajectory')

ax.set_xlim3d(-1.5, 1.5)
ax.set_ylim3d(-1.5, 1.5)
ax.set_zlim3d(0, 2)


# Plot gaze directions with Y as the vertical axis
for head_position, gaze_direction in zip(df[['HeadX', 'HeadZ', 'HeadY']].values[::100], gaze_directions[::100]):

    yaw = gaze_direction[1] * np.pi / 180
    pitch = gaze_direction[0] * np.pi / 180

    x = np.cos(yaw) * np.cos(pitch)
    y = np.sin(yaw) * np.cos(pitch)
    z = np.sin(pitch)

    print(x, y, z)

    #ax.quiver(*head_position, x, y, z, length=0.36, normalize=True, color='r', arrow_length_ratio=0.5,linewidth=2)
    ax.quiver(*head_position, x, y, z, length=0.25, normalize=True, color='r', arrow_length_ratio=0.2, linewidth=2)
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
ax.tick_params(axis='z', labelsize=17)
ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_zlabel('Z', fontsize=20)


#ax.set_box_aspect([1,1,0.5])

# ax.grid(True)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# ax.set_title('Head Movement Trajectory and Gaze Direction')
ax.view_init(elev=27, azim=150)
plt.show()


