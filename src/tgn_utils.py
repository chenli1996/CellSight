def occlusion_mapping(R_c, alpha, beta, S_c):
    """
    Maps R(c) to an integer value based on given thresholds.

    :param R_c:   The R(c) value to map
    :param alpha: List or tuple of alpha values [α0, α1, α2]
    :param beta:  The beta value
    :param S_c:   The S(c) value
    :return:      The mapped integer value
    """
    thresholds = [a * (beta ** (S_c - 1)) for a in alpha]
    # print(thresholds)

    if R_c < thresholds[0]:
        return 0
    elif thresholds[0] <= R_c < thresholds[1]:
        return 1
    elif thresholds[1] <= R_c < thresholds[2]:
        return 2
    elif thresholds[2] <= R_c:
        return 3

# Example usage:
# alpha_values = [alpha0, alpha1, alpha2]  # Replace with actual values for α0, α1, and α2
# beta_value = beta  # Replace with the actual value for β
# S_c_value = S_c  # Replace with the actual value for S(c)
# R_c_value = R_c  # Replace with the actual value for R(c)

# result = occlusion_mapping(R_c_value, alpha_values, beta_value, S_c_value)
# print(result)


import numpy as np

def ray_box_intersection(ray_origin, ray_direction, box_min, box_max):
    """
    Check if a ray intersects with a box (AABB).
    :param ray_origin: Origin of the ray (np.array)
    :param ray_direction: Direction of the ray (np.array)
    :param box_min: Minimum vertex of the box (np.array)
    :param box_max: Maximum vertex of the box (np.array)
    :return: Boolean indicating if there is an intersection
    """
    t_min = (box_min - ray_origin) / ray_direction
    t_max = (box_max - ray_origin) / ray_direction
    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)
    t_near = np.max(t1)
    t_far = np.min(t2)

    if t_near > t_far or t_far < 0:
        return False

    return True

def occlusion_count(viewpoint, target_cell_center, cell_size, target_distance):
    """
    Count the number of surrounding cells that meet the occlusion criteria.
    :param viewpoint: Viewpoint coordinates and orientation (X, Y, Z, yaw, pitch, roll)
    :param target_cell_center: Center of the target cell 'c'
    :param cell_size: Size of each cell
    :param target_distance: Distance from the viewpoint to the target cell 'c'
    :return: Number of cells occluding the target cell 'c'
    """
    count = 0
    ray_origin = np.array(viewpoint[:3])
    ray_direction = target_cell_center - ray_origin  # Assumes already normalized if needed
    ray_direction /= np.linalg.norm(ray_direction)   # Normalize the ray direction

    # Define the bounds of the surrounding cells
    offsets = [-cell_size, 0, cell_size]
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                if dx == dy == dz == 0:
                    continue  # Skip the center cell itself
                
                neighbor_center = target_cell_center + np.array([dx, dy, dz])
                box_min = neighbor_center - np.array([cell_size/2] * 3)
                box_max = neighbor_center + np.array([cell_size/2] * 3)
                distance_to_neighbor = np.linalg.norm(neighbor_center - ray_origin)
                
                if ray_box_intersection(ray_origin, ray_direction, box_min, box_max) and distance_to_neighbor < target_distance:
                    count += 1

    return count

# Example usage:
viewpoint = (1, 1, 1, 0, 0, 0)  # Example viewpoint with XYZ and yaw, pitch, roll
target_cell_center = np.array([5, 5, 5])  # Center of the target cell
cell_size = 1.0  # Size of each cell
target_distance = np.linalg.norm(target_cell_center - np.array(viewpoint[:3]))

occluding_cells = occlusion_count(viewpoint, target_cell_center, cell_size, target_distance)
print(f"Number of occluding cells: {occluding_cells}")
