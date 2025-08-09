import cv2
import os
import numpy as np
import tqdm
import glob

## color mapping
def speed_to_color(speed, max_speed, min_speed, colormap=cv2.COLORMAP_TURBO):
    """
    Map speed to a color using a colormap.
    speed: float, speed value
    max_speed: float, maximum speed for normalization
    min_speed: float, minimum speed for normalization
    Returns:
    color: tuple, RGB color value, Jul 25 version
    """
    # normalized_speed = min(speed / max_speed, 1.0)
    # normalized_speed = (speed - min_speed) / (max_speed - min_speed)
    
    normalized_speed = speed / (max_speed + min_speed)  # map [0, min + max] to [0, 1]

    speed_array = np.array([[np.uint8(normalized_speed * 255)]], dtype=np.uint8)
    color_bgr = cv2.applyColorMap(speed_array, colormap)[0][0]
    return tuple(int(c) for c in color_bgr)  # Convert BGR to RGB tuple


def coordinate_to_color(line_width, data_path, output_image_dir):
    """
    Convert a coordinate to a color value.
    """
    pixel = 224
    padding = 0.1 * pixel
    scale = pixel - 2 * padding
    # Load the coordinate data
    coord_path_list = sorted(glob.glob(os.path.join('dataset/', data_path, 'coords/*.npy')))
    print(f"Found {len(coord_path_list)} coordinate files in {data_path}coords/")
    os.makedirs(f"dataset/{data_path}images/{output_image_dir}", exist_ok=True)

    for i, coord_path in tqdm.tqdm(enumerate(coord_path_list), total=len(coord_path_list)):
        img = np.zeros((pixel, pixel, 3), dtype=np.uint8)   
        # Load the coordinates
        coords = np.load(coord_path)  # shape (time_steps, num_points, 2), the last point is the end point
        draw_path = coords[:, -1, :]  # shape (time_steps, 2)
        max_speed = np.max(np.linalg.norm(draw_path[1:] - draw_path[:-1], axis=1))  # max speed between points
        min_speed = np.min(np.linalg.norm(draw_path[1:] - draw_path[:-1], axis=1))  # min speed between points
        for t in range(-1, draw_path.shape[0] - 1): # connect the last point and the first point
            speed = np.linalg.norm(draw_path[t + 1] - draw_path[t])
            color_value  = speed_to_color(speed, max_speed, min_speed)
            cv2.line(img, (int(draw_path[t][0] * scale + padding), int((1 - draw_path[t][1]) * scale + padding)),
                        (int(draw_path[t + 1][0] * scale + padding), int((1 - draw_path[t + 1][1]) * scale + padding)),
                        color_value, line_width)
            
        # Save the image
        cv2.imwrite(f"dataset/{data_path}images/{output_image_dir}{i:06}.png", img)

if __name__ == "__main__":
    # Example usage
    coordinate_to_color(3, 'tri_2_color/', 'curve_1/')