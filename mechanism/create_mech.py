import numpy as np
import cv2
import os
from itertools import combinations
from speed_color import speed_to_color
import json
from tqdm import tqdm

## Parameters:
MIN_LENGTH = 0.1
MAX_LENGTH = 0.4
TRI_MARGIN = 0.01 # to avoid collinear points in triangle motion
COLINEAR = 0.001  # threshold for collinearity check, triangle area > COLINEAR keeps sample
CURVE_SMOOTH = 0.05  # threshold for curve smoothness check
CURVE_SIZE = 0.1  # threshold for curve size check
CR_CC = 0.6 # crank-rocker to double rocker ratio, used for filtering
# random seed for reproducibility
np.random.seed(2025)

# load isomorphism, graph dictionary
with open('mechanism/graph_vis/1111/graph_dict.json', 'r') as f:
    graph_dict = json.load(f)

# Helper functions
## triangle length generator
def triangle_numbers(min_c, max_c, margin=TRI_MARGIN):
    """
    Generate random lengths for a triangle. 
    Satisfies the triangle inequality: a + b > max_c, abs(a - b) < min_c
    Returns:
    [a, b]: list of triangle side lengths
    """
    # if max_c >= 2 * MAX_LENGTH:
    #     raise ValueError("max_c must be < 2 * MAX_LENGTH to ensure valid triangle lengths")
    while True:
        a, b = np.random.uniform(MIN_LENGTH, max(MAX_LENGTH, 0.6 * max_c), size=2)
        if a + b > max_c + margin and abs(a - b) < min_c:
            return np.array([a, b])

## Input and base link generator        
def input_base_numbers(margin=TRI_MARGIN):
    """
    Generate the base input link lengths, ensuring they differ by at least a margin.
    """
    while True:
        L1, L2 = np.random.uniform(MIN_LENGTH, MAX_LENGTH, size=2)
        L1, L2 = np.sort([L1, L2])  # Ensure L1 <= L2
        if L2 - L1 > margin:
            return np.array([L1, L2])
        
## Triangle solver
def triangle_solver(A, B, r1, r2, ref, away=False):
    """
    This function is used to: 1. build a triangle given two vertices and two link lengths; 2. solve the path in simulation.
    A, B: points in 2D space, numpy arrays of shape (2,)
    r1, r2: distances from A to C and B to C respectively
    ref: reference point C to choose the closest solution
    away: if True, return the point that is away from the reference point
    Returns:
    C: point in 2D space, numpy array of shape (2,)
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    d = np.linalg.norm(B - A)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        raise ValueError("Invalid triangle: no solution exists")
    
    # Unit vector from A to B
    u = (B - A) / d
    
    # Perpendicular unit vector
    v = np.array([-u[1], u[0]])
    
    # Calculate the distance from A to the foot of perpendicular from C to AB
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    
    # Calculate the height from C to line AB
    h_squared = r1**2 - a**2
    
    # Handle numerical precision issues
    if h_squared < 0:
        h_squared = 0
    
    h = np.sqrt(h_squared)
    
    # Two possible positions for C
    foot = A + a * u
    C1 = foot + h * v
    C2 = foot - h * v
    
    # Choose the closest point to the reference
    if away:
        if np.linalg.norm(C1 - ref) > np.linalg.norm(C2 - ref):
            return C1
        else:
            return C2
    else:
        if np.linalg.norm(C1 - ref) < np.linalg.norm(C2 - ref):
            return C1
        else:
            return C2

## colinear check
def colinear(A, B, C, margin=COLINEAR):
    """
    Check if three points A, B, C are collinear within a margin.
    A, B, C: points in 2D space, numpy arrays of shape (2,)
    margin: tolerance for collinearity
    Returns:
    bool: True if collinear, False otherwise
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)
    
    # Area of triangle formed by A, B, C should be zero for collinearity
    area = 0.5 * np.abs(A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    
    return area < margin


def find_mesh(links):
    import networkx as nx
    link_graph = nx.DiGraph()
    link_graph.add_edges_from(links)
    mesh = []
    for node in link_graph.nodes():
        parents = list(link_graph.predecessors(node))
        if len(parents) == 2:
            p1, p2 = sorted(parents)
            if link_graph.has_edge(p1, p2):
                mesh.append([p1, p2, node])
    return mesh

## linkage generator
class mechanism:
    def __init__(self, num_triangles=5, mech_id=5, pixel=224, filter=True):
        """
        Initialize the mechanism with a specified number of triangles.
        num_triangles: Number of triangles in the mechanism
        num_points: Number of points in the mechanism
        cr: crank-rocker mechanism flag
        """
        graph_list = graph_dict[f"tri_{num_triangles}"]
        iso_graph = next((g for g in graph_list if g["id"] == f"tri_{num_triangles}_{mech_id:04d}"), None)
        self.num_triangles = num_triangles
        self.num_points = num_triangles + 3  # 3 base points + num_triangles points
        self.sim_steps = 100
        self.pixel = pixel
        self.path = np.zeros((self.sim_steps, self.num_points, 2), dtype=np.float32)
        self.points = np.zeros((self.num_points, 2), dtype=np.float32)
        self.link_lengths = np.zeros(num_triangles * 2 + 2, dtype=np.float32)
        self.cid = self.num_points - 1 # coupler point index
        self.cr = np.random.random() < CR_CC  # crank-rocker mechanism flag, based on a random chance
        self.links = np.array(iso_graph["edges"], dtype=np.int32)  # Load edges from the isomorphic graph
        
        self.mesh = find_mesh(self.links)  # Find the mesh triangles from the links

        if filter:
            while True:
                self.generate_mechanism()
                self.simulate(tri_num=self.num_triangles)
                self.normalize()
                if not self.collinear_check(threshold=COLINEAR) and self.curve_smooth_check(threshold=CURVE_SMOOTH) and self.curve_range_check(threshold=CURVE_SIZE):
                # if not self.collinear_check(threshold=COLINEAR):
                    break
                else:
                    # print("Collinear points detected, regenerating mechanism...")
                    pass
        else:
            self.generate_mechanism()
            self.simulate(tri_num=self.num_triangles)
            self.normalize()


    def generate_mechanism(self):
        """
        Generate a general mechanism with two triangles.
        """
        origin = np.array([0.5, 0.5], dtype=np.float32)
        # input and base
        S, L = input_base_numbers()
        if self.cr:
            self.link_lengths[:2] = [S, L]  # input link, base link
        else:
            self.link_lengths[:2] = [L, S]
        self.triangle_dir = np.random.choice([True, False], size=self.num_triangles)  # Randomly choose triangle direction, true for away from the origin, false for towards the origin
        # for tri>2 just generate outer points
        # self.triangle_dir = np.ones(self.num_triangles, dtype=bool)  # Set all triangles to away from the origin
        self.points[:3] = [
            origin + np.array([-self.link_lengths[1] / 2, 0.0], dtype=np.float32),
            origin + np.array([-self.link_lengths[1] / 2, self.link_lengths[0]], dtype=np.float32),
            origin + np.array([self.link_lengths[1] / 2, 0.0], dtype=np.float32)
        ]

        # triangles
        for tri_num in range(1, self.num_triangles + 1):
            point_id = tri_num + 2  # Start from point 3 for the first triangle
            link_ids = [tri_num * 2, tri_num * 2 + 1]  # Link lengths for the current triangle
            parent_ids = self.links[link_ids, 0]  # Parent points for the current triangle
            self.simulate(tri_num=tri_num-1)  # Simulate one previous layer to find the moving range of the parent points
            parent_distance = np.linalg.norm(self.path[:, parent_ids[0]] - self.path[:, parent_ids[1]], axis=1)  # shape (sim_steps,)
            # print(f"Triangle {tri_num}: parent distance min={np.min(parent_distance)}, max={np.max(parent_distance)}")
            self.link_lengths[link_ids] = triangle_numbers(min_c=np.min(parent_distance), max_c=np.max(parent_distance))
            self.points[point_id] = triangle_solver(self.points[parent_ids[0]], 
                                                    self.points[parent_ids[1]], 
                                                    self.link_lengths[link_ids[0]], 
                                                    self.link_lengths[link_ids[1]],
                                                    origin, away=self.triangle_dir[0])


    def simulate(self, tri_num):
        """
        Simulate the mechanism for a given number of steps.
        Simulate given number of points, to fine the moving range of the parent points for building a new triangle
        """
        self.path[0] = self.points.copy()  # Store initial points
        
        # Store the initial input link vector for proper rotation
        initial_input_vector = self.points[1] - self.points[0]
        
        # [0, 1, 2], 0 and 2 is fixed, 1 is the input link
        for step in range(1, self.sim_steps):
            # Calculate angle for 360 degree rotation over all steps
            angle = step * (2 * np.pi / self.sim_steps)

            # Create rotation matrix
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            
            # Rotate the input link from its initial position
            rotated_vector = rotation_matrix @ initial_input_vector
            self.path[step, 1] = self.path[0, 0] + rotated_vector  # Rotate around base point (point 0)
            
            # Keep base points fixed
            self.path[step, 0] = self.path[0, 0]  # Base point remains fixed
            self.path[step, 2] = self.path[0, 2]  # Output base point remains fixed
            
            # Use triangle solver to find other points
            for tri in range(1, tri_num + 1):
                point_id = tri + 2  # Start from point 3 for the first triangle
                link_ids = [tri * 2, tri * 2 + 1]  # Link lengths for the current triangle
                parent_ids = self.links[link_ids, 0] # Parent points for the current triangle
                self.path[step, point_id] = triangle_solver(
                    self.path[step, parent_ids[0]], 
                    self.path[step, parent_ids[1]], 
                    self.link_lengths[link_ids[0]], 
                    self.link_lengths[link_ids[1]], 
                    self.path[step - 1, point_id]
                )

            
    def normalize(self):
        """
        Normalize the point positions to [0, 1].
        """
        curve_points = self.path[:, self.cid]  # Coupler points, shape (sim_steps, 2)
        # mechanism_points = self.path[0]  # Points at the current time step, shape (num_points, 2)
        mechanism_points = self.path.reshape(-1, 2)  # flatten the path to get all points
        all_points = np.vstack([curve_points, mechanism_points])  # shape (sim_steps + sim_steps * num_points, 2)
        bbox_distance = np.max(np.abs(all_points - 0.5))
        # scale the path positions
        self.path = (self.path - 0.5) * 0.5 / bbox_distance + 0.5


    def collinear_check(self, threshold):
        """
        Filter points that are, assume false and return if any points are true
        - too close to each other.
        - collinear with their neighbors.
        """
        # check if two points are too close
        # for i, j in combinations(range(self.num_points), 2):
        #     if np.linalg.norm(self.path[0, i] - self.path[0, j]) < threshold:
        #         return True

        for i, j, k in combinations(range(self.num_points), 3):
            if colinear(self.path[0, i], self.path[0, j], self.path[0, k], margin=threshold):
                return True

        return False
    
    def curve_smooth_check(self, threshold):
        """
        Check if the coupler curve is smooth enough.
        A curve is considered unsmooth if the speed change between consecutive points is greater than a threshold.
        """
        dist = []
        for i in range(0, self.sim_steps):
            dist.append(np.linalg.norm(self.path[i, self.cid] - self.path[i - 1, self.cid]))

        for i in range(0, self.sim_steps):
            gap = abs(dist[i] - dist[i - 1])
            if gap > threshold:
                return False

        return True
    

    def curve_range_check(self, threshold):
        """
        Check the curve range
        If the curve range is too small, return False.
        This is to ensure the coupler point moves within a reasonable range.
        """
        bbox_x = np.max(self.path[:, self.cid, 0]) - np.min(self.path[:, self.cid, 0])
        bbox_y = np.max(self.path[:, self.cid, 1]) - np.min(self.path[:, self.cid, 1])
        bbox_sum = bbox_x + bbox_y
        if bbox_sum < threshold:
            return False
        return True


    def image(self, line_width=3, time_step=0, draw_coupler=True, draw_mechanism=True):
        # Create a blank image
        img = np.zeros((self.pixel, self.pixel, 3), dtype=np.uint8)

        # Scale points to fit the 224x224 canvas with some padding
        # Assuming points are in range [0, 1], scale to [padding, self.pixel-padding]
        padding = 0.1 * self.pixel
        scale = self.pixel - 2 * padding

        if draw_mechanism:
            # 1. Draw the mesh triangles
            for triangle in self.mesh:
                pts = np.array([self.path[time_step][i] for i in triangle], dtype=np.float32)  # shape (3, 2)
                # Scale and flip y-coordinate, then convert to int
                pts_scaled = pts * scale + padding
                pts_scaled[:, 1] = (1 - pts[:, 1]) * scale + padding  # Flip y-coordinate properly
                pts_int = pts_scaled.astype(np.int32)
                cv2.fillPoly(img, [pts_int], (150, 150, 150))

            # 2. Draw the four-bar linkage lines with specified color and thickness
            for i, (start, end) in enumerate(self.links[2:]):  # Skip the first two links (input and base)
                cv2.line(img, (int(self.path[time_step][start][0] * scale + padding), int((1 - self.path[time_step][start][1]) * scale + padding)),
                            (int(self.path[time_step][end][0] * scale + padding), int((1 - self.path[time_step][end][1]) * scale + padding)),
                            (255, 0, 0), line_width)
            start, end = self.links[0]  # Input link
            cv2.line(img, (int(self.path[time_step][start][0] * scale + padding), int((1 - self.path[time_step][start][1]) * scale + padding)),
                        (int(self.path[time_step][end][0] * scale + padding), int((1 - self.path[time_step][end][1]) * scale + padding)),
                        (0, 255, 0), line_width)
            start, end = self.links[1]  # Base link
            cv2.line(img, (int(self.path[time_step][start][0] * scale + padding), int((1 - self.path[time_step][start][1]) * scale + padding)),
                        (int(self.path[time_step][end][0] * scale + padding), int((1 - self.path[time_step][end][1]) * scale + padding)),
                        (0, 0, 225), line_width)
            
            # 3. Draw the joints
            for i in range(self.num_points):
                cv2.circle(img, (int(self.path[time_step][i][0] * scale + padding), int((1 - self.path[time_step][i][1]) * scale + padding)),
                        radius=5, color=(0, 255, 255), thickness=-1)
                
        if draw_coupler:
            # Draw the coupler curve, map speed to color
            # max_speed = 0.05  # Normalized speed threshold
            max_speed = np.max(np.linalg.norm(self.path[1:, self.cid] - self.path[:-1, self.cid], axis=1))  # Maximum speed in the simulation
            min_speed = np.min(np.linalg.norm(self.path[1:, self.cid] - self.path[:-1, self.cid], axis=1))  # Minimum speed in the simulation
            for i in range(-1, self.sim_steps - 1):
                speed = np.linalg.norm(self.path[i + 1, self.cid] - self.path[i, self.cid])
                color_value = speed_to_color(speed, max_speed, min_speed)
                cv2.line(img, (int(self.path[i, self.cid][0] * scale + padding), int((1 - self.path[i, self.cid][1]) * scale + padding)),
                        (int(self.path[i + 1, self.cid][0] * scale + padding), int((1 - self.path[i + 1, self.cid][1]) * scale + padding)),
                        color_value, line_width)

        return img
    
    def video(self, line_width=3, filename='mechanism_simulation.avi', draw_coupler=True, draw_mechanism=True):
        """
        Generate a video of the mechanism simulation.
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (self.pixel, self.pixel))

        for step in range(self.sim_steps):
            img = self.image(line_width=line_width, time_step=step, draw_coupler=draw_coupler, draw_mechanism=draw_mechanism)
            out.write(img)

        out.release()


def data_collection(dataset_size=1000, 
                    num_triangles=2,
                    mechanism_id=0,
                    output_dir='tri_2/',
                    video=False
                    ):
    """
    Collect data for the mechanism.
    For each mechanism instance, the dataset includes:
    - Mechanism parameters: normalized curve points, mechanism points
    - Images: mechanism, curve
    - Video: mechanism simulation
    """
    # output_dir = f'tri_{num_triangles}/'
    os.makedirs("./../dataset/" + output_dir + "coords/", exist_ok=True)
    os.makedirs("./../dataset/" + output_dir + "images/mechanism/", exist_ok=True)
    os.makedirs("./../dataset/" + output_dir + "images/curve/", exist_ok=True)
    os.makedirs("./../dataset/" + output_dir + "videos/", exist_ok=True)
    notes = []  # whether crank-rocker, whether triangle-1 outer, whether triangle-2 outer
    for i in tqdm(range(dataset_size), desc=f"Data {output_dir}"):
        mech = mechanism(num_triangles=num_triangles, mech_id=mechanism_id, pixel=224, filter=True)
        # Save mechanism parameters
        np.save(f"./../dataset/{output_dir}coords/{i:06}.npy", mech.path)
        
        # Save mechanism image
        img = mech.image(line_width=2, draw_coupler=False, draw_mechanism=True)
        cv2.imwrite(f"./../dataset/{output_dir}images/mechanism/{i:06}.png", img)

        # Save curve image
        curve_img = mech.image(line_width=3, draw_coupler=True, draw_mechanism=False)
        cv2.imwrite(f"./../dataset/{output_dir}images/curve/{i:06}.png", curve_img)

        # Generate and save video
        if video:
            mech.video(line_width=3, filename=f'./../dataset/{output_dir}videos/{i:06}.mp4', draw_coupler=True, draw_mechanism=True)
            # Uncomment the line below to save the video
            # mech.video(line_width=3, filename=f'dataset/{output_dir}videos/{i:06}.avi')
            # mech.video(line_width=3, filename=f'dataset/{output_dir}videos/{i:06}.gif')
        # mech.video(line_width=3, filename=f'dataset/{output_dir}videos/{i:06}.mp4')

        # Save notes
        notes.append([i, int(mech.cr)] + [int(d) for d in mech.triangle_dir])

    # sum_cr = sum(note[1] for note in notes)
    # print(f"Total crank-rocker mechanisms: {sum_cr} out of {dataset_size}")
    # print(f"Total double rocker mechanisms: {dataset_size - sum_cr} out of {dataset_size}")

    np.savetxt(f"./../dataset/{output_dir}notes.txt", notes, fmt="%d")


def complex_dataset(graph_dict):
    for num_triangles in range(2, 4):
        graph_list = graph_dict[f"tri_{num_triangles}"]
        for iso_graph in graph_list:
            data_collection(dataset_size=10000, 
                            num_triangles=num_triangles,
                            mechanism_id=int(iso_graph["id"].split('_')[-1]),
                            output_dir=f'complex_t4/{iso_graph["id"]}/'
                            )


if __name__ == "__main__":
    # mech = mechanism(pixel=448)
    # img = mech.image(line_width=5)
    # mech.video(line_width=5, filename='mechanism_simulation.mp4')
    # cv2.imshow("Mechanism", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # data collection
    # data_collection(dataset_size=10000, output_dir='test/tri_2/', video=False)

    # complex dataset
    complex_dataset(graph_dict)
