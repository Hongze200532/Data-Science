#The below import are all that should be used in this assignment. Additional libraries are not allowed.
import numpy as np
import math
import scipy.sparse.csgraph
import matplotlib.pyplot as plt
import random
import argparse
import collections
import sys
import csv
'''
==============================
The code below here is for your occupancy grid solution
==============================
'''

def cell_index(x , y, col_number):
    '''
    Converts (x,y) coordinates of OCcupancy Grid index to CelL Index
    '''
    return y * col_number + x

def read_map_from_file(filename:str) -> tuple:
    '''
    Reads map csv file and returns map data
    '''
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        List = [tuple(map(int, row)) for row in reader if row]  # Read each row as tuple of integers

    if List:
        grid_size = tuple(List[0])   # grid size 
        start_pos = tuple(List[1])   #  start pos
        goal_pos = tuple(List[2])    #  goal pos
        obstacles = [tuple(map(int, row)) for row in List[3:]]  # obstacles
        return (grid_size, start_pos, goal_pos, obstacles)
    else: 
        print("Read Error")



def make_occupancy_grid_from_map(map_data, cell_size=5):
    '''
    Takes map and cell size (actual size of cell in grid) and returns a 2D numpy array
    '1' if occupied and '0' empty
    '''

    grid_dim, obs = map_data[0], map_data[-1]

    grid_x = grid_dim[0]
    grid_y = grid_dim[1]

    occupancy_grid = np.zeros((grid_x//cell_size, grid_y//cell_size), dtype= int)

    for x in range(cell_size, grid_x - cell_size, cell_size):
        for y in range(cell_size, grid_y - cell_size, cell_size):
            for Ox, Oy, Orad in obs: 
                distance = np.sqrt((x - Ox) **2 + (y - Oy)**2)

                Cell_x, Cell_y  = x // cell_size, y // cell_size

                if distance < Orad:
                    occupancy_grid[Cell_x , Cell_y] = 1
                    occupancy_grid[Cell_x, Cell_y-1] = 1
                    occupancy_grid[Cell_x-1, Cell_y] = 1
                    occupancy_grid[Cell_x-1, Cell_y-1] = 1
                    break
    return occupancy_grid

def make_adjacency_matrix_from_occupancy_grid(occupancy_grid):
    '''
    Converts occupancy grid into an adjacency matrix.
    Cells are connected to their neighbors unless it is occupied.
    Cost of moving a neighbor is always '1' and only horizontal and vertical connections.
    '''
    dimension = np.shape(occupancy_grid) # Dimensions from occupancy grid
    x_dim, y_dim = dimension

    adjacency_matrix = np.zeros((x_dim * y_dim, x_dim * y_dim), dtype=int)

    movements = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    # Horizontal/ Vertical movements all by 1
    
    def index(x, y):
        return x* y_dim + y
    # Convert 2D index to 1D
    for i, j in np.ndindex(x_dim, y_dim):  # Loop through all valid grid indices
        if occupancy_grid[i, j] == 0:
            for x, y in movements:
                ii, jj = i + x, j + y  # neighbor indices
                if 0 <= ii < x_dim and 0 <= jj < y_dim:  # Check in bounds
                    adjacency_matrix[index(i, j),index(ii, jj)] = 1  # Set adjacency as required
    return adjacency_matrix


def get_path_from_predecessors(predecessors, map_data, cell_size=5) :
    '''
    Takes a predecessors matrix, map_data and cell_size as input and returns the path from start to goal position.
    Take the mid-point of each cell as the coordinate for the path.
    Choose the cell in the bottom left corner if the start/ end position lies in the intersection of cells
    '''
    # Create Index List and Coordinates List
    path = []
    path_coordinates = []
    # Extract map data
    map_grid_x, map_grid_y = map_data[0]  
    start_pos = map_data[1]  
    goal_pos = map_data[2] 

    # Map dimensions --> Occupany grid dimensions
 
    start_cell = (start_pos[0]// cell_size, start_pos[1]// cell_size)
    goal_cell = (goal_pos[0]// cell_size, goal_pos[1] //cell_size)
 
    # Occupancy grid --> Occupancy Cell index
    # Dealing with Start_pos (Map_dim), Goal_pos (Map_dim) in the intersection of cells
    # Choose the lower left cell as starting pos
    def Cell_index(x, y, cell_size):
        return y * cell_size + x -1
   
    start_index = Cell_index(start_cell[0], start_cell[1], map_grid_x // cell_size)
    goal_index = Cell_index(goal_cell[0], goal_cell[1], map_grid_x // cell_size)
 
    print(f"Start: {start_index}, Goal: {goal_index}")
 
    if predecessors[goal_index, start_index] < 0:
        print('No Valid Path')
        return path_coordinates
   
    while goal_index != start_index:
        temp_index = predecessors[goal_index, start_index]
        # Temp index is the next node to traverse from goal index
        path.append((start_index, temp_index))
        # Move to the next node
        start_index = temp_index
 
    # Occupancy node -> Occupancy index
    for start_index, goal_index in path:
        pos_1 = (start_index % (map_grid_y //cell_size), start_index// (map_grid_x //cell_size))
    # Ocupancy index -> Map_dim (centre of cell)
        coordinate_1 = (pos_1[1] * cell_size + cell_size / 2, pos_1[0] * cell_size + cell_size / 2)
        path_coordinates.append(coordinate_1)


    return path_coordinates

def plot_map(ax, map_data, cell_size = 5):
    '''
    Plots map according to the description 
    '''
    if map_data:
        start_pos = map_data[1]
        goal_pos = map_data[2]
        obstacles = map_data[3]

        ax.plot(goal_pos[0], goal_pos[1], 'b*', label='Goal')
        ax.plot(start_pos[0], start_pos[1], 'y*', label='Start')

        ax.set_xlim(0, map_data[0][0])
        ax.set_ylim(0, map_data[0][1])
        ax.set_title("Shortest Path")

        # Draw vertical grid lines
        for x in range(0, map_data[0][0] + 1, cell_size):
            ax.axvline(x, color='gray', linewidth=0.5)

        # Draw horizontal grid lines
        for y in range(0, map_data[0][1] + 1, cell_size):
            ax.axhline(y, color='gray', linewidth=0.5)

        # Draw obstacles
        for obstacle in obstacles:
            # obstacle = (x, y, radius)
            c_patch = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red')
            ax.add_patch(c_patch)

        ax.legend()
    else:
        print("No map data provided")


def plot_path(ax, path):
    '''
    This function plots the path found by occupancy grid solution.
    '''
    if path:
        x_coords, y_coords = zip(*path)  # Unzip list of tuples to separate lists
        ax.plot(x_coords, y_coords, 'b-', linewidth=2)  # Connect point with a blue line
 
       
 
def test_make_occupancy_grid():
   
    map0 = ((10, 10), (1, 1), (9, 9), [])
    assert np.array_equal(make_occupancy_grid_from_map(map0, cell_size=1), np.zeros((10, 10))), "Test 1 - checking map 0 with cell size 10"
   
    map1 = ((10, 10), (1, 1), (9, 9), [(5, 5, 2)])
    assert np.array_equal(make_occupancy_grid_from_map(map1, cell_size=10), np.array([[1]])), "Test 1 - checking map 1 with cell size 10"
    assert np.array_equal(make_occupancy_grid_from_map(map1, cell_size=5), np.array([[1, 1], [1, 1]])), "Test 2 - checking map 1 with cell size 5"
 
    map1_cell_size_2_answer = np.array([[0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]])
    assert np.array_equal(make_occupancy_grid_from_map(map1, cell_size=2), map1_cell_size_2_answer), "Test 3 - checking map 1 with cell size 2"
 
    map2 = ((100, 100), (1, 1), (9, 9), [(10, 10, 5), (90, 90, 5)])
    map2_answer = np.array([[1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]])
 
    occupancy_grid1 = np.array([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
 
    adjacency_matrix1 = np.array([[0., 1., 0., 1., 0., 0., 0., 0., 0.],
        [1., 0., 1., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 1., 0., 1., 0., 0.],
        [0., 1., 0., 1., 0., 1., 0., 1., 0.],
        [0., 0., 1., 0., 1., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1., 0., 1., 0., 1.],
        [0., 0., 0., 0., 0., 1., 0., 1., 0.]])
 
    assert np.array_equal(make_adjacency_matrix_from_occupancy_grid(occupancy_grid1), adjacency_matrix1)
 
    occupancy_grid2 = np.array([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
 
    adjacency_matrix2 = np.zeros((occupancy_grid2.size, occupancy_grid2.size))
 
    assert np.array_equal(make_adjacency_matrix_from_occupancy_grid(occupancy_grid2), adjacency_matrix2)
 
    occupancy_grid3 = np.array([[0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]])
 
    adjacency_matrix3 = np.array([[0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0.]])
 
    assert np.array_equal(make_adjacency_matrix_from_occupancy_grid(occupancy_grid3), adjacency_matrix3)
 
def test_get_path():
    occupancy_grid1 = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]]
                                )
   
    distance, predecessor = scipy.sparse.csgraph.shortest_path(occupancy_grid1, return_predecessors = True)
   
    occupancy_grid1_start_pos, occupancy_grid1_goal_pos = (2,0), (0,2)
    occupancy_grid1_answer = [(12.5, 2.5), (7.5, 2.5), (2.5, 2.5), (2.5, 7.5), (2.5, 12.5)]
   
    assert occupancy_grid1_answer == get_path_from_predecessors(predecessor, 'map_a_test_get_path.csv', cell_size= 5)
   



def test_occupancy_grid():
    #test_make_occupancy_grid()
    test_get_path()
 
def occupancy_grid(file, cell_size=5):
 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
 
    map_data = read_map_from_file(file)
    plot_map(ax, map_data)
    grid = make_occupancy_grid_from_map(map_data, cell_size)
    adjacency_matrix = make_adjacency_matrix_from_occupancy_grid(grid)
    #You'll need to edit the line below to use Scipy's shortest graph function to find the path for us
    distance, predecessors = scipy.sparse.csgraph.shortest_path(adjacency_matrix, return_predecessors = True)
    path = get_path_from_predecessors(predecessors, map_data, cell_size)
    plot_path(ax, path)
 
    plt.show()
 
 

'''
==============================
The code below here is for your RRT solution
==============================
'''
# Class is like blueprint to build object --> Anything built from a Class --> Instance

# Custom class for Map
class Map:
    def __init__(self, dimensions: tuple, start: tuple, goal: tuple, obstacles: list):
        self.dimensions_x = dimensions[0] # x dimensions
        self.dimensions_y = dimensions[1] # y dimensions
        self.start_x = start[0] # start_position x
        self.start_y = start[1] # start_position y
        self.goal_x = goal[0]   # goal_position x
        self.goal_y = goal[1]   # goal_position y
        self.obstacles = obstacles # List of obstacles coord

    #__(str)__ --> Mrthod to pring class instance 
    def __str__(self):
        return (f"Map Dimensions: ({self.dimensions_x}, {self.dimensions_y})\n"
            f"Start position: ({self.start_x}, {self.start_y})\n"
            f"Goal position: ({self.goal_x}, {self.goal_y})\n"
            f"Obstacles: {self.obstacles}")



# Custom class to represent tree structure
class Node:
    def __init__(self, x, y, parent = None):
        self.x = x # node x
        self.y = y # node y
        self.parent = None #  Parent node connects the current node to layer above 
        self.child = [] # Child nodes connects current layer to to one layer below --> Stored in List
 
    def __str__(self):
        return (f'X Coordinates: {self.x}\n'
                f'Y Coordinates: {self.y}\n'
                f'Parent Node: {self.parent}\n'
                f'Child: {self.child}'
        )
 
    def Add_child(self, child): # Method to add child nodes
        self.child.append(child)
 
# Custom Class to reform RRT search
class RRT:
    def __init__(self, map: Map, step_size: int, max_iter = 500):
           
        self.start = Node(map.start_x, map.start_y) # start_pos 
        self.map_dim = (map.dimensions_x, map.dimensions_y) # Map dimensions
        self.step_size = step_size # Step size initialize
        self.max_iter = max_iter # Max random points 
        self.tree = [self.start] # Init list with with starting node
        self.random_nodes = []
        self.goal = Node(map.goal_x, map.goal_y) # goal_pos 
        self.obstacles = map.obstacles # Obstacles
        self.shortest_path = None
 
    def __str__(self):
        return (f"Start: ({self.start.x}, {self.start.y})\n"
            f"Goal: ({self.goal.x}, {self.goal.y})\n"
            f"Map Dimensions: {self.map_dim}\n"
            f"Step Size: {self.step_size}\n"
            f"Max Iterations: {self.max_iter}\n"
            f"Obstacles: {self.obstacles}\n"
            f"Tree Size: {len(self.tree)} nodes")
 
    def Distance(self, Point_1: Node, Point_2: Node):
        '''
        Returns the distance between two points
        '''
        return math.sqrt((Point_2.x- Point_1.x) **2 + (Point_2.y- Point_1.y) **2)
 
    def Closest_node(self, random: Node):
        '''
        Takes in a RRT object (tree) and random node
        Calculate the distance between nodes in the "tree" and random point
        Returns the closest one out of the tree
        '''
        return min(self.tree, key = lambda node: self.Distance(node, random))
 
    def Grow(self, start_node: Node, end_node: Node):
        """
        Takes in closest node and randomly generated node
        Determines the direction to grow the tree in.
        Grows a Node in that direction within the step_size limit
        """
        dx = end_node.x - start_node.x # Change in x
        dy = end_node.y - start_node.y # Change in y
 
        # Calculate distance between node in tree and randomly generated node
        distance = self.Distance(start_node, end_node)
       
        # If the distance is less than the step size (randomly generated node is within reach)
        # Return the randomly generated node
   
        if distance <= self.step_size:
            return end_node
        else:
            Ratio = self.step_size/ distance            # If it is not within reach, scale down by step size
            target_x = start_node.x + Ratio * dx        # Move in direction of randomly generated node by this ratio
            target_y = start_node.y + Ratio * dy
            return Node(target_x, target_y, start_node)
    
    def Valid_node(self, node):
        """
        Check if randomly generated node is within any obstacles 
        """
        for Ox, Oy, Orad in self.obstacles:
            if self.Distance(node, Node(Ox, Oy)) < Orad:
                return False
        return True

    def No_collision(self, start_node: Node, end_node: Node) -> bool:
        """
        Check for collision along the path from start to end node by incrementing in small steps
        """
        step_size = 0.01
        movements = int(self.Distance(start_node, end_node) // step_size)
        if movements == 0:  # Prevent ZeroDivisionError
            return True  # No movement needed, so no collision
 
        for steps in range(movements + 1):
            temp = steps / max(1, movements)  # Avoid division by zero
            temp_x = start_node.x + temp * (end_node.x - start_node.x)
            temp_y = start_node.y + temp * (end_node.y - start_node.y)
   
            for items in self.obstacles:
                ox, oy, orad = items
                if self.Distance(Node(temp_x, temp_y), Node(ox, oy)) < orad:
                    return False  # Path collides with an obstacle
 
        return True  # No collision
 

    
 
    def Shortest(self, goal_node):
        """
        Traces path from start to goal node 
        """
        path = []
        current_node = goal_node
 
        while current_node is not None:
            path.append((int(current_node.x), int(current_node.y)))  # Convert to integers for readability
            current_node = current_node.parent  # Move to parent
       
        if path:
            path.reverse()
            self.shortest_path = path
            print(f'shortest path: {self.shortest_path}')
        else:
            print("No shortest path")
 

    def Run(self):
        for iterations in range(self.max_iter+ 1):
            random_node = Node(random.randint(0, self.map_dim[0] - 1), random.randint(0, self.map_dim[1] - 1))
            # random node--> FOr simpliciyt x, y are nitegers
            self.random_nodes.append(random_node)

            nearest_node = self.Closest_node(random_node)
            # Determine which of the nodes in the tree is closes to randomly node

            new_node = self.Grow(nearest_node, random_node)
            # Grows tree by creating a new node in direction

            if self.Valid_node(new_node) and self.No_collision(nearest_node, new_node):
                self.tree.append(new_node)  # new node to trere
                new_node.parent = nearest_node # Set parent of randomly generated node to the nearest node
 
                if self.Distance(new_node, self.goal) <= self.step_size:
                    self.goal.parent = new_node  # Set goal's parent
                    self.tree.append(self.goal)  # Add goal to tree
                    self.Shortest(self.goal)  # Trace e shortest path
                    return self.shortest_path 
                
        self.Plot() 
    
        return None
   
    def Plot(self):
        '''
        Plots the RRT solution
        '''
        fig, ax = plt.subplots(figsize = (6, 6))
 
        ax.set_xlim(0, self.map_dim[0])
        ax.set_ylim(0, self.map_dim[1])
 
        ax.plot(self.start.x, self.start.y, 'g*', markersize = 5, label = 'Start')
        ax.plot(self.goal.x, self.goal.y, 'r*', markersize =5, label = 'Goal')
 
        for ox, oy, orad in self.obstacles:
            obstacles = plt.Circle((ox, oy), orad, color = 'red', fill = True)
            ax.add_patch(obstacles)
 
        for random_node in self.random_nodes:
            ax.plot(random_node.x, random_node.y, 'bx', markersize=3, alpha=0.4)
 
        for node in self.tree:
            if node.parent:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y], '-', color='purple', alpha = 0.5)
       
        if self.shortest_path:
            path_x, path_y = zip(*self.shortest_path)
            ax.plot(path_x, path_y, 'o', colour = "purple", linewidth = 0.3)
       
        ax.legend()
        ax.set_title("RRT")
        plt.grid(True)
        plt.show()
 

def read_map_from_file(filename:str) -> Map:
    '''
    This functions reads a csv file describing a map and returns the map data
    '''

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        List= [tuple(map(int, row)) for row in reader if row]  # Read each row as tuple of integers
 
    if List: 
        grid_size = tuple(List[0])   #  grid size --> (width, height)
        start_pos = tuple(List[1])   # start position --> (x, y)
        goal_pos = tuple(List[2])    #  goal position --> (x, y)
        obstacles = [tuple(map(int, row)) for row in List[3:]]  #  obstacles
    
        return Map(grid_size, start_pos, goal_pos, obstacles)
    else: 
        print("Error ")
 

def rrt(file_name: str, step_size=10, num_points=100):
    '''
    Initialize starting node as root of tree
    Generate trial point
    Iterate through by step size in the direction of the tree
    Check if this path is collision free or not
    If not: Add path to tree
    repeat
    '''
    Map_data = read_map_from_file(file_name)
    rrt = RRT(Map_data, step_size= step_size)
    if rrt:
        rrt.Run()
        rrt.Plot()
    return rrt.shortest_path
   
 
def test_rrt():
    test_map = Map(dimensions=(100, 100),
                   start=(1, 11),
                   goal= (50, 90),
                   obstacles = [(20, 10, 10)]
    )
    rrt = RRT(test_map, step_size = 10)
    rrt.Run()
 
    assert rrt.shortest_path is not None
    print('path exists')
    # Check that a path exist
 
    assert rrt.shortest_path[0] == (test_map.start_x, test_map.start_y)
    assert rrt.shortest_path[-1] == (test_map.goal_x, test_map.goal_y)
    print('We have start and goal position')
    # Check start and end points are at their correct posision

    for (x, y) in rrt.shortest_path:
        for ox, oy, orad in rrt.obstacles:
            assert math.sqrt((x - ox)**2 + (y - oy)**2) > orad

    print('valid path')

    print('All test complete')


'''
==============================
The code below here is used to read arguments from the terminal, allowing us to run different parts of your code.
You should not need to modify this
==============================
'''
 
def main():
 
    parser = argparse.ArgumentParser(description=" Path planning Assignment for CPA 2024/25")
    parser.add_argument('--rrt', action='store_true')
    parser.add_argument('-test_rrt', action='store_true')
    parser.add_argument('--occupancy', action='store_true')
    parser.add_argument('-test_occupancy', action='store_true')
    parser.add_argument('-file')
    parser.add_argument('-cell_size', type=int)

    args = parser.parse_args()

    if args.occupancy:
        if args.file is None:
            print("Error - Occupancy grid requires a map file to be provided as input with -file <filename>")
            exit()
        else:
            if args.cell_size:
                occupancy_grid(args.file, args.cell_size)
            else:
                occupancy_grid(args.file)

    if args.test_occupancy:
        print("Testing occupancy_grid")
        test_occupancy_grid()

    if args.test_rrt:
        print("Testing RRT")
        test_rrt()

    if args.rrt:
        if args.file is None:
            print("Error - RRT requires a map file to be provided as input with -file <filename>")
            exit()
        else:
            rrt(args.file)
 
if __name__ == "__main__":
    main()