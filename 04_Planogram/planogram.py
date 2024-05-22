# function to read the reference planogram bounding boxes 
def read_reference_file(filepath):
  """
  Reads a five-column text file with class IDs and bounding box coordinates.

  Args:
      filepath (str): Path to the text file.

  Returns:
      tuple: A tuple containing two NumPy arrays:
          - class_ids (ndarray): Array of class IDs (first column).
          - bboxes (ndarray): Array of bounding boxes (xmin, ymin, xmax, ymax) for each row.
  """

  with open(filepath, 'r') as f:
        lines = f.readlines()

  data = []
  for line in lines:
      # Split the line by whitespace (space-separated columns)
      values = line.strip().split()

      # Check if there are exactly five elements
      if len(values) != 5:
          raise ValueError(f"Line '{line}' has unexpected number of columns")

      # Convert elements to float 
      class_id, xcentre, ycentre, width, height = map(float, values)
    
      # Get the xmin(top left), ymin(top left), xmax and ymax of the boxes
      xmin = xcentre - width/2
      ymin = ycentre - height/2
      xmax = xmin + width
      ymax = ymin + height
      # Append data as a list
      data.append([class_id, xmin, ymin, xmax, ymax])

  # Convert data list to NumPy arrays
  data = np.array(data)
  #class_ids = np.array([row[0] for row in data])
  #bboxes = np.array(data[:, 1:])
  return data
  #return class_ids, bboxes
    
# Compute IOU between two boxes
def compute_iou(box1, box2):
    # Extract coordinates of the boxes
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate the coordinates of the intersection rectangle
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    # If the boxes do not intersect, return IoU as 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate the area of each box
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

# Conevert normalized length, width and coordinates into pixel
def convert_normalized_to_pixel(normalized_bbox, image_width, image_height):
    xmin, ymin, xmax, ymax = normalized_bbox
    
    # Convert normalized coordinates to pixel coordinates
    pixel_xmin = int(xmin * image_width)
    pixel_ymin = int(ymin * image_height)
    pixel_xmax = int(xmax * image_width)
    pixel_ymax = int(ymax * image_height)
    
    return pixel_xmin, pixel_ymin, pixel_xmax, pixel_ymax

# Transform bouding boxes perspective
def transform_bounding_box(bbox, homography, image_to_map):
    # Convert bounding box from (xmin, ymin, xmax, ymax) to corner points
    points = np.array([
        [bbox[0], bbox[1]],  # Top-left
        [bbox[2], bbox[1]],  # Top-right
        [bbox[2], bbox[3]],  # Bottom-right
        [bbox[0], bbox[3]]   # Bottom-left
    ], dtype='float32')

    # Add a dimension for homogeneous coordinates
    points = np.array([points])
    
    # Transform the points using the homography matrix
    transformed_points = cv2.perspectiveTransform(points, homography, (image_to_map.shape[1], image_to_map.shape[0]))
    
    # Extract the new bounding box coordinates
    x_coords = transformed_points[0, :, 0]
    y_coords = transformed_points[0, :, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    return [x_min, y_min, x_max, y_max]