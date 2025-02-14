import numpy as np
import math

def lookup_descriptor_bilinear(
    keypoint: np.ndarray, descriptor_map: np.ndarray
) -> np.ndarray:
  """Looks up descriptor value for keypoint from a dense descriptor map.

  Uses bilinear interpolation to find descriptor value at non-integer
  positions.

  Args:
    keypoint: 2-dim numpy array containing (x, y) keypoint image coordinates.
    descriptor_map: (H, W, D) numpy array representing a dense descriptor map.

  Returns:
    D-dim descriptor value at the input 'keypoint' location.

  Raises:
    ValueError, if kepoint position is out of bounds.
  """
  height, width = np.shape(descriptor_map)[:2]
  if (
      keypoint[0] < 0
      or keypoint[0] > width
      or keypoint[1] < 0
      or keypoint[1] > height
  ):
    raise ValueError(
        'Keypoint position (%f, %f) is out of descriptor map bounds (%i w x'
        ' %i h).' % (keypoint[0], keypoint[1], width, height)
    )

  x_range = [math.floor(keypoint[0])]
  if not keypoint[0].is_integer() and keypoint[0] < width-1:
    x_range.append(x_range[0] + 1)
  y_range = [math.floor(keypoint[1])]
  if not keypoint[1].is_integer() and keypoint[1] < height-1:
    y_range.append(y_range[0] + 1)

  bilinear_descriptor = np.zeros(np.shape(descriptor_map)[2])
  for curr_x in x_range:
    for curr_y in y_range:
      curr_descriptor = descriptor_map[curr_y, curr_x, :]
      bilinear_scalar = (1.0 - abs(keypoint[0] - curr_x)) * (
          1.0 - abs(keypoint[1] - curr_y)
      )
      bilinear_descriptor += bilinear_scalar * curr_descriptor
  return bilinear_descriptor
