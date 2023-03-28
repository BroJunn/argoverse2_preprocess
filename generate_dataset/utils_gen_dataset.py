import numpy as np
import yaml

import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Polygon, Point

from typing import List, Dict, Tuple, Union

def read_yaml(path):
    with open(path, encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config

def onehot2label(onehot_array: np.ndarray) -> np.ndarray:
    """convert onehot coding to label index

    Args:
        onehot_array (np.ndarray): shape (N, K)
   
    Returns:
        np.ndarray: shape (N, )
    """   
    label_array = np.zeros((len(onehot_array), ))
    label_array = np.where(onehot_array==1)[1]

    return label_array.astype(np.int32)

def rasterize_polygons(polygons: List[List[List]],
                       raster_size: Union[List[int], Tuple[int]],
                       pixel_size: float,
                       av_coord: Union[Tuple, List, np.ndarray],
                       av_pos_in_raster: Union[List[int], Tuple[int], np.ndarray]) -> np.ndarray:
    """raster a set of points to coor_dict (key: index of point counting from 0, value: None if point is not observable in range), where the av is at position of "av_pos_in_raster" in rasterized image

    Args:
        polygons (List[List]): vertex coordinates of a set of polygons
        raster_size (Union[List, Tuple]): rasterized image size
        pixel_size (float): length(e.g. meter) per pixel
        av_coord (Union[np.ndarray, List]): the coordinate of av in original coordinate system
        av_pos_in_raster (Union[List, Tuple, np.ndarray]): the position of av in rasterized image

    Returns:
        np.ndarray: rasterized mask
    """
    assert len(av_coord) == 2 and len(av_pos_in_raster) == 2 and len(raster_size) == 2
    
    # compute bounds
    xmin = av_coord[0] - (av_pos_in_raster[0] * pixel_size)
    ymin = av_coord[1] - (av_pos_in_raster[1] * pixel_size)
    xmax = xmin + (raster_size[1] * pixel_size)
    ymax = ymin + (raster_size[0] * pixel_size)

    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, raster_size[1], raster_size[0])

    # convert vertexes of polygons to shapely.geometry.Polygon object
    polygon_shapes = [Polygon(poly) for poly in polygons]

    # get mask
    mask = geometry_mask(polygon_shapes, transform=transform, out_shape=raster_size, invert=True)
    
    return mask

def rasterize_points_2_coor_dict(points: List[List],
                                 raster_size: Union[List[int], Tuple[int]],
                                 pixel_size: float,
                                 av_coord: Union[List, Tuple, np.ndarray],
                                 av_pos_in_raster: Union[List[int], Tuple[int], np.ndarray]) -> Dict[int, Union[np.ndarray, None]]:
    """raster a set of points to coor_dict (key: index of point counting from 0, value: None if point is not observable in range), where the av is at position of "av_pos_in_raster" in rasterized image

    Args:
        points (List[List]): a set of point coordinates
        raster_size (Union[List, Tuple]): rasterized image size
        pixel_size (float): length(e.g. meter) per pixel
        av_coord (Union[np.ndarray, List]): the coordinate of av in original coordinate system
        av_pos_in_raster (Union[List, Tuple, np.ndarray]): the position of av in rasterized image

    Returns:
        Dict[int, Union[np.ndarray, None]]: coordinate of points in rasterized image while keeping index order
    """
    assert len(av_coord) == 2 and len(av_pos_in_raster) == 2 and len(raster_size) == 2
    
    # compute bounds
    xmin = av_coord[0] - (av_pos_in_raster[0] * pixel_size)
    ymin = av_coord[1] - (av_pos_in_raster[1] * pixel_size)
    xmax = xmin + (raster_size[1] * pixel_size)
    ymax = ymin + (raster_size[0] * pixel_size)
    
    bounds = (xmin, ymin, xmax, ymax)
    
    # convert coordinates to Point object
    points = [Point(x, y) for x, y in points]
    
    # convert Point object to coor in raster and keep the order the same
    transform = rasterio.transform.from_bounds(*bounds, raster_size[1], raster_size[0])
    shapes = [(point, i+1) for i, point in enumerate(points)]
    mask = rasterio.features.rasterize(shapes, out_shape=raster_size, transform=transform)
    
    coor_dict = {}
    for i in range(len(points)):
        coor = np.argwhere(mask == i+1)
        coor_dict[i] = coor if len(coor) != 0 else None
        
    return coor_dict

def rasterize_points_2_mask(points: List[List],
                            raster_size: Union[List, Tuple], 
                            pixel_size: float,
                            av_coord: Union[List, Tuple, np.ndarray],
                            av_pos_in_raster: Union[List, Tuple, np.ndarray]) -> np.ndarray:
    """raster a set of points to mask, where the av is at position of "av_pos_in_raster" in rasterized image

    Args:
        points (List[List]): a set of point coordinates
        raster_size (Union[List, Tuple]): rasterized image size
        pixel_size (float): length(e.g. meter) per pixel
        av_coord (Union[np.ndarray, List]): the coordinate of av in original coordinate system
        av_pos_in_raster (Union[List, Tuple, np.ndarray]): the position of av in rasterized image

    Returns:
        np.ndarray: rasterized mask
    """
    assert len(av_coord) == 2 and len(av_pos_in_raster) == 2 and len(raster_size) == 2
    
    # compute bounds
    xmin = av_coord[0] - (av_pos_in_raster[0] * pixel_size)
    ymin = av_coord[1] - (av_pos_in_raster[1] * pixel_size)
    xmax = xmin + (raster_size[1] * pixel_size)
    ymax = ymin + (raster_size[0] * pixel_size)
    
    bounds = (xmin, ymin, xmax, ymax)
    
    # convert coordinates to Point object
    points = [Point(x, y) for x, y in points]
    
    # convert Point object to mask
    transform = rasterio.transform.from_bounds(*bounds, raster_size[1], raster_size[0])
    shapes = [(point, 1) for point in points]
    mask = rasterio.features.rasterize(shapes, out_shape=raster_size, transform=transform)
    
    return mask


def transform_coords(coords: np.ndarray,
                     angle: float,
                     rotating_point: np.ndarray) -> np.ndarray:
    """transform 2D coordinates based on the rotation and translation

    Args:
        coords (np.ndarray): shape (..., 2)
        angle (float): angle w.r.t. rotating_point
        rotating_point (np.ndarray): shape (2, )

    Returns:
        np.ndarray: shape (..., 2)
    """        
    # calculate the sine and cosine of the heading
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([[cos_angle, -sin_angle], 
                                [sin_angle, cos_angle]])

    # subtract the coordinates of rotating_point from all the coordinates
    coords_for_rot = coords - rotating_point

    # rotate the coordinates around rotating_point
    transformed_coords = np.dot(coords_for_rot, rotation_matrix.T) + rotating_point

    return transformed_coords