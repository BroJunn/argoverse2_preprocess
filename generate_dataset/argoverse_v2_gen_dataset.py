import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import copy
from matplotlib import pyplot as plt
import sys


import torch
from torch.utils.data import Dataset

from av2.datasets.motion_forecasting import scenario_serialization 
from av2.map.map_api import ArgoverseStaticMap

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from utils_gen_dataset import rasterize_polygons, transform_coords, rasterize_points_2_coor_dict
from utils.utils_general import read_yaml

class Argov2PreprocessDataset(Dataset):
    def __init__(self,
                 root: str,
                 config: Dict[str, any]) -> None:
        self.root = root
        self.config = config

        self.scene_names = self._get_index_map()
        self.scenario_raw_data_paths = sorted(list(Path(root).rglob("*.parquet")))
        self.map_raw_data_paths = sorted(list(Path(root).rglob("*.json")))

        assert len(self.scene_names) == len(self.scenario_raw_data_paths) == len(self.map_raw_data_paths), 'There exists scene lack of raw data file'

        self.obj_class_to_ohe_idx = {
        "vehicle": 0,
        "pedestrian": 1,
        "motorcyclist": 2, 
        "cyclist": 3,
        "bus": 4,
        "static": 5, 
        "background": 5, 
        "construction": 5, 
        "riderless_bicycle": 5, 
        "unknown": 6
        }

        self.LENGTH_VEHICLE = self.config['LENGTH_VEHICLE']
        self.WIDTH_VEHICLE = self.config['WIDTH_VEHICLE']
        self.LENGTH_BUS = self.config['LENGTH_BUS']
        self.WIDTH_BUS = self.config['WIDTH_BUS']
    
    def __getitem__(self, index: int) -> Dict:
        argo_data = self._read_argo_data(index)
        processed_argo_data = self._process_argo_data(argo_data)
        
        dic_map = self._read_argo_map(index)
        
        rasterized_map, list_occu_mask_of_diff_labels = self._rasterize(processed_argo_data, dic_map, argo_data)
        dict_rasterized_scene = {}
        dict_rasterized_scene['argo_id'] = argo_data['argo_id']
        # dict_rasterized_scene['config'] = self.config
        dict_rasterized_scene['rasterized_map'] = rasterized_map
        # dict_rasterized_scene['rasterized_agents'] = np.array(list_occu_mask_of_diff_labels)
        dict_rasterized_scene['rasterized_agents_coors'] = \
            self._mask2coor(np.array(list_occu_mask_of_diff_labels))
        
        rendered_map_temp = copy.deepcopy(rasterized_map)
        rendered_map_temp[:, :, 1:] = ((rendered_map_temp[:, :, 1:] + 1.0) / 2 * 255)
        rendered_map_temp[:, :, 0] *= 255
        rendered_map_temp = rendered_map_temp.astype(np.uint8)
        
        plt.figure()
        plt.imshow(rendered_map_temp)
        plt.savefig(f'./generate_dataset/temp/{index}.png')
        
        return dict_rasterized_scene
    
    def __len__(self) -> int:
        return len(self.scene_names)

    def _get_index_map(self) -> List:
        return sorted(os.listdir(self.root))
    
    def _read_argo_data(self, idx: int) -> Dict[str, any]:
        """read each scene from raw data and arrange them

        Args:
            idx (int): index

        Returns:
            Dict[str, any]: all kinds of information are ordered by [target vehicle, av, ...]
        """  
        scenario = scenario_serialization.load_argoverse_scenario_parquet(self.scenario_raw_data_paths[idx])
        city = copy.deepcopy(scenario.city_name)

        trajs, headings, vels, obj_classes, obj_types, obj_ids, steps, obj_label_indexes  = [], [], [], [], [], [], [], []
        for track in scenario.tracks:
            traj = np.array([list(object_state.position) for object_state in track.object_states])
            heading =  np.array([object_state.heading for object_state in track.object_states])
            vel = np.array([list(object_state.velocity) for object_state in track.object_states])
            step = np.array([object_state.timestep for object_state in track.object_states])
            obj_label_index = self.obj_class_to_ohe_idx[track.object_type]
            
            obj_class = np.zeros((max(self.obj_class_to_ohe_idx.values())+1, 50), np.float32) # class: vehicle, pedestrian, motorcyclist, cyclist, bus
            if track.object_type in self.obj_class_to_ohe_idx:
                obj_class[obj_label_index] = 1
            else:
                # Object types could be unknown (which then results in an empty OHE)
                pass

            obj_id = track.track_id
            obj_type = track.category.value

            obj_types.append(obj_type)      # indicates whether it is the target/focal agent or not --> 3 == focal agent
            obj_classes.append(obj_class)   # indicates the object class
            obj_ids.append(obj_id)          # contains the object id, from this the AV can be idetified --> obj_id == AV
            trajs.append(traj)
            headings.append(heading)
            vels.append(vel)
            steps.append(step)
            obj_label_indexes.append(obj_label_index)

        agt_idx = obj_types.index(3)                    # get target/ focal agent index
        trajs.insert(0, trajs.pop(agt_idx))
        headings.insert(0, headings.pop(agt_idx))
        vels.insert(0, vels.pop(agt_idx))
        obj_classes.insert(0, obj_classes.pop(agt_idx))
        obj_types.insert(0, obj_types.pop(agt_idx))
        obj_ids.insert(0, obj_ids.pop(agt_idx))
        steps.insert(0, steps.pop(agt_idx))
        obj_label_indexes.insert(0, obj_label_indexes.pop(agt_idx))

        av_idx = obj_ids.index('AV')                    # get AV index
        trajs.insert(1, trajs.pop(av_idx))
        headings.insert(1, headings.pop(av_idx))
        vels.insert(1, vels.pop(av_idx))
        obj_classes.insert(1, obj_classes.pop(av_idx))
        obj_types.insert(1, obj_types.pop(av_idx))
        obj_ids.insert(1, obj_ids.pop(av_idx))
        steps.insert(1, steps.pop(av_idx))
        obj_label_indexes.insert(1, obj_label_indexes.pop(av_idx))
        
        data = dict()
        data["argo_id"] = scenario.scenario_id 
        data['city'] = city
        data['trajs'] = trajs
        data['steps'] = steps
        data['vels'] = vels
        data['headings'] = headings
        data['obj_classes_onehot'] = np.asarray(obj_classes, np.float32)
        data['track_ids'] = obj_ids
        data['obj_classes_label'] = obj_label_indexes
        return data

    def _process_argo_data(self, an_argo_data: Dict[str, any]) -> List[Dict[str, any]]:
        """process the argo_data of a scene into "time sequence form"

        Args:
            an_argo_data (Dict[str, any]): _description_

        Returns:
            List[Dict[str, any]]: list of time sequence
        """        
        mask_aval = np.zeros((self.config['NUM_ALL_STEPS'] ,len(an_argo_data['steps'])))
        
        for i in range(len(an_argo_data['steps'])):
            mask_aval[an_argo_data['steps'][i], i] = 1
        mask_aval = mask_aval.astype(np.int32)
        
        list_info_in_time_sequence = []
        for i in range(self.config['NUM_ALL_STEPS']):
            dict_info_timestep = self._get_info_at_a_timestep(an_argo_data, mask_aval, i)
            
            rec_pts = self._gen_rec_pts(dict_info_timestep['trajs'], dict_info_timestep['headings'], dict_info_timestep['obj_classes_label'])
            dict_info_timestep['rec_pts'] = rec_pts 
            
            # visualization test
            # plt.figure()
            
            # plt.scatter(dict_info_timestep['trajs'][:, 0], dict_info_timestep['trajs'][:, 1], c='r')
            # for n in range(rec_pts.shape[0]):
            #     for m in range(rec_pts.shape[1]):
            #         plt.scatter(rec_pts[n, m, 0], rec_pts[n, m, 1], c='g')
            # plt.axis("equal")
            # plt.show()
            # print(1)
            list_info_in_time_sequence.append(dict_info_timestep)
        
        return list_info_in_time_sequence
        
    def _get_info_at_a_timestep(self,
                                an_argo_data: Dict,
                                mask_aval: np.ndarray,
                                timestep: int) -> Dict[str, np.ndarray]:
        """get trajs, headings and class_label of all agents at a single time step

        Args:
            an_argo_data (Dict): argo data of the scene
            mask_aval (np.ndarray): mask that denotes if agents are available at this time step
            timestep (int):

        Returns:
            Dict[str, np.ndarray]: info_dict at a time step
        """        

        avail_agent_index: list = np.where(mask_aval[timestep] == 1)[0]
        dict_info_timestep = {}
        list_temp_trajs, list_temp_headings, list_temp_classes = [], [], []
        
        for i in avail_agent_index:
            idx_cur_timestep = np.where(an_argo_data['steps'][i] == timestep)[0]
            list_temp_trajs.append(an_argo_data['trajs'][i][idx_cur_timestep.item()])
            list_temp_headings.append(an_argo_data['headings'][i][idx_cur_timestep.item()])
            list_temp_classes.append(an_argo_data['obj_classes_label'][i])

        dict_info_timestep['trajs'] = np.array(list_temp_trajs)
        dict_info_timestep['headings'] = np.array(list_temp_headings)
        dict_info_timestep['obj_classes_label'] = np.array(list_temp_classes)
        
        return dict_info_timestep

    def _offset_per_agent(self, obj_type: str) -> np.ndarray:
        """generate offsets based on the class of agent

        Args:
            obj_type (str): _description_

        Raises:
            KeyError: no such agent class with shape

        Returns:
            np.ndarray: offsets
        """        
        if obj_type == "vehicle":
            suffix = 'VEHICLE'
        elif obj_type == 'bus':
            suffix = 'BUS'
        else:
            raise KeyError
        
        offset = np.array([[eval('self.LENGTH_' + suffix) / 2, eval('self.WIDTH_' + suffix) / 2],
                           [-eval('self.LENGTH_' + suffix) / 2, eval('self.WIDTH_' + suffix) / 2],
                           [-eval('self.LENGTH_' + suffix) / 2, -eval('self.WIDTH_' + suffix) / 2],
                           [eval('self.LENGTH_' + suffix) / 2, -eval('self.WIDTH_' + suffix) / 2]])

        return offset

    def _gen_rec_pts(self,
                     center_points: np.ndarray,
                     headings: np.ndarray,
                     types: np.ndarray) -> np.ndarray:
        """generate a rec pts based on the size and heading of the agents

        Args:
            center_points (np.ndarray): shape (N, 2)
            headings (np.ndarray): shape (N,)
            types (np.ndarray): shape (N,)

        Returns:
            np.ndarray: shape (N, 4 2)
        """    
        assert len(center_points) == len(headings) == len(types)

        offsets = np.zeros((len(center_points), 4, 2))
        offsets[np.where(types == 0)] = self._offset_per_agent("vehicle")
        offsets[np.where(types == 4)] = self._offset_per_agent("bus")
        
        rec_pts = copy.deepcopy(offsets)
        
        for i in range(len(center_points)):
            rec_pts[i] = transform_coords(offsets[i], headings[i], np.array([0, 0])) + center_points[i]
            
        return rec_pts

    def _read_argo_map(self, idx: int) -> Dict[int, Dict[str, any]]:
        """read argo_map of a scene from raw data

        Args:
            idx (int): index

        Returns:
            Dict[int, Dict[str, any]: info_dict of argo map
        """        
        avm = ArgoverseStaticMap.from_json(self.map_raw_data_paths[idx])
        dic_map = {}
        for seg_id, lane_segment in avm.vector_lane_segments.items():
            dic_temp = {}
            center_line_points = avm.get_lane_segment_centerline(seg_id)[:, :2]
            dic_temp['center_line_points'] = center_line_points
            dic_temp['polygon_boundary'] = lane_segment.polygon_boundary[:-1, :2]
            dic_temp['successors'] = lane_segment.successors
            dic_map[seg_id] = dic_temp

        return dic_map
    
    
    def _render_rasterized_map_with_direction(self,
                                              dict_lane_boundary: Dict[int, List[List]],
                                              dict_center_line_point: Dict[int, List[List]],
                                              dict_center_line_points_direction: Dict[int, List[List]],
                                              av_coor: np.ndarray) -> np.ndarray:
        """render rasterized_map_with_derection

        Args:
            dict_lane_boundary (Dict[int, List[List]]): info_dict of (transformed) lane boundary
            dict_center_line_point (Dict[int, List[List]]): info_dict of (transformed) center_line_point
            dict_center_line_points_direction (Dict[int, List[List]]): info_dict of (transformed) center_line_point
            av_coor (np.ndarray): shape (2,)
        
        Returns:
            np.ndarray: shape (grid_height_cells, grid_width_cells, 3)  1.channel = occupancy of lane; 2nd&3rd channel = normalized direction
        """                
        
        rendered_map = np.zeros((self.config['grid_height_cells'], self.config['grid_width_cells'], 3))
        count_map = np.zeros((self.config['grid_height_cells'], self.config['grid_width_cells']))

        for seg_id, lane_boundary in dict_lane_boundary.items():
            mask_lane = rasterize_polygons([lane_boundary],
                                           (self.config['grid_height_cells'], self.config['grid_width_cells']),
                                           1 / self.config['pixels_per_meter'],
                                           av_coor,
                                           (self.config['sdc_x_in_grid'], self.config['grid_height_cells'] - self.config['sdc_y_in_grid']))
            
            center_line_point_coor_dict = \
                rasterize_points_2_coor_dict(dict_center_line_point[seg_id],
                                             (self.config['grid_height_cells'], self.config['grid_width_cells']),
                                             1 / self.config['pixels_per_meter'],
                                             av_coor,
                                             (self.config['sdc_x_in_grid'], self.config['grid_height_cells'] - self.config['sdc_y_in_grid']))
                
            mask_lane_coor = np.argwhere(mask_lane)
            map_center_line_point_index = {}
            mask_center_line_point_coor = []
            for i, coor in enumerate(center_line_point_coor_dict.values()):
                if isinstance(coor, np.ndarray):
                    mask_center_line_point_coor.append(coor)
                    map_center_line_point_index[len(mask_center_line_point_coor)-1] = i
            if len(mask_center_line_point_coor) != 0:
                mask_center_line_point_coor = np.squeeze(np.array(mask_center_line_point_coor), 1)
            #  or len(mask_center_line_point_coor) == 0        
            if len(mask_lane_coor) == 0 or len(mask_center_line_point_coor) == 0  :
                continue
            mask_lane_coor_temp = np.repeat(mask_lane_coor[:, np.newaxis, :], len(mask_center_line_point_coor), axis=1)
            mask_center_line_point_coor_temp = np.repeat(mask_center_line_point_coor[np.newaxis, :, :], len(mask_lane_coor), axis=0)
            dis_map = np.linalg.norm(mask_lane_coor_temp - mask_center_line_point_coor_temp, axis=2)
            assign_center_line_point = np.argmin(dis_map, axis=1)
            
            # generate image
            for i in range(len(mask_lane_coor)):
                rendered_map[mask_lane_coor[i][0], mask_lane_coor[i][1], 0] = 1
                rendered_map[mask_lane_coor[i][0], mask_lane_coor[i][1], 1:] += \
                    dict_center_line_points_direction[seg_id][map_center_line_point_index[assign_center_line_point[i]]]
                count_map[mask_lane_coor[i][0], mask_lane_coor[i][1]] += 1
            
        count_map = count_map.astype(np.int8)
        for coor in np.argwhere(count_map != 0):
            rendered_map[coor[0], coor[1], 1:] /= count_map[coor[0], coor[1]]

        return rendered_map
        # rendered_map[:, :, 1:] = ((rendered_map[:, :, 1:] + 1.0) / 2 * 255)
        # rendered_map[:, :, 0] *= 255
        # rendered_map = rendered_map.astype(np.uint8)
        
        
        
            # dict_center_line_point[seg_id] = 
            # if seg_id == 353687823 or seg_id == 353687939 or seg_id == 353622963:
            #     plt.figure()
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(rendered_map)
            #     # plt.savefig('temp.png')
            #     # plt.show()
            #     print(1)
            

        # plt.figure()
        # plt.imshow(rendered_map)
        # # plt.imshow(rendered_map, origin='lower')
        # # plt.savefig('./scene_roadmap_1_avg.png')
        # plt.show()
        # print(1)
            
            # plt.figure()
            
            
            
                # plt.subplot(1, 2, 2)
                # plt.plot(np.array(dict_lane_boundary[seg_id])[:, 0], np.array(dict_lane_boundary[seg_id])[:, 1], c='g')
                # dx = np.array(dict_center_line_point[seg_id])[1:, 0] - np.array(dict_center_line_point[seg_id])[:-1, 0]
                # dy = np.array(dict_center_line_point[seg_id])[1:, 1] - np.array(dict_center_line_point[seg_id])[:-1, 1]
                # for i in range(len(dx)):       
                #     plt.arrow(np.array(dict_center_line_point[seg_id])[i+1, 0], np.array(dict_center_line_point[seg_id])[i+1, 1], dx[i], dy[i], head_width=0.3, head_length=0.5, fc='red', ec='red')   
                # plt.scatter(np.array(dict_center_line_point[seg_id])[:, 0], 
                #             np.array(dict_center_line_point[seg_id])[:, 1], marker='o')
                # plt.scatter(np.array(dict_center_line_point[seg_id])[:2, 0] + np.array(dict_transformed_center_line_points_direction[seg_id])[:2, 0], 
                #         np.array(dict_center_line_point[seg_id])[:2, 1] + np.array(dict_transformed_center_line_points_direction[seg_id])[:2, 1], marker='+')
                # plt.scatter(np.array(dict_center_line_point[seg_id])[2:, 0] + np.array(dict_transformed_center_line_points_direction[seg_id])[2:, 0], 
                #         np.array(dict_center_line_point[seg_id])[2:, 1] + np.array(dict_transformed_center_line_points_direction[seg_id])[2:, 1], marker='*')
                # plt.show()
            # plt.savefig(f'./img_temp_3/temp_{seg_id}.png')
            # plt.close()
            
            
            
            # for i in range(len(dict_transformed_center_line_points_direction[seg_id])):
            #     plt.scatter(dict_transformed_center_line_points_direction[seg_id][i][0] + dict_center_line_point[seg_id])[i, 0],
            #                 dict_transformed_center_line_points_direction[seg_id][i][1], marker='+')
            
            # plt.subplot(1, 3, 3)
            # for i in range(len(dict_transformed_center_line_points_direction[seg_id])):    
            #     if i <= 2:
            #         plt.scatter(dict_transformed_center_line_points_direction[seg_id][i][0],dict_transformed_center_line_points_direction[seg_id][i][1], marker='+')
            #     else:
            #         plt.scatter(dict_transformed_center_line_points_direction[seg_id][i][0],dict_transformed_center_line_points_direction[seg_id][i][1], marker='*')
            # plt.savefig(f'./img_temp_3/temp_{seg_id}.png')
            # plt.close()
            # print() # shenfen zuo; shenhuang xia; huang you;fen shang
        
    
    def _rasterize(self,
                   processed_argo_data: List[Dict[str, any]],
                   argo_map: Dict[int, Dict[str, any]],
                   argo_data: Dict[str, any]):
        """_summary_

        Args:
            processed_argo_data (List[Dict[str, any]]): argo data in time sequence
            argo_map (Dict[int, Dict[str, any]]): info_dict of argo map
            argo_data (Dict[str, any]): info_dict arranged from raw data
        """        
        av_coor = argo_data['trajs'][1][self.config['referenced_time_step']]
        av_heading = argo_data['headings'][1][self.config['referenced_time_step']]
        
        # transform rectangle points of agents
        transformed_rec_pts = []
        for rec_pts_per_timestep in processed_argo_data:
            transformed_rec_pts.append(transform_coords(rec_pts_per_timestep['rec_pts'], np.pi/2 - av_heading, av_coor))
        
        # get agents occupancy
        list_occu_mask_of_diff_labels = []
        for i, rec_pts_per_timestep in enumerate(processed_argo_data):
            list_masks_temp = []
            for k in range(7):
                array_agents_poly = transformed_rec_pts[i][rec_pts_per_timestep['obj_classes_label'] == k]
                if len(array_agents_poly) != 0:
                    mask_agents = rasterize_polygons(array_agents_poly.tolist(),
                                                    (self.config['grid_height_cells'], self.config['grid_width_cells']),
                                                    1 / self.config['pixels_per_meter'],
                                                    av_coor,
                                                    (self.config['sdc_x_in_grid'], self.config['grid_height_cells'] - self.config['sdc_y_in_grid']))
                else:
                    mask_agents = np.zeros((self.config['grid_height_cells'], self.config['grid_width_cells']))
                list_masks_temp.append(mask_agents)
            list_occu_mask_of_diff_labels.append(np.array(list_masks_temp))
            
        # transform lane_boundary and center_line_point
        dict_lane_boundary, dict_center_line_point = {}, {}
        for seg_id, lane_dict in argo_map.items():
            transformed_lane_boundary = transform_coords(lane_dict['polygon_boundary'], np.pi/2 - av_heading, av_coor)
            transformed_center_line_point = transform_coords(lane_dict['center_line_points'], np.pi/2 - av_heading, av_coor)
            dict_lane_boundary[seg_id] = transformed_lane_boundary.tolist()
            dict_center_line_point[seg_id] = transformed_center_line_point.tolist()
        
        # get transformed center_line_point_directions
        dict_transformed_center_line_points_direction = {}
        for seg_id, transformed_center_line_point in dict_center_line_point.items():
            center_line_points = np.array(transformed_center_line_point)
            transformed_center_line_points_direction = np.zeros(center_line_points.shape)
            transformed_center_line_points_direction[:-1] = center_line_points[1:] - center_line_points[:-1]
            
            if len(argo_map[seg_id]['successors']) != 0:
                bool_valid_successor = False
                for successor in argo_map[seg_id]['successors']:
                    if successor in argo_map.keys():
                        transformed_center_line_points_direction[-1] = dict_center_line_point[successor][1] - center_line_points[-1]
                        bool_valid_successor = True
                        break
                    
                if not bool_valid_successor:
                    transformed_center_line_points_direction[-1] = transformed_center_line_points_direction[-2]
                    # print(f'{seg_id}\'s successor cannot be found in this scene')
                    
            else:
                transformed_center_line_points_direction[-1] = transformed_center_line_points_direction[-2]
                # print(f'no successor for this {seg_id}')
            transformed_center_line_points_direction /= np.linalg.norm(transformed_center_line_points_direction, axis=1, keepdims=True) 
            dict_transformed_center_line_points_direction[seg_id] = transformed_center_line_points_direction
            
        rasterized_map = self._render_rasterized_map_with_direction(dict_lane_boundary,
                                                                    dict_center_line_point,
                                                                    dict_transformed_center_line_points_direction,
                                                                    av_coor)

        return rasterized_map, list_occu_mask_of_diff_labels

    def _mask2coor(self, mask: np.ndarray) -> np.ndarray:
        """convert mask to coordinates

        Args:
            mask (np.ndarray): 

        Returns:
            np.ndarray: 
        """        
        return np.argwhere(mask == 1)
        
# if __name__ == '__main__':
#     dataset = Argov2PreprocessDataset('/home/yujun/Dataset/train/', read_yaml('/home/yujun/Code/argoverse2_preprocess/generate_dataset/config/gen_dataset.yaml'))
#     dataset.__getitem__(2)
