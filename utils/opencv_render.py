import cv2 as cv
import colorsys
import numpy as np

class CVRender:
    def __init__(self, num_layers, img_size) -> None:
        self.layers = np.zeros((num_layers, img_size[0], img_size[1]), dtype=np.float32)
        
    def reset(self):
        self.layers = np.zeros((self.num_layers, self.img_size[0], self.img_size[1]), dtype=np.float32)
    
    def draw_hue_polygons(self, rasterized_directed_polygon):
        coors = np.argwhere(rasterized_directed_polygon[:, :, 0] == 1)
        for coor in coors:
            angle = np.arctan2(rasterized_directed_polygon[coor][2], rasterized_directed_polygon[coor][1]) * 180 / np.pi
            if angle < 0:
                angle += 360
                
            bgr_color = colorsys.hsv_to_rgb(angle/360.0, 1, 1)
            # Default opencv color mode is bgr, so we convert it to that
            rgb_color = (bgr_color[2], bgr_color[1], [bgr_color[0]])
            self.layers[:, coor[0], coor[1]] = np.array(rgb_color)
            
    def show_one_layer(self, window_name, layer_idx, offset=0.0, scale=255.0, do_block=False):
        cv.namedWindow(window_name, flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)

        r_chan = np.uint8((self.layers[layer_idx] + offset) * scale)

        out_img = np.stack([r_chan, r_chan, r_chan], axis=2)
        cv.imshow(window_name, out_img)
        if do_block:
            cv.waitKey()

    def show_two_layers(self, window_name, start_layer, offset=0.0, scale=255.0, do_block=False):
        cv.namedWindow(window_name, flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)

        r_chan = np.uint8((self.layers[start_layer] + offset) * scale)
        g_chan = np.uint8((self.layers[start_layer+1] + offset) * scale)
        b_chan = np.zeros((self.layers.shape[1], self.layers.shape[2]), dtype=np.uint8)

        out_img = np.stack([r_chan, g_chan, b_chan], axis=2)
        cv.imshow(window_name, out_img)
        if do_block:
            cv.waitKey()

    def show_three_layers(self, window_name, start_layer, offset=0.0, scale=255.0, do_block=False):
        cv.namedWindow(window_name, flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)

        r_chan = np.uint8((self.layers[start_layer] + offset) * scale)
        g_chan = np.uint8((self.layers[start_layer + 1] + offset) * scale)
        b_chan = np.uint8((self.layers[start_layer + 2] + offset) * scale)

        out_img = np.stack([r_chan, g_chan, b_chan], axis=2)
        cv.imshow(window_name, out_img)
        if do_block:
            cv.waitKey()