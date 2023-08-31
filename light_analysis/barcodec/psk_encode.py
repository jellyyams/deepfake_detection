import numpy as np
import cv2
import os
from enum import Enum
import sys

class BorderType(Enum):
    SOLID = 1
    GRID = 2

corner_markers = False
border_type = BorderType.GRID

N = 30 #size of one cell in SLM pixels
buffer_space = 30  #num pixels (SLM) between cells
cell_color = (0, 0, 140)
border_width = 30 #must be leq N
border_buffer_space = 30

if cell_color[0] == 0:
    corner_marker_color = (255, 0, 0)
else:
    corner_marker_color = (0, 255, 0)


W = 640
H = 360

if corner_markers:
    max_info_cells_W = int((W - 2*N - 2*border_width - border_buffer_space) / (N + buffer_space))
    max_info_cells_H = int((H - 2*N - 2*border_width - border_buffer_space) / (N + buffer_space))

    max_border_cells_W = int((W -2*N) / (border_width + border_buffer_space))
    max_border_cells_H = int((H -2*N + border_buffer_space) / (border_width + border_buffer_space))
else:
    max_info_cells_W = int((W - 2*border_width - border_buffer_space) / (N + buffer_space))
    max_info_cells_H = int((H - 2*border_width - border_buffer_space) / (N + buffer_space))

    max_border_cells_W = int((W + border_buffer_space) / (border_width + border_buffer_space))
    max_border_cells_H = int((H + border_buffer_space)/ (border_width + border_buffer_space))
   
max_bits = max_info_cells_H * max_info_cells_W 


"""
Specify the mapping between a symbol and the phase offset
could also be something like a : 0, b: 1, c : 3, d: 4
"""
symbol_map = {
    0 : 0,
    1 : 1
}
num_symbols = len(symbol_map.keys())

output_dir_path = 'aug27_psk/r{}_g{}_b{}_N{}_b{}_s{}'.format(cell_color[2], cell_color[1], cell_color[0], N, border_width, num_symbols)
os.makedirs(output_dir_path, exist_ok=True)


def add_border(frame):
    if corner_markers:
        offset = N
    else:
        offset = 0

   
    bottom_row = (max_border_cells_H - 1) * (border_width+border_buffer_space) + offset
    right_col = (max_border_cells_W - 1) * (border_width+border_buffer_space) + offset
 
   
    for i in range(max_border_cells_W):
        frame[offset:offset+border_width, offset+i*(border_width+border_buffer_space):offset+i*(border_width+border_buffer_space)+border_width, 0] = cell_color[0]
        frame[offset:offset+border_width, offset+i*(border_width+border_buffer_space):offset+i*(border_width+border_buffer_space)+border_width, 1] = cell_color[1]
        frame[offset:offset+border_width, offset+i*(border_width+border_buffer_space):offset+i*(border_width+border_buffer_space)+border_width, 2] = cell_color[2]

        frame[bottom_row:bottom_row+border_width, offset+i*(border_width + border_buffer_space):offset+i*(border_width+border_buffer_space)+border_width, 0] = cell_color[0]
        frame[bottom_row:bottom_row+border_width, offset+i*(border_width+border_buffer_space):offset+i*(border_width+border_buffer_space)+border_width, 1] = cell_color[1]
        frame[bottom_row:bottom_row+border_width, offset+i*(border_width+border_buffer_space):offset+i*(border_width+border_buffer_space)+border_width, 2] = cell_color[2]
       

        if border_type == BorderType.SOLID and i < max_border_cells_W - 1:
            frame[offset:offset+border_width, offset+i*(border_width+border_buffer_space)+border_width:offset+i*(border_width+border_buffer_space)+2*border_width, 0] = cell_color[0]
            frame[offset:offset+border_width, offset+i*(border_width+border_buffer_space)+border_width:offset+i*(border_width+border_buffer_space)+2*border_width, 1] = cell_color[1]
            frame[offset:offset+border_width, offset+i*(border_width+border_buffer_space)+border_width:offset+i*(border_width+border_buffer_space)+2*border_width, 2] = cell_color[2]

            frame[bottom_row:bottom_row+border_width, offset+i*(border_width + border_buffer_space)+border_width:offset+i*(border_width+border_buffer_space)+2*border_width, 0] = cell_color[0]
            frame[bottom_row:bottom_row+border_width, offset+i*(border_width+border_buffer_space)+border_width:offset+i*(border_width+border_buffer_space)+2*border_width, 1] = cell_color[1]
            frame[bottom_row:bottom_row+border_width, offset+i*(border_width+border_buffer_space)+border_width:offset+i*(border_width+border_buffer_space)+2*border_width, 2] = cell_color[2]
        
            

    for i in range(max_border_cells_H):
        frame[offset+i*(border_width+border_buffer_space):offset+i*(border_width+border_buffer_space)+border_width, offset:offset+border_width, 0] = cell_color[0]
        frame[offset+i*(border_width+border_buffer_space):offset+i*(border_width+border_buffer_space)+border_width, offset:offset+border_width, 1] = cell_color[1]
        frame[offset+i*(border_width+border_buffer_space):offset+i*(border_width+border_buffer_space)+border_width, offset:offset+border_width, 2] = cell_color[2]
      
      
        frame[offset+i*(border_width+border_buffer_space):offset+i*(border_width+border_buffer_space)+border_width, right_col:right_col+border_width, 0] = cell_color[0]
        frame[offset+i*(border_width+border_buffer_space):offset+i*(border_width+border_buffer_space)+border_width, right_col:right_col+border_width, 1] = cell_color[1]
        frame[offset+i*(border_width+border_buffer_space):offset+i*(border_width+border_buffer_space)+border_width, right_col:right_col+border_width, 2] = cell_color[2]
        
        if border_type == BorderType.SOLID and i < max_border_cells_H - 1:
            frame[offset+i*(border_width+border_buffer_space)+border_width:offset+i*(border_width+border_buffer_space)+2*border_width, offset:offset+border_width, 0] = cell_color[0]
            frame[offset+i*(border_width+border_buffer_space)+border_width:offset+i*(border_width+border_buffer_space)+2*border_width, offset:offset+border_width, 1] = cell_color[1]
            frame[offset+i*(border_width+border_buffer_space)+border_width:offset+i*(border_width+border_buffer_space)+2*border_width, offset:offset+border_width, 2] = cell_color[2]
        
        
            frame[offset+i*(border_width+border_buffer_space)+border_width:offset+i*(border_width+border_buffer_space)+2*border_width, right_col:right_col+border_width, 0] = cell_color[0]
            frame[offset+i*(border_width+border_buffer_space)+border_width:offset+i*(border_width+border_buffer_space)+2*border_width, right_col:right_col+border_width, 1] = cell_color[1]
            frame[offset+i*(border_width+border_buffer_space)+border_width:offset+i*(border_width+border_buffer_space)+2*border_width, right_col:right_col+border_width, 2] = cell_color[2]
            

    return frame
   
def add_corner_markers(frame):

    frame[:N,:N,0] = corner_marker_color[0]
    frame[:N,:N,1] = corner_marker_color[1]
    frame[:N,:N,2] = corner_marker_color[2]

    frame[:N,W-N:W,0] = corner_marker_color[0]
    frame[:N,W-N:W,1] = corner_marker_color[1]
    frame[:N,W-N:W,2] = corner_marker_color[2]

    frame[H-N:H,:N,0] = corner_marker_color[0]
    frame[H-N:H,:N,1] = corner_marker_color[1]
    frame[H-N:H,:N,2] = corner_marker_color[2]

    frame[H-N:H,W-N:W,0] = corner_marker_color[0]
    frame[H-N:H,W-N:W,1] = corner_marker_color[1]
    frame[H-N:H,W-N:W,2] = corner_marker_color[2]

    return frame


#build test bitstring to encode
input_bitstring = ''

# for i in range(5):
#     input_bitstring += '01100100111001101001000010010100011101'

# for i in range(5):
#     input_bitstring += '0110010111010101101'

for i in range(3):
    input_bitstring += '0110011101'

if len(input_bitstring) > max_bits:
    print("Input bitstring too long.")
    sys.exit(0)

offset = border_width + buffer_space
if corner_markers:
    offset += N 

for i in range(num_symbols):
    frame = np.zeros((H, W, 3)).astype(np.float32)
    for cell_num, b in enumerate(input_bitstring):

        phase = symbol_map[int(b)]
        # if phase % num_symbols == i  or (phase + 1) % num_symbols == i:
        if i in [(phase % num_symbols) + q for q in range(int(num_symbols/2))]:
            cell_row = int(cell_num / max_info_cells_W)
            cell_col = int(cell_num % max_info_cells_W)

            cell_top = offset + cell_row*(N+buffer_space) 
            cell_left = offset + cell_col*(N+buffer_space) 

            frame[cell_top:cell_top+N+1, cell_left:cell_left+N+1, 0] = cell_color[0]
            frame[cell_top:cell_top+N+1, cell_left:cell_left+N+1, 1] = cell_color[1]
            frame[cell_top:cell_top+N+1, cell_left:cell_left+N+1, 2] = cell_color[2]
    
    if i < (num_symbols / 2):
        frame = add_border(frame)

    if corner_markers:
        frame = add_corner_markers(frame)

    cv2.imshow('Code', frame)
    cv2.waitKey(0)

    frame = cv2.flip(frame, 0)
    
    #chr(ord('@')+i+1)
    cv2.imwrite('{}/frame{}.bmp'.format(output_dir_path, i), frame)
