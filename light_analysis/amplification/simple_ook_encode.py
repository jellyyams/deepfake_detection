import numpy as np
import cv2 
import sys

corner_markers = True

N = 15 #size of one cell in SLM pixels
block_dim = 3 #dimension of one block, in num cells
max_blocks_W = None
max_blocks_H = None
buffer_space = 15 #num pixels (SLM) between blocks
num_info_cells = block_dim*block_dim - 1 #number of info-bearing cells per block
output_dir_path = 'aug6_codes'
cell_color = (60, 0, 60)

W = 640
H = 360

if max_blocks_H and max_blocks_H:
    max_bits = max_blocks_H * max_blocks_W * num_info_cells
else:
    if corner_markers:
        max_blocks_W = int((W - 2*N + buffer_space) / (N*block_dim + buffer_space))
        print(max_blocks_W)
        max_blocks_H = int((H - 2*N + buffer_space) / (N*block_dim + buffer_space))
        print(max_blocks_H)
        max_bits = max_blocks_H * max_blocks_W * num_info_cells
    else:
        max_blocks_horiz = W / (N + buffer_space)



#build test bitstring to encode
input_bitstring = ''

for i in range(14):
    input_bitstring += '110101001101010110111001'
input_bitstring += '1101110101110110110101101111001010001011111101011111001110110000'


# for i in range(7):
#     input_bitstring += '110101001101010110111001'
# input_bitstring += '11011101011101101101011011110010100010111111010111110011'

# for i in range(3):
#     input_bitstring += '110101001101010110111001'
# input_bitstring += '11010100'

if len(input_bitstring) > max_bits:
    print(len(input_bitstring))
    print("Cannot encode this many bits")
    sys.exit()


def make_block_pair(input_bitstring, color):
    if len(input_bitstring) != num_info_cells:
        print("Invalid bitstring length for one block.\n Bitstring must be of length ", num_info_cells)
        return 

    block = np.zeros((block_dim * N, block_dim * N, 3))
    for i, b in enumerate(input_bitstring):
        if i >= int(num_info_cells / 2): #account for pilot cell in center
            i += 1
        cell_row = int(i / block_dim)
        cell_col = i % block_dim
   
        if b == '1':  
            cell_pix_row_start = cell_row * N
            if cell_pix_row_start != 0:
                cell_pix_row_start -= 1
            cell_pix_row_end = (cell_row + 1) * N
            cell_pix_col_start = cell_col * N
            if cell_pix_col_start != 0:
                cell_pix_col_start -= 1
            cell_pix_col_end = (cell_col + 1) * N

      
            block[cell_pix_row_start: cell_pix_row_end, cell_pix_col_start:cell_pix_col_end, 0] = color[0]
            block[cell_pix_row_start: cell_pix_row_end, cell_pix_col_start:cell_pix_col_end, 1] = color[1]
            block[cell_pix_row_start: cell_pix_row_end, cell_pix_col_start:cell_pix_col_end, 2] = color[2]
    

    blocks = [block]
    
    block_copy = block.copy()
    cell_rowcol = int(block_dim/2)
    block_copy[cell_rowcol * N:(cell_rowcol+1) * N, cell_rowcol * N:(cell_rowcol+1) * N, 0] = color[0]
    block_copy[cell_rowcol * N:(cell_rowcol+1) * N, cell_rowcol * N:(cell_rowcol+1) * N, 1] = color[1]
    block_copy[cell_rowcol * N:(cell_rowcol+1) * N, cell_rowcol * N:(cell_rowcol+1) * N, 2] = color[2]

    blocks.append(block_copy)
    
    return blocks

on_frame = np.zeros((H, W, 3))
off_frame = np.zeros((H, W, 3))
display = True

if cell_color[0] == 0:
    corner_marker_color = (255, 0, 0)
else:
    corner_marker_color = (0, 255, 0)

for i in range(0,  len(input_bitstring), num_info_cells):
    block_num = i / num_info_cells
    input_subbitstring = input_bitstring[i:i+num_info_cells]
    block_pair = make_block_pair(input_subbitstring, cell_color)
    
    block_row = int(block_num / max_blocks_W)
    block_col = int(block_num % max_blocks_W)

    block_pix_row_start = int(block_row * N * block_dim)
    # if block_pix_row_start != 0:
    #     block_pix_row_start -= 1
    block_pix_row_end = int((block_row + 1) * N * block_dim)
    block_pix_col_start = int(block_col * N * block_dim)
    # if block_pix_col_start != 0:
    #     block_pix_col_start -= 1
    block_pix_col_end = int((block_col + 1) * N * block_dim)

    if corner_markers:
        block_pix_row_start += N + buffer_space * block_row
        block_pix_row_end += N + buffer_space * block_row 
        block_pix_col_start += N + buffer_space * block_col
        block_pix_col_end += N + buffer_space * block_col
    else:
        block_pix_row_start += buffer_space * block_row
        block_pix_row_end += buffer_space * block_row 
        block_pix_col_start += buffer_space * block_col
        block_pix_col_end += buffer_space * block_col 
 
    print(block_pix_row_start, block_pix_row_end, block_pix_col_start, block_pix_col_end)
    on_frame[block_pix_row_start:block_pix_row_end, block_pix_col_start: block_pix_col_end, :] = block_pair[1]
    off_frame[block_pix_row_start:block_pix_row_end, block_pix_col_start: block_pix_col_end, :] = block_pair[0]


if corner_markers:
    on_frame[:N,:N,0] = corner_marker_color[0]
    on_frame[:N,:N,1] = corner_marker_color[1]
    on_frame[:N,:N,2] = corner_marker_color[2]

    on_frame[:N,W-N:W,0] = corner_marker_color[0]
    on_frame[:N,W-N:W,1] = corner_marker_color[1]
    on_frame[:N,W-N:W,2] = corner_marker_color[2]

    on_frame[H-N:H,:N,0] = corner_marker_color[0]
    on_frame[H-N:H,:N,1] = corner_marker_color[1]
    on_frame[H-N:H,:N,2] = corner_marker_color[2]

    on_frame[H-N:H,W-N:W,0] = corner_marker_color[0]
    on_frame[H-N:H,W-N:W,1] = corner_marker_color[1]
    on_frame[H-N:H,W-N:W,2] = corner_marker_color[2]

    off_frame[:N,:N,0] = corner_marker_color[0]
    off_frame[:N,:N,1] = corner_marker_color[1]
    off_frame[:N,:N,2] = corner_marker_color[2]

    off_frame[:N,W-N:W,0] = corner_marker_color[0]
    off_frame[:N,W-N:W,1] = corner_marker_color[1]
    off_frame[:N,W-N:W,2] = corner_marker_color[2]

    off_frame[H-N:H,:N,0] = corner_marker_color[0]
    off_frame[H-N:H,:N,1] = corner_marker_color[1]
    off_frame[H-N:H,:N,2] = corner_marker_color[2]

    off_frame[H-N:H,W-N:W,0] = corner_marker_color[0]
    off_frame[H-N:H,W-N:W,1] = corner_marker_color[1]
    off_frame[H-N:H,W-N:W,2] = corner_marker_color[2]

   
if display:
    cv2.imshow('testwin', on_frame)
    cv2.waitKey(0) # wait for ay key to exit window
    cv2.destroyAllWindows() # close all windows

    cv2.imshow('testwin', off_frame)
    cv2.waitKey(0) # wait for ay key to exit window
    cv2.destroyAllWindows() # close all windows

#flip to account for slm's flipped display
on_frame = cv2.flip(on_frame, 0)
on_frame = cv2.flip(on_frame, 1)
off_frame = cv2.flip(off_frame, 0)
off_frame = cv2.flip(off_frame, 1)
cv2.imwrite('{}/on_frame_r{}_g{}_b{}_N{}_buff{}.bmp'.format(output_dir_path, cell_color[2], cell_color[1], cell_color[0], N, buffer_space), on_frame)
cv2.imwrite('{}/off_frame_r{}_g{}_b{}_N{}_buff{}.bmp'.format(output_dir_path, cell_color[2], cell_color[1], cell_color[0], N, buffer_space), off_frame)
