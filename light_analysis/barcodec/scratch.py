#checkerboard-esque border

# if border_type == BorderType.GRID:
    #     h = int(H / N)
    #     w = int(W / N)
    #     checkerboard_chan1 = cell_color[0] * np.kron([[1, 0] * (w//2), [0, 1] * (w//2)] * (h//2), np.ones((N, N)))
    #     checkerboard_chan2 = cell_color[1] * np.kron([[1, 0] * (w//2), [0, 1] * (w//2)] * (h//2), np.ones((N, N)))
    #     checkerboard_chan3 = cell_color[2] * np.kron([[1, 0] * (w//2), [0, 1] * (w//2)] * (h//2), np.ones((N, N)))

    #     checkerboard = np.dstack((np.dstack((checkerboard_chan1, checkerboard_chan2)), checkerboard_chan3))
    #     bordered_frame = checkerboard.copy()
    #     bordered_frame[N:-N,N:-N] = frame[N:-N,N:-N]    
     
    #     return bordered_frame
    # else:   
    #     frame[:,0:N, 0] = cell_color[0]
    #     frame[:,0:N, 1] = cell_color[1]
    #     frame[:,0:N, 2] = cell_color[2]

    #     frame[0:N,:, 0] = cell_color[0]
    #     frame[0:N,:, 1] = cell_color[1]
    #     frame[0:N,:, 2] = cell_color[2]

    #     frame[:, -N:, 0] = cell_color[0]
    #     frame[:, -N:, 1] = cell_color[1]
    #     frame[:, -N:, 2] = cell_color[2]

    #     frame[-N:, :, 0] = cell_color[0]
    #     frame[-N:, :, 1] = cell_color[1]
    #     frame[-N:, :, 2] = cell_color[2]

        

    #     return frame