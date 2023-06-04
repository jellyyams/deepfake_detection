from moviepy.editor import *

def stack_vids(vid1_path, vid2_path, out_name, vid3_path=None, out_fps=30, vertical=False):
    clip1 = VideoFileClip(vid1_path)
    clip2 = VideoFileClip(vid2_path)
    if vid3_path is not None:
        clip3 = VideoFileClip(vid3_path)
        if vertical:
            clips = [[clip1], [clip2], [clip3]]
        else:
            clips = [[clip1, clip2, clip3]]
    else:
        if vertical:
            clips = [[clip1], [clip2]]
        else:
            clips = [[clip1, clip2]]
    final = clips_array(clips)
    final.write_videofile(out_name, fps=out_fps)

