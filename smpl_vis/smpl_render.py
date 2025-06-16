import cv2
import torch
from einops import einsum, rearrange
from utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay, create_camera_sensor, get_video_lwh, get_writer
from utils.smpl_utils import make_smplx
from tqdm import tqdm
import os
import argparse
import numpy as np
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)

from utils.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points

CRF = 23 

def render(pred_ay_verts, filename):
    J_regressor = torch.load("./body_model/smpl_neutral_J_regressor.pt").cuda()
    faces_smpl = make_smplx("smpl").faces

    def move_to_start_point_face_z(verts):
        "XZ to origin, Start from the ground, Face-Z"
        # position
        verts = verts.clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        # face direction
        T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")  # (L, J, 3)
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=15,    # 20
        target_center_height=1,   # 1.0
        vec_rot=1
    )
    
    # length, width, height = get_video_lwh(video_path)
    length, width, height = (395, 1280, 720)
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    # renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    debug_cam = False

    render_length = length if not debug_cam else 8
    # -- rendering code -- #
    video_path = "./video_output/"
    os.makedirs(video_path, exist_ok=True)
    writer = get_writer(video_path + filename + '_render.mp4', fps=30, crf=CRF)

    for i in tqdm(range(render_length), desc=f"Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
        writer.write_frame(img)
    writer.close()

def get_verts(motion_file, device):
    smpl_parser_n = SMPL_Parser(model_path="../smpl_retarget/smpl_model/smpl", gender="neutral")

    framerate = motion_file['mocap_framerate']

    root_trans = motion_file['trans']
    pose_aa = np.concatenate([motion_file['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
    betas = motion_file['betas']

    skip = int(framerate // 30)
    trans = torch.from_numpy(root_trans[::skip])
    pose_aa_walk = torch.from_numpy(pose_aa[::skip]).float()

    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, torch.from_numpy(betas).unsqueeze(0), trans)
    return verts.to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True)
    args = parser.parse_args()
    filepath = args.filepath

    device = 'cuda:0'

    filename = filepath.split('/')[-1]
    assert filename.split('.')[-1] == 'npz', "must npz file"

    motion_file = np.load(filepath, allow_pickle = True)
    verts = get_verts(motion_file, device)
    verts[:,:,1], verts[:,:,2] = verts[:,:,2], verts[:,:,1]
    render(verts, filename.split('.')[0])

