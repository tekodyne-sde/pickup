# core/camera_pipeline.py
import depthai as dai
from config import FRAME_WIDTH, FRAME_HEIGHT

def create_v3_pipeline(device: dai.Device) -> tuple:
    """
    Constructs the DepthAI pipeline adhering strictly to v3 architecture 
    and mandatory hardware safety constraints.
    """
    pipeline = dai.Pipeline(device)

    # 1. Unified Camera Nodes (DepthAI v3)
    cam_rgb = pipeline.create(dai.node.Camera)
    cam_rgb.build(boardSocket=dai.CameraBoardSocket.CAM_A)
    
    cam_left = pipeline.create(dai.node.Camera)
    cam_left.build(boardSocket=dai.CameraBoardSocket.CAM_B)
    
    cam_right = pipeline.create(dai.node.Camera)
    cam_right.build(boardSocket=dai.CameraBoardSocket.CAM_C)

    # 2. Stereo Depth Node
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    
    # CRITICAL: Sizing constraint to prevent multiple-of-16 X_LINK_ERROR crashes
    stereo.setOutputSize(FRAME_WIDTH, FRAME_HEIGHT)
    
    # CRITICAL: Hardware filters for clean geometry telemetry
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    # Request the MAXIMUM hardware-supported resolution for the Mono sensors (OV9282). 
    # The StereoDepth node will calculate high-density disparity, and then upscale/align 
    # it to match your 1080p RGB output.
    cam_left.requestOutput((640, 400)).link(stereo.left)
    cam_right.requestOutput((640, 400)).link(stereo.right)

    # Request final host outputs for our capture worker
    rgb_out = cam_rgb.requestOutput((FRAME_WIDTH, FRAME_HEIGHT))
    depth_out = stereo.depth

    return pipeline, rgb_out, depth_out
