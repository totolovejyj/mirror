import tensorflow as tf
import cv2
import time
import os
from picamera import PiCamera
import argparse
import math
from posenet import *
from picamera.array import PiRGBArray
from posenet.utils import _process_input
from easydict import EasyDict

# from playsound import playsound
#from posenet.utils import read_cap
from openvino.inference_engine import IENetwork, IECore

#def parse_args():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--model', type=str, default='./posenet_input_337_449/model-mobilenet_v1_101.xml')
#    parser.add_argument('--cam_id', type=int, default=0)
#    parser.add_argument('--cam_width', type=int, default= 640)
#    parser.add_argument('--cam_height', type=int, default=480)
#    parser.add_argument('--scale_factor', type=float, default=0.7125)
#    parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
#    parser.add_argument("-l", "--cpu_extension",
#                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
#                           "the kernels implementations.", type=str, default=None)
#    parser.add_argument("-d", "--device",
#                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
#                           " acceptable. The sample will look for a suitable plugin for device specified. "
#                           "Default value is CPU", default="MYRIAD", type=str)
#    args = parser.parse_args()
#    return args

args = EasyDict()
args.model = './posenet_input_337_449/model-mobilenet_v1_101.xml'
args.device = 'MYRIAD'
args.scale_factor = 0.7125
args.cpu_extension = None
print(args)
        
camera = PiCamera()
camera.resolution = (640,480)
#camera.framerate =32

def counting_rightarm(keypoint_coords, old_raiseup):
    Count = False
    raiseup = old_raiseup
    if keypoint_coords[0, :, :][10][0] < keypoint_coords[0, :, :][12][0] + 20:
        ready = "ing"
        shoulder_min = keypoint_coords[0, :, :][6][0] - 15
        shoulder_max = keypoint_coords[0, :, :][6][0] + 15

        if shoulder_min < keypoint_coords[0, :, :][10][0] < shoulder_max:
            raiseup = True
    hip_min = keypoint_coords[0, :, :][12][0] - 15
    hip_max = keypoint_coords[0, :, :][12][0] + 15
    if old_raiseup == True and hip_min < keypoint_coords[0, :, :][10][0] < hip_max:
        Count = True
        raiseup = False
    return Count, raiseup


def checking_rightarm(keypoint_coords, old_raiseup, old_rightarm):
    Check = False
    raiseup = old_raiseup
    max_rightarm = min(old_rightarm, keypoint_coords[0, :, :][10][0])
    hip = keypoint_coords[0, :, :][12][0]
    shoulder_min = keypoint_coords[0, :, :][6][0] + 30
    shoulder_max = keypoint_coords[0, :, :][6][0] + 15
    if keypoint_coords[0, :, :][10][0] < hip + 20:
        if shoulder_max < max_rightarm < shoulder_min:
            raiseup = True
    hip_min = hip - 15
    hip_max = hip + 15
    if raiseup == True and hip_min < keypoint_coords[0, :, :][10][0] < hip_max:
        if shoulder_max< max_rightarm < shoulder_min:
            Check = True
            raiseup = False
            max_rightarm = 1000
    return Check, raiseup, max_rightarm

def main():
    
    ankle_height = 0
    counting = 0
    old_raiseup = False
    okay = False
    raiseup = False
    old_minwrist = 720
    
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + '.bin'
    
    with tf.Session() as sess:
        #model_cfg, model_outputs = posenet.load_model(101, sess)
        #output_stride = model_cfg['output_stride']
        output_stride = 16
        checkraiseup, rightarm = 0, 720        
        rawCapture = PiRGBArray(camera, size = (640,480))

        #start = time.time()
        frame_count = 0
        framenum =0
        ie = IECore()
        if args.cpu_extension and 'CPU' in args.device:
            ie.add_extension(args.cpu_extension, "CPU")
        # Read IR
        ie.set_config({'VPU_HW_STAGES_OPTIMIZATION':'NO'}, "MYRIAD")
        net = IENetwork(model=model_xml, weights=model_bin)
        
        n, c, w, h = net.inputs['image'].shape
        #print("width, height: ", w, h) #337, 513
        net.batch_size = n
        exec_net = ie.load_network(network=net, device_name=args.device,num_requests=2)
        del net
        #cap = cv2.VideoCapture(r"./ligt_oneleg_correct.mp4")
        
        #while True:
        for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True, splitter_port=2):
            pos_temp_data = []
            sum = 0
            framenum +=1
            input_image = frame.array
            #print(input_image.shape)
            #input_image = cv2.resize(input_image, (720, 480))
            #if framenum == 3:
            #    exit()
            input_image, display_img, output_scale = _process_input(
                input_image,scale_factor=args.scale_factor, output_stride=output_stride)
            
            #input_image, display_img, output_scale = read_cap(
            #    cap, scale_factor=args.scale_factor, output_stride=output_stride)
            #print(input_image.shape)
            #input_image = np.swapaxes(input_image, 0, 1)
            #input_image = np.swapaxes(input_image, 0, 2)
            input_image = np.expand_dims(input_image, 0)
            input_image = np.transpose(input_image, (0, 3, 1, 2))
            #print(input_image.shape)
            
            #frame, display_img, output_scale = _process_input(input_image)
            #frame = cv2.resize(input_image, dsize=(w, h))
            #frame = input_image.reshape((n, c, h, w))
            
            res = exec_net.infer({'image': input_image})
            heatmaps_result = res['heatmap']
            offsets_result = res['offset_2/Add']
            displacement_fwd_result = res["displacement_fwd_2/Add"]
            displacement_bwd_result = res["displacement_bwd_2/Add"]
            #print("Heatmap: ", heatmaps_result.shape)
            
            heatmaps_result = np.transpose(heatmaps_result, (0, 2, 3, 1))
            offsets_result = np.transpose(offsets_result, (0, 2, 3, 1))
            displacement_fwd_result = np.transpose(displacement_fwd_result, (0, 2, 3, 1))
            displacement_bwd_result = np.transpose(displacement_bwd_result, (0, 2, 3, 1))

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.001)
            
            #(keypoint_coords)
            keypoint_coords *= output_scale        
                        
            Count, raiseup = counting_rightarm(keypoint_coords, raiseup)

            rightwrist = keypoint_coords[0, :, :][10][0]
            minwrist = min(rightwrist, old_minwrist)
            shoulder_min = keypoint_coords[0, :, :][6][0] + 30
            shoulder_max = keypoint_coords[0, :, :][6][0] + 15
            hip_min =  keypoint_coords[0, :, :][12][0] - 15
            hip_max =  keypoint_coords[0, :, :][12][0] + 15
        
            
            if Count:
                counting +=1
                print("================================")
                print(counting)
                minwrist = 720
                f = open('demofile.txt', 'w')
                f.write(str(counting))
                f.close()
                #time.sleep(3)
            old_minwrist = minwrist

            overlay_image = posenet.draw_skel_and_kp(
                display_img, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.1, min_part_score=0.1)
            
            #cv2.putText(overlay_image, str(counting), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1)
            cv2.imshow('posenet', overlay_image)
            rawCapture.truncate(0)
            #rawCapture.seek(0)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        
#         print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()
