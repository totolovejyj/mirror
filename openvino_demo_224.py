import tensorflow as tf
import cv2
import time
import os
import sys
from picamera import PiCamera
import argparse
import math
from posenet import *
from picamera.array import PiRGBArray
from posenet.utils import _process_input
from easydict import EasyDict
from liftoneleg import *
from metric import test_per_frame
# from playsound import playsound
# from posenet.utils import read_cap
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
    
def posenet2openpose(keypoints):
    
    keypoints = keypoints[0] # max score keypoint
    keypoints = np.flip(keypoints, 1) #(y,x) -> (x,y)
    # print("========after flip========")
    # print(keypoints)
    
    # get keypoint for neck
    right_shoulder = keypoints[6]
    left_shoulder = keypoints[5]
    neck_x = right_shoulder[0] + (left_shoulder[0] - right_shoulder[0]) / 2
    neck_y = min(left_shoulder[1], right_shoulder[1])
    neck_top = np.array([0, 0])
    neck_bottom = np.array([neck_x, neck_y])
    keypoints = np.insert(keypoints, 0, neck_top, axis=0)
    keypoints = np.insert(keypoints, 1, neck_bottom, axis=0)
    #print("========after insert neck========")
    #print(keypoints)
    
    # consist keypoints in openpose order
    openpose_order = [0,1,6,8,10,5,7,9,12,14,16,11,13,15]
    res_keypoints = [keypoints[i] for i in openpose_order]
    res_keypoints = np.array(res_keypoints)
    #print("========after reorder========")
    #print(res_keypoints)
    return res_keypoints
    

def main(thres):
    
    #ankle_height = 0
    #counting = 0
    #old_raiseup = False
    #okay = False
    #raiseup = False
    #old_minwrist = 720
    
    #Global varaible
    LEG_LABEL = np.load('./groundtruth.npy')
    threshold = thres
    print(threshold)
    
    #Initialize analzing parameter
    previous_pose_kpts = []
    result = [-1, -1, -1, -1, -1]
    count = 0
    start_frame, end_frame = 1000000, -1
    max_angle = 0
    min_angle = 90
    completed_half = False
    total_len_frame = 0
    
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + '.bin'
    
            
    with tf.Session() as sess:
        #model_cfg, model_outputs = posenet.load_model(101, sess)
        #output_stride = model_cfg['output_stride']
        output_stride = 16
        checkraiseup, rightarm = 0, 720        
        rawCapture = PiRGBArray(camera, size = (640,480))
        
        start = time.time()
        frame_count = 0
        framenum =0
        score_list = []
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
        for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True, splitter_port=1):
            frame_start = time.time()
            pos_temp_data = []
            framenum +=1
            input_image = frame.array
            input_image, display_img, output_scale = _process_input(
                input_image,scale_factor=args.scale_factor, output_stride=output_stride)
            print("display: ", display_img.shape)
            print("preprocess : ", input_image.shape)
            input_image = np.expand_dims(input_image, 0)
            input_image = np.transpose(input_image, (0, 3, 1, 2))
            #print(input_image.shape)
            
            #file_path = ("test%02d.png" %framenum)
            #cv2.imwrite(file_path,display_img)
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
                min_pose_score=0.1)
            
            keypoint_coords *= output_scale   
            
            # convert posenet keypoints 2 openpose format
            openpose_keypoints = posenet2openpose(keypoint_coords)
            
            #Select joints
            select_keypoints = np.concatenate((openpose_keypoints[2],openpose_keypoints[5],openpose_keypoints[8],
                                        openpose_keypoints[10],openpose_keypoints[11],openpose_keypoints[13])).reshape(-1, 2)
                                     
            #Analyze posture
            previous_pose_kpts.append(select_keypoints)
            liftoneleg = LiftOneLeg(previous_pose_kpts)
            angle, leg_status = liftoneleg.check_leg_up_down()
            
            if angle > max_angle:
                max_angle = angle
                max_frame = cv2.imwrite("./blog/static/img/best.png", display_img)
                    
            #Update status and count
            leg_status, completed_half, count_update, start_frame_update, end_frame_update= \
                        liftoneleg.count_repetition(angle, leg_status, completed_half, count, framenum, start_frame, end_frame)
            if (count_update == count +1):
                print("count : %d" %count_update)
                score = test_per_frame(previous_pose_kpts[start_frame-total_len_frame:end_frame-total_len_frame], LEG_LABEL)
                print("**************************")
                print("score : %d" %score)
                score_list.append(score)
                f= open('score.txt', 'w')
                f.write(str(int(score)))
                f.close()
                total_len_frame += len(previous_pose_kpts)
                previous_pose_kpts = []
            
            count, start_frame, end_frame = count_update, start_frame_update, end_frame_update 
            
            f = open('demofile.txt', 'w')
            f.write(str(count))
            f.close()
            
            # write for feedback!!
            if count == 5:
                exercise_time = time.time() - start
                # write max angle
                f = open('max_angle.txt', 'w')
                f.write(str(int(max_angle)))
                f.close() 
                # write exercise time
                f= open('time.txt', 'w')
                f.write(str(int(exercise_time)))
                f.close()
                # write score 
                f= open('final_score.txt', 'w')
                f.write(str(int(sum(score_list)/count)))
                f.close()
                sys.exit(0)
                #return 0

            overlay_image = posenet.draw_skel_and_kp(
                display_img, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.1, min_part_score=0.1)
            
            #cv2.putText(overlay_image, str(counting), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1)
            cv2.imshow('posenet', overlay_image)
            rawCapture.truncate(0)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print('Average FPS: ', (time.time() - frame_start))


if __name__ == "__main__":
    main()
