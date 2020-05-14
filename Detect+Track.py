import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float, default=0.18, help='Score threshold for displaying bounding boxes')
parser.add_argument('-fps', '--fps', dest='fps', type=int, default=1, help='Show FPS on detection/display visualization')
parser.add_argument('-src', '--source', dest='video_source', default=0, help='Device index of the camera.')
parser.add_argument('-wd', '--width', dest='width', type=int, default=320, help='Width of the frames in the video stream.')
parser.add_argument('-ht', '--height', dest='height', type=int, default=180, help='Height of the frames in the video stream.')
parser.add_argument('-ds', '--display', dest='display', type=int, default=1, help='Display the detected images using OpenCV. This reduces FPS')
parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int, default=4, help='Number of workers.')
parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int, default=5, help='Size of the queue.')
args = parser.parse_args()


# load frozen tensorflow model into memory
def load_inference_graph():
	print("> ====== loading frozen graph into memory")
	detection_graph = tf.Graph()

	with detection_graph.as_default():

		od_graph_def = tf.compat.v1.GraphDef()

		with tf.io.gfile.GFile('/home/shinjinee/Documents/Python Programs/Antidrone/drone-detection/Trained-Model/drone-detection.pb', 'rb') as fid:

			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

		sess = tf.compat.v1.Session(graph=detection_graph)

	print(">  ====== Inference graph loaded.")
	return detection_graph, sess
	
	
def detect_objects(image_np, detection_graph, sess):
    
    image_tensor = detection_graph.get_tensor_by_name('input_1:0')
    detection_boxes = detection_graph.get_tensor_by_name( 'filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0')
    detection_scores = detection_graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0')
    
    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores) = sess.run([detection_boxes, detection_scores], feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)
    
def draw_box_on_image(image_np, boxes):
	(left, bottom, right, top) = (boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3])
	p1 = (int(left), int(top))
	p2 = (int(right), int(bottom))
	cv2.rectangle(image_np, p1, p2, (255, 0, 0), 1, cv2.LINE_AA)
		
def track_drone(image_np, boxes):
	tracker = cv2.TrackerKCF_create()
	ok = tracker.init(image_np, (boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]))
	
	while ok:
		timer = cv2.getTickCount()
		
		(left, bottom, right, top) = (boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3])
		p1 = (int(left), int(top))
		p2 = (int(right), int(bottom))
		cv2.rectangle(image_np, p1, p2, (255, 0, 0), 1, cv2.LINE_AA)
		#draw_box_on_image(image_np, boxes):
		
		fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
		
		cv2.putText(image_np, "Status: Tracking", (2,15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,255), 1, cv2.LINE_AA)
		cv2.putText(image_np, "FPS: "+str(int(fps)), (2,30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,255), 1, cv2.LINE_AA)
		
		cv2.imshow('drone', image_np)
		
		ok, image_np = cap.read()
		ok, boxes[0] = tracker.update(image_np)
		
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			cap.release()
			sys.exit(0)
    
detection_graph, sess=load_inference_graph()

cap = cv2.VideoCapture('/home/shinjinee/Documents/Python Programs/Antidrone/WhatsApp Video 2020-04-29 at 00.47.07_low.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

im_width, im_height = (cap.get(3), cap.get(4))

cv2.namedWindow('drone', cv2.WINDOW_NORMAL)

while True:
		
	ret, image_np = cap.read()
	h,w,c = image_np.shape
	
	cv2.putText(image_np, "Status: Lost", (2,15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,255), 1, cv2.LINE_AA)
	cv2.imshow('drone', image_np)
	
	boxes, scores = detect_objects(image_np, detection_graph, sess)
	print (scores[0])
	print (boxes[0])
	
	if (scores[0] > args.score_thresh):
		track_drone(image_np, boxes)
	
	if cv2.waitKey(25) & 0xFF == ord('q'):
				break
            	
cv2.destroyAllWindows()
cap.release()
