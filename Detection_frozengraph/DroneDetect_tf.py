import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float, default=0.2, help='Score threshold for displaying bounding boxes')
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

	#for op in detection_graph.get_operations():
	#	print(op.values())
		
	#for op in detection_graph.get_operations():
	#	print(str(op.name))
		
	#for n in  detection_graph.as_graph_def().node:
	#	print(n.name)
	return detection_graph, sess
	
	
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
	# Each box represents a part of the image where a particular object was detected.
	# Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    
    image_tensor = detection_graph.get_tensor_by_name('input_1:0')
    detection_boxes = detection_graph.get_tensor_by_name( 'filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0')
    detection_scores = detection_graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0')
    #detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    #num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores) = sess.run([detection_boxes, detection_scores], feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)

def draw_box_on_image(score_thresh, scores, boxes, im_width, im_height, image_np):
	if (scores[0] > score_thresh):
		(left, bottom, right, top) = (boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3])
		p1 = (int(left), int(top))
		p2 = (int(right), int(bottom))
		cv2.rectangle(image_np, p1, p2, (255, 0, 0), 3, cv2.LINE_AA)
    
detection_graph, sess=load_inference_graph()

cap = cv2.VideoCapture('/home/shinjinee/Documents/Python Programs/Antidrone/WhatsApp Video 2020-04-29 at 00.47.07_low.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

im_width, im_height = (cap.get(3), cap.get(4))

cv2.namedWindow('drone', cv2.WINDOW_NORMAL)

while True:
		
	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	ret, image_np = cap.read()
	#image_np=cv2.flip(image_np, 1)
	h,w,c = image_np.shape
	try:
		image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
	except:
		print("Error converting to RGB")
	
	boxes, scores = detect_objects(image_np, detection_graph, sess)
	print (scores[0])
	print (boxes[0])
	draw_box_on_image(args.score_thresh, scores, boxes, im_width, im_height, image_np)
		
	cv2.imshow('drone', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
			
	if cv2.waitKey(25) & 0xFF == ord('q'):
				break
            	
cv2.destroyAllWindows()
cap.release()
