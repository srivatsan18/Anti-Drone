from core import Core
import cv2


c = Core()

cap=cv2.VideoCapture('/home/shinjinee/Documents/Python Programs/Antidrone/WhatsApp Video 2020-04-29 at 00.47.07.mp4')

for i in range (630):
	_ , frame = cap.read()

while (True):
	
	_ , frame = cap.read()
	
	drawing_image = c.get_drawing_image(frame)

	processed_image, scale = c.pre_process_image(frame)

	c.set_model(c.get_model())
	boxes, scores, labels = c.predict_with_graph_loaded_model(processed_image, scale)
	
	detections = c.draw_boxes_in_image(drawing_image, boxes, scores)
	print (detections)	
	
	drawing_image = cv2.cvtColor(drawing_image, cv2.COLOR_RGB2BGR)
	cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
	cv2.imshow('frame', drawing_image)

	if cv2.waitKey(10) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

