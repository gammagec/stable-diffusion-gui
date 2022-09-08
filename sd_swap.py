import cv2
import numpy as numpy
import dlib
import mediapipe as mp
from scripts.run import run_img, load_model
from src.common.namespace import Namespace
from resizeimage import resizeimage
from PIL import Image
import matplotlib.image
import os, argparse
import yaml

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--config",
		type=str,
		nargs="?",
		default="swap_config.yaml",
		help="the config file to use"
	)

	opt = parser.parse_args()

	config_path = os.path.join(".", opt.config)
	if not os.path.exists(config_path):
		print("No swap_config.yaml file found")
		exit()

	config_file = open(config_path)		
	config = yaml.load(config_file, Loader=yaml.UnsafeLoader)			

	out_path = config.out_dir
	cap = cv2.VideoCapture(config.video_in)

	frame_size = (
		int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
		int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		)
	fps = config.fps
	mode = config.mode
	write_pics = config.write_pics
	frame_skip = config.frame_skip
	show_preview = config.show_preview
	half = config.half
	skip_frames = config.skip_frames
	stop_after = config.stop_after
	include_originals = config.include_originals
	expand_up = config.expand_up
	expand = config.expand
	all_faces = config.all_faces
	write_pics_no_face_frames = config.write_pics_no_face_frames

	if not write_pics:
		print(f"creating video with size {frame_size} and frame rate {fps}")	
		out_vid = cv2.VideoWriter(f'{out_path}/out.mp4',
			cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

	model = load_model(config.config, config.ckpt_loc, config.embeddings, half = half)
	print("model loaded")

	class CascadeFaceDetect:
		def __init__(self):
			self.faceCascade = cv2.CascadeClassifier(config.face_cascade)

		def get_faces(self, frame):		
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			faces = self.faceCascade.detectMultiScale(
				gray, scaleFactor = 1.1,
				minNeighbors = 20,
				flags = cv2.CASCADE_SCALE_IMAGE
			)
			return faces

	class DeepCaffeFaceDetect:
		def __init__(self):
			self.dnn_model = cv2.dnn.readNetFromCaffe(
				prototxt=config.deep_caffe_prototxt, 
				caffeModel=config.deep_caffe_model)
			self.min_confidence = config.deep_caffe_confidence

		def get_faces(self, frame):
			image_height, image_width, _ = frame.shape
			preprocessed_image = cv2.dnn.blobFromImage(
				frame, scalefactor = 1.0, size = (300, 300),
				mean = (104.0, 117.0, 123.0), swapRB = False, crop = False)
			self.dnn_model.setInput(preprocessed_image)

			results = self.dnn_model.forward()
			faces = []
			for face in results[0][0]:
				face_confidence = face[2]
				if face_confidence > self.min_confidence:
					bbox = face[3:]
					x1 = int(bbox[0] * image_width)
					y1 = int(bbox[1] * image_height)
					x2 = int(bbox[2] * image_width)
					y2 = int(bbox[3] * image_height)
					faces.append((x1, y1, x2 - x1, y2 - y1))
				
			return faces

	class DlibFaceDetect:
		def __init__(self):
			self.detector = dlib.cnn_face_detection_model_v1(
				config.dlib_detector)
			self.new_width = 600 # scales the input image to this widget

		def get_faces(self, frame):
			height, width, _ = frame.shape
			new_width = self.new_width
			new_height = int((self.new_width / width) * height)
			resized_image = cv2.resize(frame.copy(), (new_width, new_height))
			imgRGB = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

			results = self.detector(imgRGB, 0)
			faces = []
			for face in results:        
				# Retriece the bounding box of the face.
				bbox = face.rect
			
				# Retrieve the bounding box coordinates and scale them according to the size of original input image.
				x1 = int(bbox.left() * (width/new_width))
				y1 = int(bbox.top() * (height/new_height))
				x2 = int(bbox.right() * (width/new_width))
				y2 = int(bbox.bottom() * (height/new_height))
				faces.append((x1, y1, x2 - x1, y2 - y1))

			return faces

	class MediaPipeFaceDetect:
		def __init__(self):
			mp_face_detection = mp.solutions.face_detection
			self.detector = mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence=0.2)

		def get_faces(self, frame):
			image_height, image_width, _ = frame.shape
			imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			results = self.detector.process(imgRGB)

			faces = []
			if results.detections: 
				# Iterate over the found faces.
				for face_no, face in enumerate(results.detections):			
					face_bbox = face.location_data.relative_bounding_box
					
					x1 = int(face_bbox.xmin * image_width)
					y1 = int(face_bbox.ymin * image_height)
					w = int(face_bbox.width * image_width)
					h = int(face_bbox.height * image_height)					

					faces.append((x1, y1, w, h))
			print(f'got {len(faces)} faces')
			return faces

	count = 0
	num = 0

	faceDetect = None
	if mode == "cascade":
		faceDetect = CascadeFaceDetect()	
	elif mode == "deep":
		faceDetect = DeepCaffeFaceDetect()
	elif mode == "dlib":
		faceDetect = DlibFaceDetect()	
	elif mode == "mediapipe":
		faceDetect = MediaPipeFaceDetect()

	while (cap.isOpened()):	
		# Capture frame-by-frame	
		ret, frame = cap.read()		
		if not ret:
			break
		
		if count < skip_frames or count % frame_skip > 0:
			count += 1
			continue

		if stop_after > 0:
			print(f"processing {num} of {stop_after}")

		print(f'processing frame {count}')

		if show_preview:
			cv2.imshow('Original', frame)

		faces = faceDetect.get_faces(frame)

		original_image = frame.copy()

		count += 1		
		
		if len(faces) > 0:
			num += 1
			original_pil = Image.fromarray(original_image)
			for face in faces:		
				(x, y, w, h) = face
				x -= expand
				y -= expand_up
				w += expand * 2
				h += expand_up + expand
				box = (x, y, x + w, y + h)
				im_pil = Image.fromarray(frame).convert("RGB").crop(box = box)
				im_pil_src = im_pil.copy()			
				(width, height) = im_pil.size
				new_width = 512
				new_height = int((new_width / width) * height)
				im_pil = im_pil.resize((new_width, new_height))
				im_pil = resizeimage.resize_cover(im_pil, [512, 512])
				print(f'size is {im_pil.size}')
				im_np = numpy.asarray(im_pil)		
				if show_preview:
					cv2.imshow('Face', im_np)
				output_image = run_img(model, im_pil, config.prompt, 
					steps = config.steps, scale = config.scale, strength = config.strength, half = half)			
				output_back = resizeimage.resize_crop(
					resizeimage.resize_width(output_image, w), 
					[w, h])
				original_pil.paste(output_back, (x, y))
				if include_originals:
					original_pil.paste(im_pil_src, (x + w, y))

			original_np = numpy.asarray(original_pil)
			if show_preview:
				cv2.imshow('SD Merged', original_np)
			if write_pics:
				# write pic			
				original_np = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)			
				matplotlib.image.imsave(f'{out_path}/out_{count}.png', original_np)						
			else:
				out_vid.write(original_np)		
		else:		
			if not write_pics:					
				out_vid.write(frame)
			elif write_pics_no_face_frames:
				original_np = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)			
				matplotlib.image.imsave(f'{out_path}/out_{count}.png', original_np)										


		if stop_after > 0 and num > stop_after:
			break

		# define q as the exit button
		# only with 'show_preview' enabled
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	 
	# release the video capture object
	cap.release()
	if not write_pics:					
		out_vid.release()

	# Closes all the windows currently opened.
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
