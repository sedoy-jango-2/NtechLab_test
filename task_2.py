import sys
import json
import cv2, glob
import numpy as np

from keras.models import Sequential, Model, load_model

if __name__ == "__main__":
	sizeW = 80
	sizeH = 110
	
	images = glob.glob(str(sys.argv[1]) + "*.jpg")
	
	data_img = [cv2.imread(img, 0) for img in images]
	for i in range(0, len(data_img)):
		data_img[i] = cv2.resize(data_img[i],(sizeW, sizeH))
	data = np.asarray(data_img)
	
	data = data.reshape(data.shape[0], sizeW, sizeH, 1)
	
	for data_point in data:
		data_point = data_point / 255
	
	model = load_model('model.h5')
	
	result = model.predict(data)
	
	data_to_json = {}
	gender = {0:'female', 1:'male'}
	
	for i in range(len(result)):
		data_to_json[images[i]] = gender[np.argmax(result[i], axis=0)]
	
	with open("process_results.json", "w") as write_file:
		json.dump(data_to_json, write_file)
		