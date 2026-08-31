
def calculate_brightness(img):
	# Write your code here
	import numpy as np

	try:
		img = np.array(img)
	except:
		return -1
	if np.all([(img>=0)& (img<=255)]) and img.size != 0:
		return np.mean(img)

	else:
		return -1 