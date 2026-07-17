import cv2 as cv
import numpy as np

# Paths to model files
proto_file = "models/colorization_deploy_v2.prototxt"
model_file = "models/colorization_release_v2.caffemodel"
pts_file = "models/pts_in_hull.npy"

# Load cluster centers
pts_in_hull = np.load(pts_file)

# Load the Caffe model
net = cv.dnn.readNetFromCaffe(proto_file, model_file)

# Add cluster centers as 1x1 convolution kernel
pts = pts_in_hull.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, np.float32)]

# Load grayscale image
img = cv.imread("data/grayscale.png")
img_rgb = (img[:, :, [2, 1, 0]]).astype(np.float32) / 255.0
img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)

# Resize to network input size
img_resized = cv.resize(img_rgb, (224, 224))
img_lab_resized = cv.cvtColor(img_resized, cv.COLOR_RGB2Lab)
l_channel = img_lab_resized[:, :, 0]

# Prepare input
l_channel -= 50  # mean-centering
net.setInput(cv.dnn.blobFromImage(l_channel))

# Predict ab channels
ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize to original image size
ab_channel_us = cv.resize(ab_channel, (img.shape[1], img.shape[0]))

# Combine with original L channel
lab_output = np.concatenate((img_lab[:, :, 0][:, :, np.newaxis], ab_channel_us), axis=2)
img_bgr_out = cv.cvtColor(lab_output, cv.COLOR_Lab2BGR)

# Save and show result
cv.imwrite("data/colorized_output.png", img_bgr_out)
cv.imshow("Colorized", img_bgr_out)
cv.waitKey(0)
cv.destroyAllWindows()
