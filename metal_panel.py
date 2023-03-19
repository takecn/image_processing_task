import cv2
import numpy as np

# 画像を読み込み．
img = cv2.imread('./images/metal_panel.jpg')

# 画像をトリミング（トリム画像毎に2値化するため．その後合成してマスク処理する．）．
h, w = img.shape[:2]

x, y = 1930, 1370
trimmed_img2 = img[y:y+150, x:x+150]

x, y = 4680, 1020
trimmed_img3 = img[y:y+350, x:w]

x, y = 2830, 2100
trimmed_img4 = img[y:h, x:w]

x, y = 4830, 1070
trimmed_img5 = img[y:y+40, x:w]

x, y = 4500, 1110
trimmed_img6 = img[y:y+200, x:w]

# 各画像をグレースケール化．cv2.cvtColor(画像データ，色変換方法)
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayscaled_img2 = cv2.cvtColor(trimmed_img2, cv2.COLOR_BGR2GRAY)
grayscaled_img3 = cv2.cvtColor(trimmed_img3, cv2.COLOR_BGR2GRAY)
grayscaled_img4 = cv2.cvtColor(trimmed_img4, cv2.COLOR_BGR2GRAY)
grayscaled_img5 = cv2.cvtColor(trimmed_img5, cv2.COLOR_BGR2GRAY)
grayscaled_img6 = cv2.cvtColor(trimmed_img6, cv2.COLOR_BGR2GRAY)

# 各画像を平滑化．
smoothed_image2 = cv2.medianBlur(grayscaled_img2, ksize=5)
smoothed_image4 = cv2.medianBlur(grayscaled_img4, ksize=5)

# 各画像を2値化．
_, binary_img = cv2.threshold(
    grayscaled_img, 50, 255, cv2.THRESH_BINARY)

_, binary_img2 = cv2.threshold(
    smoothed_image2, 200, 255, cv2.THRESH_BINARY)

_, binary_img3 = cv2.threshold(
    grayscaled_img3, 45, 255, cv2.THRESH_BINARY)

_, binary_img4 = cv2.threshold(
    smoothed_image4, 250, 255, cv2.THRESH_BINARY)

_, binary_img5 = cv2.threshold(
    grayscaled_img5, 0, 255, cv2.THRESH_BINARY)

_, binary_img6 = cv2.threshold(
    grayscaled_img6, 0, 255, cv2.THRESH_BINARY)

# 各2値画像をオープニング．
kernel = np.ones((27, 27), np.uint8)
erosion_img = cv2.erode(binary_img, kernel, iterations=2)
opening_img = cv2.dilate(erosion_img, kernel, iterations=2)

kernel = np.ones((3, 3), np.uint8)
erosion_img = cv2.erode(binary_img3, kernel, iterations=1)
opening_img3 = cv2.dilate(erosion_img, kernel, iterations=1)

kernel = np.ones((4, 4), np.uint8)
erosion_img = cv2.erode(binary_img4, kernel, iterations=2)
opening_img4 = cv2.dilate(erosion_img, kernel, iterations=2)

# 元画像と同じ画像サイズのゼロ行列（黒塗り画像）を生成し，各2値画像を合成．
x, y = 1930, 1010
h, w = img.shape[:2]
trimmed_img = opening_img[y:h, x:w]
mask = np.zeros(img.shape[:2], np.uint8)
mask[y:y+h, x:x+w] = trimmed_img

x, y = 1930, 1370
h, w = binary_img2.shape[:2]
mask[y:y+h, x:x+w] = binary_img2

x, y = 4680, 1020
h, w = opening_img3.shape[:2]
mask[y:y+h, x:x+w] = opening_img3

x, y = 2830, 2100
h, w = opening_img4.shape[:2]
mask[y:y+h, x:x+w] = opening_img4

x, y = 4830, 1070
h, w = binary_img5.shape[:2]
mask[y:y+h, x:x+w] = binary_img5

x, y = 4500, 1110
h, w = binary_img6.shape[:2]
mask[y:y+h, x:x+w] = binary_img6

# 合成画像の輪郭を抽出．
contours, _ = cv2.findContours(
    mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 合成画像の輪郭内部を塗り潰し．
cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)

# 元画像をマスク処理し，金属パネル領域のみを表示．
masked_img = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('masked_img', masked_img)
cv2.waitKey(0)
