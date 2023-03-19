import cv2
import numpy as np

# 課題1-①：画像を読み込み．
img = cv2.imread('./images/milkdrop.bmp')
cv2.imshow('img', img)

# 画像の左端をトリミング．
h, w = img.shape[:2]
x, y = 15, 0
trimmed_img = img[y:y+h, x:x+w]

# 画像をグレースケール化．
grayscaled_img = cv2.cvtColor(trimmed_img, cv2.COLOR_BGR2GRAY)

# 画像を平滑化．
smoothed_image = cv2.medianBlur(grayscaled_img, ksize=7)

# 課題1-②：画像を2値化．
_, binary_img = cv2.threshold(
    smoothed_image, 136, 255, cv2.THRESH_BINARY)

# 課題1-③：2値化画像の輪郭を抽出．
contours, _ = cv2.findContours(
    binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 輪郭内面積が30未満の輪郭を除去．
fixed_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area >= 30:
        fixed_contours.append(contour)

# 元画像と同じ画像サイズのゼロ行列（黒塗り画像）を生成．
mask = np.zeros(img.shape[:2], np.uint8)

# 課題1-④：輪郭の内部を色塗り．
cv2.drawContours(mask, fixed_contours, -1, 255, cv2.FILLED)

# トリミングした分を補正（アフィン変換で平行移動）．
x_shift = 15
shift_matrix = np.float32([[1, 0, x_shift], [0, 1, 0]])
mask = cv2.warpAffine(mask, shift_matrix, mask.shape[:2])
trimmed_img = cv2.warpAffine(trimmed_img, shift_matrix, mask.shape[:2])

# 課題1-⑤：ミルククラウン領域のみを表示．
masked_img = cv2.bitwise_and(trimmed_img, trimmed_img, mask=mask)
cv2.imshow('masked_img', masked_img)
cv2.waitKey(0)
