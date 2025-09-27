import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- نمونه داده ها از tensor ---
# probabilities برای کلاس خودرو، shape: [50]
probs = np.array([0.9300, 0.0771, 0.0611, 0.0233, 0.0179,
                  0.0178, 0.0156, 0.0154, 0.0149, 0.0141,
                  0.0134, 0.0113, 0.0104, 0.0094, 0.0092,
                  0.0091, 0.0087, 0.0087, 0.0085, 0.0079,
                  0.0079, 0.0078, 0.0074, 0.0073, 0.0072,
                  0.0071, 0.0071, 0.0068, 0.0064, 0.0062,
                  0.0062, 0.0059, 0.0056, 0.0054, 0.0053,
                  0.0050, 0.0050, 0.0048, 0.0044, 0.0041,
                  0.0041, 0.0040, 0.0040, 0.0038, 0.0032,
                  0.0029, 0.0029, 0.0029, 0.0028, 0.0026])

# inter_coord: [50, 6] => [cx, cy, w2d, h2d, w3d, h3d]، فقط 2D استفاده می‌کنیم
coords = np.array([
    [0.2530, 0.5316, 0.0279, 0.0245, 0.0392, 0.0448],
    [0.3105, 0.5163, 0.0181, 0.0167, 0.0305, 0.0327],
    [0.2520, 0.5320, 0.0269, 0.0253, 0.0377, 0.0408],
    [0.2509, 0.5309, 0.0279, 0.0242, 0.0378, 0.0422],
    [0.2527, 0.5332, 0.0216, 0.0176, 0.0421, 0.0448],
    # ... ادامه 50 خط
])

# --- فقط 5 بالاترین احتمال ---
top_idx = np.argsort(probs)[-5:][::-1]
top_coords = coords[top_idx]
top_probs = probs[top_idx]

# --- بارگذاری تصویر ---
img = plt.imread("1.PNG")  # مسیر تصویر خودت
img_h, img_w = img.shape[:2]

# --- نمایش باکس ها روی تصویر ---
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(img)

for i in range(5):
    cx, cy, w, h = top_coords[i, :4]  # 2D
    # تبدیل نسبت 0-1 به پیکسل
    x = (cx - w/2) * img_w
    y = (cy - h/2) * img_h
    w_pix = w * img_w*4
    h_pix = h * img_h*4
    rect = Rectangle((x, y), w_pix, h_pix,
                     linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y-5, f"{top_probs[i]:.2f}", color='yellow', fontsize=12)

# --- قابلیت زوم با crop ---
# مثلا زوم روی منطقه خاصی
# ax.set_xlim(left_px, right_px)
# ax.set_ylim(bottom_px, top_px)

plt.show()
