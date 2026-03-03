from mmseg.apis import init_model, inference_model, show_result_pyplot
import cv2
import os

config_path = 'configs/vmamba-tiny_upernet_4xb4-40k_crack500-512x512.py'
checkpoint_path = "./checkpoints/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth"

model = init_model(config_path, checkpoint_path, device='cuda:0')
print(model)

image_path = '/dataset/siyuanchen/research/mmsegmentation/demo/demo.png'
image = cv2.imread(image_path)
result = inference_model(model, image)

# 绘制分割结果并保存（原图 + 预测叠加）
os.makedirs('output', exist_ok=True)
out_file = 'output/seg_result.png'
# show_result_pyplot 需要 RGB；cv2 读入为 BGR
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
vis_img = show_result_pyplot(
    model, image_rgb, result,
    opacity=0.5,
    show=False,
    out_file=out_file,
    draw_gt=False,
    draw_pred=True,
)
print('Segmentation result saved to:', out_file)