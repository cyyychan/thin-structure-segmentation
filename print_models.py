from mmseg.apis import init_model, inference_model, show_result_pyplot
import cv2
import os

config_path = 'configs/dinov2_l_multilayers_fcn_convt_4xb4-40k_crack500-512x512.py'
# checkpoint_path = "seg_exp/exp_dinov2_fcn_crack500/work_dirs/dinov2_fcn_4xb4-40k_crack500-512x512/iter_40000.pth"  # None 则只建图不加载权重
checkpoint_path = None

model = init_model(config_path, checkpoint_path, device='cuda:0')
print(model)

# image_path = './road_SKyviQq.jpg'
# image = cv2.imread(image_path)
# result = inference_model(model, image)

# # 绘制分割结果并保存（原图 + 预测叠加）
# os.makedirs('output', exist_ok=True)
# out_file = 'output/seg_result.png'
# # show_result_pyplot 需要 RGB；cv2 读入为 BGR
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# vis_img = show_result_pyplot(
#     model, image_rgb, result,
#     opacity=0.5,
#     show=False,
#     out_file=out_file,
#     draw_gt=False,
#     draw_pred=True,
# )
# print('Segmentation result saved to:', out_file)