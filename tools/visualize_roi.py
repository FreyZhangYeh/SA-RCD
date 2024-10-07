import numpy as np
import cv2
import data_utils

def draw_boxes_on_rgb_image(rgb_img, selected_points, box_width=228, color=(0, 165, 255), alpha=0.5):
    """
    在 RGB 图像上为选定的点绘制半透明橙色框。
    
    参数:
    - rgb_img: h, w, 3 的 RGB 图像 (numpy 数组)
    - selected_points: 随机选择的非零深度点的坐标列表
    - box_width: 框的宽度，默认为228
    - color: 框的颜色 (默认为橙色, BGR 格式)
    - alpha: 透明度 (0 表示完全透明，1 表示完全不透明)
    
    返回:
    - 带有框的 RGB 图像
    """
    h, w, _ = rgb_img.shape
    overlay = rgb_img.copy()
    
    # 遍历所有的选定点
    for y, x in selected_points:
        # 计算框的左右边界
        left = max(int(x - box_width // 2), 0)
        right = min(int(x + box_width // 2), w)
        
        # 在框的区域内填充颜色 (橙色 BGR: 0, 165, 255)
        overlay[:, left:right] = color
    
    # 使用 cv2.addWeighted 实现透明效果
    result_img = cv2.addWeighted(overlay, alpha, rgb_img, 1 - alpha, 0)
    
    return result_img

def randomly_select_nonzero_points(dr, num_points=6):
    """
    从给定的dr中在指定范围 [1/4w:3/4w, 1/4h:3/4h] 内选择6个非零点。
    
    参数:
    - dr: h,w 的雷达深度图 (numpy 数组)
    - num_points: 要选择的点的数量，默认为6
    
    返回:
    - 选中的非零深度点的坐标列表
    """
    h, w = dr.shape
    # 定义选择点的范围
    x_min, x_max = w // 4, 3 * w // 4
    y_min, y_max = h // 4, 3 * h // 4
    
    # 使用两次掩码分别过滤 y 和 x 的范围
    y_indices, x_indices = np.where(dr > 0)
    valid_indices = np.where((x_indices >= x_min) & (x_indices <= x_max) &
                             (y_indices >= y_min) & (y_indices <= y_max))
    
    non_zero_points = np.vstack((y_indices[valid_indices], x_indices[valid_indices])).T

    # 如果非零点的数量不足6个，调整数量
    num_points = min(num_points, len(non_zero_points))
    
    # 随机选择 num_points 个非零点
    selected_points = non_zero_points[np.random.choice(non_zero_points.shape[0], num_points, replace=False)]
    
    return selected_points


if __name__ == '__main__':


    rgb_path= "/data/zfy_data/nuscenes/nuscenes_origin/samples/CAM_FRONT/n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295368012404.jpg"
    dr_path = "/data/zfy_data/nuscenes/nuscenes_derived_test/radar_image/scene_99/CAM_FRONT/n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295368012404.png"
    
    rgb_img = data_utils.load_image(rgb_path)
    dr = data_utils.load_depth(dr_path)

    selected_points = randomly_select_nonzero_points(dr, num_points=6)
    result_img = draw_boxes_on_rgb_image(rgb_img, selected_points, box_width=228, alpha=0.5)

    # 将结果保存为图像文件
    cv2.imwrite('/data/zfy_data/nuscenes/virl_for_paper/roi/roi_of_radarnet.png', result_img)
