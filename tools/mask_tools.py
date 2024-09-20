import numpy as np
import cv2
            
def mask2poly(mask):
    """
    mask to polygon.
    
    Parameters:
        mask (np.array) -- gray binary image
    
    Return:
        approx_contours (list of np.array) -- polygons
    
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    approx_contours = []
    for cnt in contours:
        cnt = cnt.astype(np.int32)
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx_contours.append(approx)
    return approx_contours


def poly2mask(img_size, contour_pnts):
    """ get mask from polygon
    
    Parameters:
        img_size (list or tuple) -- (width, height) the region size
        contour_pnts (list or tuple) -- [[[x1, y1], [x2, y2], ...]] contour points
    
    Return:
        masks (np.array) -- total mask in one array
    
    """
    if contour_pnts is not None:
        masks = np.zeros([img_size[1],img_size[0]], dtype=np.uint8)
        for i in range(len(contour_pnts)):
            mask_pts = np.array(contour_pnts[i]).reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(masks, [mask_pts], 1)
    else:
        masks = np.zeros([img_size[1],img_size[0]], dtype=np.int8)
    
    return masks


def find_zero_to_one_transition_per_row(mask):
    first_transition_points = []
    for y in range(mask.shape[0]):
        transition_found = False
        for x in range(1, mask.shape[1]):  # Iterate from left to right
            if mask[y, x-1] == 0 and mask[y, x] == 1:
                first_transition_points.append((x, y))  # x is the transition point
                transition_found = True
                break  # Stop the loop after finding the first transition point in the row
        if not transition_found:
            # If there is no 0 to 1 transition, use None to indicate this
            # first_transition_points.append((None, y))
            pass  # Alternatively, ignore rows without any transitions

    return first_transition_points


def find_one_to_zero_transition_per_row(mask):
    first_transition_points = []
    for y in range(mask.shape[0]):
        transition_found = False
        for x in range(1, mask.shape[1]):  # Iterate from left to right
            if mask[y, x-1] == 1 and mask[y, x] == 0:
                first_transition_points.append((x-1, y))  # x-1 is the transition point
                transition_found = True
                break  # Stop the loop after finding the first transition point in the row
        if not transition_found:
            # If there is no transition from 1 to 0, use None to indicate this
            # first_transition_points.append((None, y))
            pass  # Alternatively, ignore rows without any transitions

    return first_transition_points


def find_bounding_box(mask):
    # 마스크에서 True 값을 가진 행과 열의 인덱스를 찾습니다.
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # True 값이 시작되고 끝나는 행과 열의 인덱스를 찾습니다.
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # 찾아낸 인덱스를 통해 bounding box 반환
    return (rmin, rmax, cmin, cmax)


def find_end_points_efficient(mask):
    # True 값이 있는 위치의 행과 열 인덱스를 찾습니다.
    rows, cols = np.where(mask)
    
    if len(rows) == 0:
        return None, None  # 마스크에 True 값이 없다면 None 반환

    # 좌측 끝점과 우측 끝점을 찾기 위해 최소 및 최대 열 인덱스를 가진 행의 인덱스를 구합니다.
    min_col_idx = np.argmin(cols)
    max_col_idx = np.argmax(cols)

    left_end_point = (rows[min_col_idx], cols[min_col_idx])
    right_end_point = (rows[max_col_idx], cols[max_col_idx])

    return left_end_point, right_end_point
