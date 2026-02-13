import cv2
import numpy as np
import matplotlib.pyplot as plt
import zombie

wost_data = {
    "solverType": "wost",
    "solver": {
        "boundaryCacheSize": 500,
        "domainCacheSize": 500,
        "nWalksForCachedSolutionEstimates": 1025,
        "nWalksForCachedGradientEstimates": 1025,
        "maxWalkLength": 10240,
        "epsilonShell": 1e-4,
        "minStarRadius": 1e-4,
        "radiusClampForKernels": 0,
        "ignoreDirichlet": False,
        "ignoreNeumann": True,
        "ignoreSource": True,
        "nWalks": 1024,
        "setpsBeforeApplyingTikhonov": 1024,
        "setpsBeforeUsingMaximalSpheres": 1024,
        "disableGradientAntitheticVariates": False,
        "disableGradientControlVariates": False,
        "useCosineSamplingForDirectionalDerivatives": False,
        "silhouettePrecision": 1e-3,
        "russianRouletteThreshold": 0.99,
        "useFiniteDifferencesForBoundaryDerivatives": True,
    },
    "modelProblem": {
        "geometry": "cube.obj",
        # "boundary": '/home/hviswan/Documents/Neural-Monte-Carlo-Fluid-Simulation/examples/karman/geometry_1cyl_long_open.obj',
        "absorptionCoeff": 0.0,
        "normalizeDomain": False,
        "flipOrientation": False,
        "isDoubleSided": False,
        "isWatertight": False,
    },
    "output": {
        "solutionFile": "./solutions/wost.png",
        "txtdir": "./solutions/",
        "gridRes": 300,
        "boundaryDistanceMask": 1e-3,
        "saveDebug": True,
        "saveColormapped": True,
        "colormap": "viridis",
        "colormapMinVal": 0.0,
        "colormapMaxVal": 1.0,
    },
}


def get_normalized_masked_region_points(mask):
    """
    Returns (x, y) in [0, 1] for all pixels where mask == 1.
    Bottom-left origin convention.
    """
    h, w = mask.shape
    ys, xs = np.where(mask == 1)

    xs_norm = xs / (w)
    ys_norm = ys / (h)  # Flip y for bottom-left origin
    # xs_norm = xs
    # ys_norm = ys

    coords = np.stack([xs_norm, ys_norm], axis=1)
    return coords  # shape: (N, 2)


def generate_normalized_boundary_obj(mask):
    """
    Extract boundary and write OBJ with (x, y) in [0, 1] and lines connecting neighbors.
    Bottom-left origin.
    """
    h, w = mask.shape
    mask = mask.astype(np.uint8).copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise ValueError("No contours found.")

    print("MASK SHAPE: ", mask.shape)

    boundary = contours[0].squeeze()
    print("BOUNDARY SHAPE: ", boundary.shape)
    if boundary.ndim != 2:
        raise ValueError("Degenerate contour.")

    x = boundary[:, 0] / (w)
    y = boundary[:, 1] / (h)
    # x = boundary[:, 0]
    # y = boundary[:, 1]

    coords = np.stack([x, y], axis=1)
    sorted_indices = np.lexsort((coords[:, 1], coords[:, 0]))
    coords = coords[sorted_indices]

    obj_lines = []
    print("coords: ", coords.shape)
    # Vertices
    for x, y in coords:
        obj_lines.append(f"v {x} {y} 0.0\n")

    # Line segments
    for segment in range(coords.shape[0] - 2):
        obj_lines.append(f"l {segment + 2} {segment + 1}\n")

    return obj_lines, coords


def add_random_square_masks(image, num_masks=5, mask_size=100):
    """
    image: 2D grayscale image
    num_masks: number of small square masks to apply
    mask_size: size of each square
    """
    masked_image = image.copy()
    mask = np.zeros_like(image, dtype=np.uint8)

    h, w = image.shape

    for _ in range(num_masks):
        x = np.random.randint(0, w - mask_size)
        y = np.random.randint(0, h - mask_size)
        masked_image[y : y + mask_size, x : x + mask_size] = 0  # zero out the region
        mask[y : y + mask_size, x : x + mask_size] = 1  # mask = 1 in hole region
        mask_region = np.asarray(image[y : y + mask_size, x : x + mask_size]).astype(
            np.float32
        )

    # boundary_points[:,1] = 1 - boundary_points[:,1]/(h-1)
    return masked_image, mask, mask_region, x, y


img = cv2.imread("test_image.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
max_val = gray.max()
gray = gray / max_val

print(gray.shape)

masked_image, mask, mask_region, start_x, start_y = add_random_square_masks(
    gray, num_masks=1
)

print("START X START Y ", start_x, start_y)

h, w = gray.shape
boundary = [
    [start_x / (w), start_y / (h)],
    [start_x / (w), (start_y + 99) / (h)],
    [(start_x + 99) / (w), start_y / (h)],
    [(start_x + 99) / (w), (start_y + 99) / (h)],
]
boundary = np.asarray(boundary)
sorted_indices = np.lexsort((boundary[:, 1], boundary[:, 0]))
boundary = boundary[sorted_indices]
print(boundary)
geometry_string = []
for x, y in boundary:
    geometry_string.append(f"v {x} {y} 0.0\n")
geometry_string.append(f"l 1 2\n")
geometry_string.append(f"l 2 4\n")
geometry_string.append(f"l 4 3\n")
geometry_string.append(f"l 3 1\n")

with open(
    "/home/hviswan/Documents/Neural-Monte-Carlo-Fluid-Simulation/bindings/zombie_3d_surface/square.obj",
    "w+",
) as f:
    for i in geometry_string:
        f.write(i)


# print(geometry_string)

coords = boundary
# geometry_string, coords = generate_normalized_boundary_obj(mask)
points = get_normalized_masked_region_points(mask)
# print(points.shape)
# print(points)
# exit()
# print(mask_region.shape)
# print(coords)
# print(geometry_string)
# exit()


sceneConfig = wost_data["modelProblem"]
sceneConfig["sourceValue"] = (
    "/home/hviswan/Documents/Neural-Monte-Carlo-Fluid-Simulation/src/3d/sourceterm.png"
)
sceneConfig["isReflectingBoundary"] = "data/is_reflecting_boundary.pfm"
sceneConfig["absorbingBoundaryValue"] = "data/absorbing_boundary_value.pfm"
sceneConfig["reflectingBoundaryValue"] = "data/reflecting_boundary_value.pfm"
sceneConfig["sourceValue"] = "data/source_value.pfm"
solverConfig = wost_data["solver"]
outputConfig = wost_data["output"]
pred_region = np.zeros_like(gray).astype(np.float32)
end_x = start_x + 99
end_y = start_y + 99
sceneConfig["geometry"] = "square.obj"
scene = zombie.Scene2D(
    sceneConfig,
    "../../bindings/zombie_3d_surface/",
    gray.tolist(),
    pred_region.tolist(),
    False,
    start_x,
    start_y,
    end_x,
    end_y,
)
# print(points)

samples, p_arr, grad_arr = zombie.zombie2d(scene, solverConfig, outputConfig, points)
# print(p_arr)
print(len(points))


source_term = np.asarray(p_arr)

pred_region = np.zeros_like(gray).astype(np.float32)
h, w = mask.shape
for i in range(len(samples)):
    point = samples[i]
    x = round(point[0] * (w))
    y = round((point[1]) * (h))
    pred_region[y][x] = source_term[i]
    # print(p_arr[i])

# source_term = pred_region[start_y:start_y+300, start_x:start_x+300]
# print(source_term)
# print(mask_region)
# print(points)
# #print(geometry_string)
# pred_region = np.zeros_like(gray).astype(np.float32)
# #print(source_term)
# print(geometry_string)

scene_2 = zombie.Scene2D(
    sceneConfig,
    "../../bindings/zombie_3d_surface/",
    gray.tolist(),
    pred_region.tolist(),
    True,
    start_x,
    start_y,
    end_x,
    end_y,
)

samples, p_arr, grad_arr = zombie.zombie2d(scene_2, solverConfig, outputConfig, points)

print(len(p_arr))

p_arr = np.asarray(p_arr)

print(samples[0])

print(round(samples[0][0] * w), round(samples[0][1] * h))
print(gray[round(samples[0][1] * h)][round(samples[0][0] * w)])

print(start_x, start_y, end_x, end_y)

print("P_ARR: ", len(p_arr))
print(p_arr)
print(len(samples))
print(len(points))


import torch

pred_region = gray.copy()
h, w = mask.shape
for i in range(len(samples)):

    point = samples[i]
    x = round(point[0] * (w))
    y = round(point[1] * (h))
    # print(x, y)
    if p_arr[i] > 0:
        print(p_arr[i])
        print(y, x)
    pred_region[y][x] = p_arr[i]
    # print(p_arr[i])

pred_region_mask = pred_region[start_y : start_y + 100, start_x : start_x + 100]


ls = torch.nn.MSELoss()
print(ls(torch.from_numpy(pred_region_mask), torch.from_numpy(mask_region)))

print(pred_region_mask)
print("MASK")
print(mask_region)


# print(geometry_string)
# print(points)

plt.imshow(masked_image, cmap="gray")
plt.title("Grayscale")
plt.axis("off")
plt.show()

plt.imshow(pred_region, cmap="gray")
plt.title("Filled")
plt.show()
