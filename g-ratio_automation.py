import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import glob
import pandas as pd
from openpyxl.workbook import Workbook

# Converts pixels to actual distance
def pixels_to_meter(pixels):
    convert_to_meters = pixel_to_meter * pixels

    return convert_to_meters

# Stacks images into one image
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

# Gets the center coordinates
def get_center(x, y, w, h):
    # Add half of the width / heigh to the starting coordinate to calculate center
    cx = x + int(w / 2)
    cy = y + int(h / 2)

    return cx, cy

# Get the gene type from image file name
def get_gene(path):
    path = path.upper()

    if 'KO' in path:
        gene = "KO"

    elif 'WT' in path:
        gene = "WT"

    else:
        gene = "Unknown"

    return gene

# Coupling contours belonging to the same fiber + calculating data needed from each fiber
def get_couples(contours):
    info = []

    # Iterate contoures and find pairs with close centers
    for i in range(len(contours)):
        for j in range(len(contours)):

            # Calculate contour area
            area = cv2.contourArea(contours[j])
            # Filter out non-myelin countours based on size
            if i != j and 10000 > area > 150:

                # Draw rectangle around contour and calculate center
                x, y, w, h = cv2.boundingRect(contours[i])
                circle_center1 = get_center(x, y, w, h)

                x, y, w, h = cv2.boundingRect(contours[j])
                circle_center2 = get_center(x, y, w, h)

                # Calculate distance between both centers
                distance = math.dist(circle_center1, circle_center2)

                # Determine if same fiber according to centers distance
                if distance < 5:

                    # Calculate G-Ratio
                    area1 = cv2.contourArea(contours[i])
                    area2 = cv2.contourArea(contours[j])
                    g_ratio = math.sqrt(area1) / math.sqrt(area2)

                    # Determine proper numerator and denominator per fiber
                    if 0 < g_ratio < 1:

                        # Draw center of the fiber on BGR frame
                        cv2.circle(bgr_closing_frame, circle_center2, 1, (255, 0, 0), -1)
                        # Draw G-Ratio of the fiber on BGR frame
                        cv2.putText(bgr_closing_frame, str(round(g_ratio, 3)),
                                    (circle_center2[0] - 20,
                                     circle_center2[1] - 20), font,
                                    fontScale, color[0], thickness, cv2.LINE_AA)

                        # Draw center of the fiber on original frame
                        cv2.circle(bgr_original, circle_center2, 1, (255, 0, 0), -1)
                        # Draw G-Ratio of the fiber on original frame
                        cv2.putText(bgr_original, str(round(g_ratio, 3)),
                                    (circle_center2[0] - 20,
                                     circle_center2[1] - 20), font,
                                    fontScale, color[0], thickness, cv2.LINE_AA)

                        # Calculate inner circle area and radius in meters
                        inner_circle_contour = contours[i]
                        inner_circle_area = pixels_to_meter(area1)
                        inner_circle_radius = pixels_to_meter(math.sqrt(area1 / np.pi))  # sqrt((pi * R1^2) / (pi))

                        # Calculate outer circle area and radius in meters
                        outer_circle_contour = contours[j]
                        outer_circle_area = pixels_to_meter(area2)
                        outer_circle_radius = pixels_to_meter(math.sqrt(area2 / np.pi))  # sqrt((pi * R2^2) / (pi))

                        # Calculate axon diameter
                        axon_diameter = inner_circle_radius * 2

                        # Get the gene type (WT / KO)
                        gene = get_gene(path)

                        # Add calculated data to data list
                        info.append([inner_circle_contour, outer_circle_contour,
                                     inner_circle_area, outer_circle_area,
                                     inner_circle_radius, outer_circle_radius,
                                     axon_diameter, gene, g_ratio])

    return info

# Draw the contours in color
def draw_couples(couples):
    for cnt in couples:
        # Generate a random color
        R = np.random.randint(255)
        G = np.random.randint(255)
        B = np.random.randint(255)

        # Draw contours on both frames
        cv2.drawContours(bgr_closing_frame, [cnt[0]], 0, (R, G, B), 1)
        cv2.drawContours(bgr_closing_frame, [cnt[1]], 0, (R, G, B), 1)
        cv2.drawContours(bgr_original, [cnt[0]], 0, (R, G, B), 1)
        cv2.drawContours(bgr_original, [cnt[1]], 0, (R, G, B), 1)


# Save the data to an excel file
def export_excel(info):
    # Iterate on data
    for inf in info:
        # Add current image data to previous image data
        inner_circle_areas.append(inf[2])
        outer_circle_areas.append(inf[3])
        inner_circle_radiuses.append(inf[4])
        outer_circle_radiuses.append(inf[5])
        axon_diameters.append(inf[6])
        genes.append(inf[7])
        g_ratios.append(inf[8])

    # Naming the columns
    data = {"Inner circle area": inner_circle_areas,
            "Outer circle area": outer_circle_areas,
            "Inner circle radius": inner_circle_radiuses,
            "Outer circle radius": outer_circle_radiuses,
            "Axon diameter": axon_diameters,
            "Genes": genes,
            "G ratio": g_ratios}

    # Convert to dataframe
    df = pd.DataFrame(data)

    df.to_excel("G-ratio Automation.xlsx", index=False)

    print("saved to excel")

    return df


# Scatter plot for the data 
def scatter_plot(df):
    # Check if the dataframe is empty
    if df.empty:
        print("Dataframe empty, cannot scatter plot")
    else:
        colors = ['red', 'blue']
        genes = ['KO', 'WT']

        # Iterate on gene types
        for i in range(len(genes)):
            # Get data for current gene type
            data = df[df['Genes'] == genes[i]]

            # Get data points
            x = data['Axon diameter']
            y = data['G ratio']

            plt.scatter(x, y, c=colors[i], label=genes[i])
            plt.xlabel("Axon diameter")
            plt.ylabel("G ratio")
            plt.legend()

        plt.show()

        print("scatter plot")

def print_threshold(selected_threshold):
    print(selected_threshold)


# Variables for excel
df = pd.DataFrame()
inner_circle_areas = []
outer_circle_areas = []
inner_circle_radiuses = []
outer_circle_radiuses = []
axon_diameters = []
genes = []
g_ratios = []

# Plot properties
figsize = (10, 10)

# Text properties
font = cv2.FONT_HERSHEY_PLAIN
fontScale = 1
color = [(255, 0, 0), (0, 0, 0)]  # RGB format
thickness = 1  # 2px

# Length pixel to meter properties
tenMicroMeter = 10 * pow(10, -6)
pixel_to_meter = tenMicroMeter / 196  # 10um / 196 = pixel_to_length

# Dimension properties
dim = (1000, 1000)

# Create gray value threshold window
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 45)
cv2.createTrackbar("Threshold", "TrackBars", 255, 255, print_threshold)
circles = []

# Get a list of the input images paths
paths = glob.glob("Images/*.tif")

# Iterate on paths
for path, i in zip(paths, range(len(paths))):

    # Main loop
    while True:

        # Read gray image
        gray_img = cv2.imread(path, 0)

        # Resize image
        original_img = cv2.resize(gray_img, dim, interpolation=cv2.INTER_AREA)

        # Bilateral filter
        bilateral_img = cv2.bilateralFilter(original_img, 15, 50, 50)

        # Get selected position on threshold bar
        gray_max = cv2.getTrackbarPos("Threshold", "TrackBars")
        after_filtered = cv2.inRange(bilateral_img, 0, gray_max)

        # Reverse binary mask
        after_filtered = cv2.bitwise_not(after_filtered)

        # Closing morphology operation
        closing_img = cv2.morphologyEx(after_filtered, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), np.uint8))

        # Draw image index
        cv2.putText(original_img, f' Image number: {i + 1} / {len(paths)}', (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, color[1], thickness)

        # Connect all 4 images
        imgStack = stackImages(0.5, ([original_img, bilateral_img],
                                     [after_filtered, closing_img]))

        # Show combined image in new window
        cv2.imshow('imgStack', imgStack)

        # Show the countours drawn on original and filtered images - Click C
        if cv2.waitKey(1) & 0xFF == ord('c'): 
            # Convert to BGR color format
            bgr_closing_frame = cv2.cvtColor(after_filtered, cv2.COLOR_GRAY2BGR)
            bgr_original = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

            # Find Contours
            contours, hierarchy = cv2.findContours(closing_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Get coupled contours and calculate info per fiber
            info = get_couples(contours)
            # Draw random color line around countours
            draw_couples(info)

            # Show the countours
            countour_image = np.hstack((bgr_original, bgr_closing_frame))
            plt.figure(figsize=figsize)
            plt.imshow(countour_image, cmap='gray', vmin=0, vmax=255)
            plt.title("countour_image")
            plt.show()

        # Save data to excel - Click S
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            # Convert to BGR color format
            bgr_closing_frame = cv2.cvtColor(after_filtered, cv2.COLOR_GRAY2BGR)
            bgr_original = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

            # Find Contours
            contours, hierarchy = cv2.findContours(closing_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Get coupled contours and calculate info per fiber
            info = get_couples(contours)
            # Draw random color line around countours
            draw_couples(info)

            # Export to excel
            df = export_excel(info)

            # After saving excel, move on to the next image
            break

# After processing all the images, show a plot with the results
scatter_plot(df)
