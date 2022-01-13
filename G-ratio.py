import cv2
import numpy as np
import math
import glob
import pandas as pd
import threading
from openpyxl.workbook import Workbook
from matplotlib import pyplot as plt
from tkinter import *


class GUI:
    after_filtered, original_img, closing_img = None, None, None
    bgr_closing_frame, bgr_original, path = None, None, None
    bilateral_img = None

    # gui buttons
    def plot_button(self):
        # convert to BGR color format
        bgr_closing_frame = cv2.cvtColor(gui.after_filtered, cv2.COLOR_GRAY2BGR)
        bgr_original = cv2.cvtColor(gui.original_img, cv2.COLOR_GRAY2BGR)

        # findContours
        contours, hierarchy = cv2.findContours(gui.closing_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        info = get_couples(contours)
        draw_couples(info)
        info = []

        plot_img = np.hstack((bgr_original, bgr_closing_frame))
        plt.figure(figsize=figsize)
        plt.imshow(plot_img, cmap='gray', vmin=0, vmax=255)
        plt.title("plot_img")
        plt.show()

    def save_button(self):
        # convert to BGR color format
        bgr_closing_frame = cv2.cvtColor(gui.after_filtered, cv2.COLOR_GRAY2BGR)
        bgr_original = cv2.cvtColor(gui.original_img, cv2.COLOR_GRAY2BGR)

        # findContours
        contours, hierarchy = cv2.findContours(gui.closing_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # search for myelin
        info = get_couples(contours)

        # draw myelins contoursss
        draw_couples(info)

        # export to excel
        df = export_excel(info)

        plot_img = np.hstack((bgr_original, bgr_closing_frame))
        plt.figure(figsize=figsize)
        plt.imshow(plot_img, cmap='gray', vmin=0, vmax=255)
        plt.title("plot_img")
        plt.show()

    def scatter_plot_button(self):
        scatter_plot(df)


# import class
gui = GUI()


def thread_func():
    print("thread started")  # send_time = time.time()
    window = Tk()

    window.geometry("800x300")
    window.configure(bg="#ffffff")
    window.title("g-ratio")

    window.resizable(False, False)

    while True:
        # tk GUI
        canvas = Canvas(
            window,
            bg="#ffffff",
            height=300,
            width=800,
            bd=0,
            highlightthickness=0,
            relief="ridge")
        canvas.place(x=0, y=0)

        background_img = PhotoImage(file=f"background.png")
        background = canvas.create_image(
            400.0, 150.0,
            image=background_img)

        # slider
        global s1
        s1 = Scale(window, from_=0, to=255, length=600, tickinterval=0,
                   orient=HORIZONTAL, bd=0, bg='white', highlightthickness=0)
        s1.set(255)
        s1.place(x=100, y=100)

        img0 = PhotoImage(file=f"img0.png")
        b0 = Button(
            image=img0,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: gui.save_button(),
            relief="flat")

        b0.place(
            x=325, y=159,
            width=150,
            height=50)

        img1 = PhotoImage(file=f"img1.png")
        b1 = Button(
            image=img1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: gui.plot_button(),
            relief="flat")

        b1.place(
            x=108, y=159,
            width=150,
            height=50)

        img2 = PhotoImage(file=f"img2.png")
        b2 = Button(
            image=img2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: gui.scatter_plot_button(),
            relief="flat")

        b2.place(
            x=542, y=159,
            width=150,
            height=50)

        window.mainloop()


def pixels_to_meter(pixels):
    convert_to_meters = pixel_to_meter * pixels

    return convert_to_meters


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


def get_center(x, y, w, h):
    cx = x + w // 2
    cy = y + h // 2

    return cx, cy


def get_couples(contours):
    info = []

    for i in range(len(contours)):
        for j in range(len(contours)):

            area = cv2.contourArea(contours[j])

            if i != j and 10000 > area > 150:

                x, y, w, h = cv2.boundingRect(contours[i])
                circle_center1 = get_center(x, y, w, h)

                x, y, w, h = cv2.boundingRect(contours[j])
                circle_center2 = get_center(x, y, w, h)

                distance = math.dist(circle_center1, circle_center2)

                if distance < 5:

                    area1 = cv2.contourArea(contours[i])
                    area2 = cv2.contourArea(contours[j])

                    # calc g-ratio
                    g_ratio = math.sqrt(area1) / math.sqrt(area2)

                    # inner circle divided outer circle
                    if 0 < g_ratio < 1:
                        cv2.circle(gui.bgr_closing_frame, circle_center2, 1, (255, 0, 0), -1)
                        cv2.putText(gui.bgr_closing_frame, str(round(g_ratio, 3)),
                                    (circle_center2[0] - 20,
                                     circle_center2[1] - 20), font,
                                    fontScale, color[0], thickness, cv2.LINE_AA)

                        cv2.circle(gui.bgr_original, circle_center2, 1, (255, 0, 0), -1)
                        cv2.putText(gui.bgr_original, str(round(g_ratio, 3)),
                                    (circle_center2[0] - 20,
                                     circle_center2[1] - 20), font,
                                    fontScale, color[0], thickness, cv2.LINE_AA)

                        # calc inner circle parameters
                        inner_circle_contour = contours[i]
                        inner_circle_area = pixels_to_meter(area1)
                        inner_circle_radius = pixels_to_meter(math.sqrt(area1 / np.pi))  # sqrt((pi * R1^2) / (pi))

                        # calc outer circle parameters
                        outer_circle_contour = contours[j]
                        outer_circle_area = pixels_to_meter(area2)
                        outer_circle_radius = pixels_to_meter(math.sqrt(area2 / np.pi))  # sqrt((pi * R2^2) / (pi))

                        # calc axon diameter
                        axon_diameter = inner_circle_radius * 2

                        # give me gene (WT / KO)
                        gene = get_gene(gui.path)

                        # append
                        info.append([inner_circle_contour, outer_circle_contour,
                                     inner_circle_area, outer_circle_area,
                                     inner_circle_radius, outer_circle_radius,
                                     axon_diameter, gene, g_ratio])

    return info


def draw_couples(couples):
    for cnt in couples:
        R = np.random.randint(255)
        G = np.random.randint(255)
        B = np.random.randint(255)

        cv2.drawContours(gui.bgr_closing_frame, [cnt[0]], 0, (R, G, B), 1)
        cv2.drawContours(gui.bgr_closing_frame, [cnt[1]], 0, (R, G, B), 1)
        cv2.drawContours(gui.bgr_original, [cnt[0]], 0, (R, G, B), 1)
        cv2.drawContours(gui.bgr_original, [cnt[1]], 0, (R, G, B), 1)


def get_gene(path):
    path = path.upper()

    if 'KO' in path:
        gene = "KO"

    elif 'WT' in path:
        gene = "WT"

    else:
        gene = "Unknown"

    return gene


def export_excel(info):
    for inf in info:
        inner_circle_areas.append(inf[2])
        outer_circle_areas.append(inf[3])
        inner_circle_radiuses.append(inf[4])
        outer_circle_radiuses.append(inf[5])
        axon_diameters.append(inf[6])
        genes.append(inf[7])
        g_ratios.append(inf[8])

    data = {"Inner circle area": inner_circle_areas,
            "Outer circle area": outer_circle_areas,
            "Inner circle radius": inner_circle_radiuses,
            "Outer circle radius": outer_circle_radiuses,
            "Axon diameter": axon_diameters,
            "Genes": genes,
            "G ratio": g_ratios, }

    df = pd.DataFrame(data)

    df.to_excel("G-ratio Automation.xlsx", index=False)

    print("saved to excel")

    return df


def scatter_plot(df):
    colors = ['red', 'blue']
    genes = ['KO', 'WT']

    for i in range(2):
        data = df[df['Genes'] == genes[i]]

        x = data['Axon diameter']
        y = data['G ratio']

        plt.scatter(x, y, c=colors[i], label=genes[i])
        plt.xlabel("Axon diameter")
        plt.ylabel("G ratio")
        plt.legend()

    plt.show()

    print("scatter plot")


def empty(a):
    pass


# variables for excel
inner_circle_areas = []
outer_circle_areas = []
inner_circle_radiuses = []
outer_circle_radiuses = []
axon_diameters = []
genes = []
g_ratios = []

# Text properties
font = cv2.FONT_HERSHEY_PLAIN
fontScale = 1
color = [(255, 0, 0), (0, 0, 0)]  # RGB format
thickness = 1  # 2px

# length pixel to meter properties
tenMicroMeter = 10 * pow(10, -6)
pixel_to_meter = tenMicroMeter / 196  # 10um / 196 = pixel_to_length

# dimension properties
dim = (1000, 1000)

# plot properties
figsize = (10, 10)

circles = []

# slider
s1 = []
df = []


# global variables


def main():

    # Input path
    paths = glob.glob("Images/*.tif")

    for gui.path, i in zip(paths, range(len(paths))):

        while True:
            # read gray img
            gray_img = cv2.imread(gui.path, 0)

            # resize image
            gui.original_img = cv2.resize(gray_img, dim, interpolation=cv2.INTER_AREA)

            # bilateral filter
            gui.bilateral_img = cv2.bilateralFilter(gui.original_img, 15, 50, 50)

            # threshold
            gui.after_filtered = cv2.inRange(gui.bilateral_img, 0, s1.get())

            # reverse binary mask
            gui.after_filtered = cv2.bitwise_not(gui.after_filtered)

            # closing morphology operation
            gui.closing_img = cv2.morphologyEx(gui.after_filtered, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), np.uint8))

            # draw image index
            cv2.putText(gui.original_img, f' Image number: {i + 1} / {len(paths)}', (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, color[1], thickness)

            imgStack = stackImages(0.5, ([gui.original_img, gui.bilateral_img],
                                         [gui.after_filtered, gui.closing_img]))

            cv2.imshow('imgStack', imgStack)
            cv2.waitKey(1)

    scatter_plot(df)


if __name__ == '__main__':
    x = threading.Thread(target=thread_func, daemon=True)
    x.start()
    main()
