import tkinter as tk
import cv2
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from ImageCanvas import ImageCanvas
import numpy as np
from sklearn.cluster import KMeans
image:np.ndarray = None
crop = False
border_types = {
    "Border Replicate": cv2.BORDER_REPLICATE,
    "Border Reflect": cv2.BORDER_REFLECT,
    "Border Reflect 101": cv2.BORDER_REFLECT_101,
    "Border Wrap": cv2.BORDER_WRAP,
    "Border Constant": cv2.BORDER_CONSTANT,
    "None":None
}
root = tk.Tk()
style = ttk.Style()
color_left = "lightblue"
color_middle = "white"
color_right = "lightgreen"
style.theme_use("default")
style.configure("TButton", 
                foreground="white",
                background="#003066",
                font=("Arial", 12, "bold"),
                padding = (5,5,5,5),
                margin = (0,10,0,0)
                )
style.configure("TLabel",
                font = ("Arial", 12, "bold"),
                margin = (0,10,0,0)
                )
style.configure("TScale",
                background = "lightblue",
                font = ("Arial", 12, "bold"),
                margin = (0,10,0,0),
                sliderthickness =20,
                sliderlength= 10,
                troughrelief = "flat" ,
                )
style.configure("TCheckbutton",
                foreground = "black",
                background = color_left,
                font = ("Arial", 12),
                margin = (0,10,0,0)
                )
style.configure("TRadiobutton",
                foreground = "black",
                background = color_middle,
                font = ("Arial", 12),
                margin = (0,10,0,0)
                )
style.configure(
    "TCombobox",
    foreground="lightblue",  # Text color
    background="black",  # Background of the entry field
    font=("Arial", 10, "bold"),
    padding = (5,5,5,5),
    margin = (0,10,0,0)
)
# Configure the dropdown list
style.map(
    "TCombobox",
    fieldbackground=[("readonly", "lightblue")],  # Background of the entry field (readonly)
    background=[("readonly", "lightblue")], 
    foreground=[("readonly", "white")]# Background of the dropdown menu
)
style.map(
    "TButton",
    background=[("pressed", "black"), ("active", "black")],
    foreground=[("pressed", "white"), ("active", "white")]
)
sepia = tk.BooleanVar()
cyanotype = tk.BooleanVar()
vignette = tk.BooleanVar()
compress_type = tk.StringVar()
root.state("zoomed")
root.title("Image Editor")
def overlayChange(e):
    canvas.image_overlay_alpha = overlay_slider_alpha.get()
    canvas.image_overlay_position = (overlay_slider_x.get(), overlay_slider_y.get())
    canvas.show_image(image)
def change_crop_state():
    global crop
    crop = not crop
def color_manipulation():
    global image
    canvas.color_manipulation = (slider_color_red.get()/100, slider_color_green.get()/100, slider_color_blue.get()/100)
    canvas.show_image(image)
def convertGrayScale():
    global image
    canvas.isGrayScale = not canvas.isGrayScale
    canvas.show_image(image)
def importImageClick():
    global image
    filename = filedialog.askopenfilename(
    title="Select an Image File",
    filetypes=[("Image Files", "*.png;*.jpg;*.jpeg"),  # Include supported image formats
               ("All Files", "*.*")]  # Option to show all files
    )
    if filename:  # Check if a file was selected
        resetTransformations()
        image = cv2.imread(filename)
        if image is not None:  # Check if the image was successfully read
            canvas.show_image(image)
def save():
    file_types = [
        ("PNG files", "*.png"),
        ("JPG files", "*.jpg"),
        ("All files", "*.*")
    ]
    path = None
    compression_type = compress_type.get()
    if compression_type == "None":
        pass    
    elif compression_type == "DCT":
        file_types.pop(0)
    else:
        file_types.pop(1)
    
    path = filedialog.asksaveasfilename(
    defaultextension=".png", 
    filetypes=file_types
    )
    if path:
        canvas.save(path, compression_type)
def negativeTransformation():
    global image
    image = cv2.bitwise_not(image)
    canvas.show_image(image)

def onRotateChange(value):
    global image
    angle = int(value)
    canvas.rotation_degree = angle
    canvas.show_image(image)

def flipHorizontal():
    global image
    image = cv2.flip(image, 1)
    canvas.show_image(image)

def flipVertical():
    global image
    image = cv2.flip(image, 0)
    canvas.show_image(image)

def flipDiagonal():
    global image
    image = cv2.flip(image, 1)
    image = cv2.flip(image, 0)
    canvas.show_image(image)

def translate(e):
    global image
    try:
        entry_x = entry_translatex.get() if entry_translatex.get() != "" else 0
        entry_y = entry_translatey.get() if entry_translatey.get() != "" else 0
        canvas.translate = (int(entry_x), int(entry_y))
        canvas.show_image(image)
    except ValueError:
        return

def scale(e):
    global image
    try:
        entry_x = entry_scalingx.get() if entry_scalingx.get() != "" else 100
        entry_y = entry_scalingy.get() if entry_scalingy.get() != "" else 100
        entry_x = int(entry_x)
        entry_y = int(entry_y)
        canvas.resize = (entry_x/100, entry_y/100)
        canvas.show_image(image)
    except ValueError:
        return

def color_filter():
    global image
    color_filters = canvas.color_filters
    # Define the filters and their corresponding variables in a list
    filter_options = [
        ("sepia", sepia),
        ("cyanotype", cyanotype),
        ("vignette", vignette)
    ]

    # Iterate over the filters and update the color_filters list
    for filter_name, filter_var in filter_options:
        if filter_var.get() == 1 and filter_name not in color_filters:
            color_filters.append(filter_name)
        elif filter_var.get() == 0 and filter_name in color_filters:
            color_filters.remove(filter_name)

    # Update the canvas property
    canvas.color_filters = color_filters
    canvas.show_image(image)

def brightnessChange(event):
    global image
    brightness = event.widget.get()
    canvas.brightness = float(brightness)
    canvas.show_image(image)

def contrastChange(event):
    global image
    contrast = event.widget.get()
    canvas.contrast = float(contrast)
    canvas.show_image(image)
def borderChange(event):
    global image
    border = event.widget.get()
    canvas.border = border_types[border]
    canvas.show_image(image)
def paddingChange(event):
    global image
    padding = event.widget.get()
    canvas.padding = int(padding)
    canvas.show_image(image)
def contrast_stretching():
    canvas.contrast_stretch = not canvas.contrast_stretch
    canvas.show_image(image)
def binaryOperation(operation):
    global image
    operations = {
            "AND": cv2.bitwise_and,
            "OR": cv2.bitwise_or,
            "XOR": cv2.bitwise_xor,
            "NOT": cv2.bitwise_not
        }
    if operation == "NOT":
        if image is None: return
        image = operations[operation](image)
        canvas.show_image(image)
        return
    filename = filedialog.askopenfilename(
    title="Select the second Image File",
    filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),  # Include supported image formats
               ("All Files", "*.*")]  # Option to show all files
    )
    if filename:
        image2 = cv2.imread(filename)
        label_canvas2.pack()
        canvas2.pack()
        canvas2.show_image(image2)

        if operation == "BLEND":
            canvas.image_blend = image2
            canvas.show_image(image)
            label_blend.grid(row=label_blend_pos[0], column=label_blend_pos[1], columnspan=4, sticky="ew")
            alpha_slider.grid(row=alpha_slide_pos[0],column=alpha_slide_pos[1], columnspan=4, sticky="ew", pady=(0,10))
            return
        elif operation == "MATCH" or operation == "MATCH_ORB":
            if operation =="MATCH_ORB":
                canvas.isOrb = True
            else:
                canvas.isOrb = False
            canvas.image_match = image2
            canvas.show_image(image)
        elif operation == "TEMPLATE":
            label_canvas2.pack()
            canvas2.pack()
            canvas.image_template = image2
            canvas2.show_image(image2)
            canvas.show_image(image)
        elif operation == "OVERLAY":
            canvas.image_overlay = image2
            label_overlay.pack()
            overlay_slider_alpha.pack(expand=True, fill="x")
            label_overlay_x.pack()
            overlay_slider_x.pack(fill="x", expand=True)
            label_overlay_y.pack()
            overlay_slider_y.pack(fill="x", expand=True)
            canvas.image_overlay_position= (overlay_slider_x.get(), overlay_slider_y.get())
            canvas.image_overlay_alpha = overlay_slider_alpha.get()
            canvas.show_image(image)
        elif image2 is not None:
            image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))
            image = operations[operation](canvas.getTransformedImage(), image2)
            resetTransformations()
            canvas.show_image(image)
    else:
        return
def start_crop(event):
    global start_x, start_y, rect_id, crop
    if not crop : return
    start_x, start_y = event.x, event.y
    rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline="red")

def draw_crop(event):
    global rect_id, crop
    if not crop : return
    if rect_id:
        canvas.coords(rect_id, start_x, start_y, event.x, event.y)

def end_crop(event):
    global start_x, start_y, image, crop
    if not crop : return
    if rect_id and image is not None:
        # Get canvas coordinates of the cropping rectangle
        x1, y1, x2, y2 = canvas.coords(rect_id)
        
        # Convert canvas coordinates to image coordinates by accounting for canvas scaling
        scale_x = image.shape[1] / canvas.winfo_width()  # Scaling factor for x
        scale_y = image.shape[0] / canvas.winfo_height()  # Scaling factor for y
        
        # Adjust coordinates based on scaling
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        if x1 < x2 and y1 < y2:
            image = canvas.getTransformedImage()[y1:y2, x1:x2]
            canvas.show_image(image)
            resetTransformations()
            crop = not crop
        else:
            return
def resetTransformations():
    global image
    canvas.resetTransformation()
    entry_translatex.delete(0, tk.END)
    entry_translatey.delete(0, tk.END)
    entry_scalingx.delete(0, tk.END)
    entry_scalingy.delete(0, tk.END)
    slider_rotate.set(0)
    alpha_slider.set(0.5)
    label_blend.pack_forget()
    alpha_slider.pack_forget()
    label_overlay.pack_forget()
    overlay_slider_alpha.pack_forget()
    overlay_slider_alpha.set(1)
    label_overlay_x.pack_forget()
    overlay_slider_x.pack_forget()
    overlay_slider_x.set(0)
    label_overlay_y.pack_forget()
    overlay_slider_y.pack_forget()
    overlay_slider_y.set(0)
    canvas.show_image(image)

def alphaChange(event):
    global image
    canvas.image_blend_alpha = event.widget.get()
    canvas.show_image(image)

def fourier_transform():
    global image
    resetTransformations()
    if image is None:
        return
    
    try:
        # Get the selection (FFT or DFT)
        transform_type = select_box_fourier.get()
        
        # Convert the image to grayscale for simplicity in Fourier Transform
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if transform_type == "FFT":
            f_transform = np.fft.fft2(gray_image)
            f_shift = np.fft.fftshift(f_transform)  # Center the low frequencies
            image = 20 * np.log(np.abs(f_shift))
            
            
        elif transform_type == "DFT":
            dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            image = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        
        else:
            return
        
        # Normalize and display the result
        normalized_spectrum = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image = cv2.cvtColor(normalized_spectrum, cv2.COLOR_GRAY2BGR)
        canvas.show_image(image)
    except Exception as e:
        return
def apply_blur_filter():
    global image
    canvas.blur_filter = select_box_blur.get()
    canvas.show_image(image)

def applyEdgeDetection():
    global image
    edge_detection = select_box_edge.get()
    canvas.edge_detection = edge_detection
    canvas.show_image(image)
def apply_histogram_equalization():
    global image
    canvas.histogram = not canvas.histogram
    canvas.show_image(image)
def applySegmentation():
    global image
    segmentation = select_box_segmentation.get()
    if segmentation == "K-means":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pixels = image.reshape((-1, 1))  # Each pixel is a 1D vector (grayscale intensity)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)  # We choose k=2 for simplicity
        Kmean_image = kmeans.labels_.reshape(image.shape)
        kmeans_image = (Kmean_image * 255).astype(np.uint8)
        image = kmeans_image
        canvas.show_image(image)
    if segmentation =="Thresholding":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        image = thresholded_image
        canvas.show_image(image)

def applyRestore(method):
    restorations = canvas.restoration
    if method not in restorations:
        restorations.append(method)
    else:
        restorations.remove(method)
    canvas.restoration = restorations
    canvas.show_image(image)

def gammaChange(event):
    global image
    gamma = event.widget.get()
    canvas.gamma = float(gamma)
    canvas.show_image(image)
def imageMorph(operation):
    morphs = canvas.image_morph
    if operation not in morphs:
        morphs.append(operation)
    else:
        morphs.remove(operation)
    canvas.image_morph = morphs
    canvas.show_image(image)
components = 0
components_right = 0
def addToLeftGrid(component, colspan=1, sticky="ew", **kwargs):
    global components
    row = components // 4
    column = components % 4
    component.grid(row=row, column=column, columnspan=colspan, sticky=sticky, **kwargs)
    components += colspan
    return (row, column)

def addToRightGrid(component, colspan=1, sticky="ew", **kwargs):
    global components_right
    row = components_right // 4
    column = components_right % 4
    component.grid(row=row, column=column, columnspan=colspan, sticky=sticky, **kwargs)
    components_right += colspan
    return (row, column)


# Configure grid layout
root.grid_columnconfigure(0, weight=1)  # Left menu
root.grid_columnconfigure(1, weight=2)  # Canvas
root.grid_columnconfigure(2, weight=1)  # Right menu
root.grid_rowconfigure(0, weight=1)


# Create frames
frame1 = tk.Frame(root, background=color_left, width=425, height=844)  # Left menu
frame2 = tk.Frame(root, background=color_middle, width=850, height=844)     # Canvas (middle)
frame3 = tk.Frame(root, background=color_right, width=425, height=844) # Right menu
frame1.rowconfigure(0, weight=1)
frame2.rowconfigure(0, weight=1)
frame3.rowconfigure(0, weight=1)
frame1.grid(row=0, column=0, sticky="nswe")
frame2.grid(row=0, column=1, sticky="nswe")
frame3.grid(row=0, column=2, sticky="nswe")
frame1.pack_propagate(False)
frame2.pack_propagate(False)
frame3.pack_propagate(False)

ttk.Label(frame2, text="Canvas Area", background=color_middle).pack(pady=10)

canvas = ImageCanvas(frame2)
        

label1 = ttk.Label(frame2, text="Image", background="white", font=("Arial", 12, "bold"), foreground="black")
label1.pack()
button = ttk.Button(frame2, text="Import Image", command=importImageClick)
button.pack()
canvas.bind("<ButtonPress-1>", start_crop)
canvas.bind("<B1-Motion>", draw_crop)
canvas.bind("<ButtonRelease-1>", end_crop)
canvas.pack()


frame_grid_left = tk.Frame(frame1, background=color_left, padx=10)
frame_grid_left.columnconfigure(0, weight=1)
frame_grid_left.columnconfigure(1, weight=1)
frame_grid_left.columnconfigure(2, weight=1)
frame_grid_left.columnconfigure(3, weight=1)
frame_grid_left.pack(fill="both", expand=True)

frame_grid_right = tk.Frame(frame3, background=color_right, padx=10)
frame_grid_right.columnconfigure(0, weight=1)
frame_grid_right.columnconfigure(1, weight=1)
frame_grid_right.columnconfigure(2, weight=1)
frame_grid_right.columnconfigure(3, weight=1)
frame_grid_right.pack(fill="both", expand=True)

label_basic_operations = ttk.Label(frame_grid_left, text="Basic Operations", anchor="center",font=("Arial", 15, "bold"), background=color_left)
addToLeftGrid(label_basic_operations, colspan=4, pady=10)
button_grayscale = ttk.Button(frame_grid_left, text="Gray scale",command=convertGrayScale)
addToLeftGrid(button_grayscale, colspan=2)
button_negative = ttk.Button(frame_grid_left, text="Negative Transformation", command=negativeTransformation)
addToLeftGrid(button_negative, colspan=2)
label_color_manipulation = ttk.Label(frame_grid_left, text="Color Manipulation", anchor="center", background=color_left)
addToLeftGrid(label_color_manipulation, colspan=4, pady = 10)

label_color_red = ttk.Label(frame_grid_left, text="Red: 100", anchor="center", background=color_left)
addToLeftGrid(label_color_red, colspan=4)
slider_color_red = ttk.Scale(frame_grid_left, from_=0, to=200, orient="horizontal", command=lambda value: label_color_red.config(text=f"Red: {float(value):.0f}"))
addToLeftGrid(slider_color_red, colspan=4)
label_color_green = ttk.Label(frame_grid_left, text="Green: 100", anchor="center", background=color_left)
addToLeftGrid(label_color_green, colspan=4)
slider_color_green = ttk.Scale(frame_grid_left, from_=0, to=200, orient="horizontal", command=lambda value: label_color_green.config(text=f"Green: {float(value):.0f}"))
addToLeftGrid(slider_color_green, colspan=4)
label_color_blue = ttk.Label(frame_grid_left, text="Blue: 100", anchor="center", background=color_left)
addToLeftGrid(label_color_blue, colspan=4)
slider_color_blue = ttk.Scale(frame_grid_left, from_=0, to=200, orient="horizontal", command=lambda value: label_color_blue.config(text=f"Blue: {float(value):.0f}"))
addToLeftGrid(slider_color_blue, colspan=4)
for i in [slider_color_red, slider_color_green, slider_color_blue]:
    i.set(100)
    i.bind("<ButtonRelease-1>", lambda e: color_manipulation())

label_flip_operation = ttk.Label(frame_grid_left, text="Flip Operations", anchor="center", background=color_left)
addToLeftGrid(label_flip_operation, colspan=4, pady = 10)
button_flipv_icon = ImageTk.PhotoImage(Image.open("icons/vertical_flip.png").resize((30,30)))
button_fliph_icon = ImageTk.PhotoImage(Image.open("icons/horizontal_flip.png").resize((30,30)))
button_flipd_icon = ImageTk.PhotoImage(Image.open("icons/diagonal_flip.png").resize((30,30)))
container_flip = tk.Frame(frame_grid_left, background=color_left)
buttons_container = tk.Frame(container_flip, background=color_left)
button_fliph = ttk.Button(buttons_container, image=button_fliph_icon, command=flipHorizontal)
button_fliph.pack(side="left")
button_flipv = ttk.Button(buttons_container, image=button_flipv_icon, command=flipVertical)
button_flipv.pack(side="left", padx=20)
button_flipd = ttk.Button(buttons_container, image=button_flipd_icon, command=flipDiagonal)
button_flipd.pack(side="left")
buttons_container.pack(side="top")
addToLeftGrid(container_flip, colspan=4)

label_translate = ttk.Label(frame_grid_left, text="Translate", background=color_left, anchor="center")
addToLeftGrid(label_translate, colspan=4, pady = 10)
container_translate = tk.Frame(frame_grid_left, background=color_left)
pack_container_translate = tk.Frame(container_translate, background=color_left)
label_translatex = ttk.Label(pack_container_translate, text="X:", background=color_left)
label_translatex.pack(side="left", padx=10)
entry_translatex = tk.Entry(pack_container_translate)
entry_translatex.bind("<Return>", translate)
entry_translatex.pack(side="left")
label_translatex2 = ttk.Label(pack_container_translate, text="px" , background=color_left)
label_translatex2.pack(side="left")
label_translatey = ttk.Label(pack_container_translate, text="Y:", background=color_left)
label_translatey.pack(side="left", padx=10)
entry_translatey = tk.Entry(pack_container_translate)
entry_translatey.bind("<Return>", translate)
entry_translatey.pack(side="left")
label_translatey2 = ttk.Label(pack_container_translate, text="px", background=color_left)
label_translatey2.pack(side="left")
pack_container_translate.pack(side="top")
addToLeftGrid(container_translate, colspan=4)

label_scaling = ttk.Label(frame_grid_left, text="Scale", background=color_left, anchor="center")
addToLeftGrid(label_scaling, colspan=4, pady=10)
container_scaling = tk.Frame(frame_grid_left, background=color_left)
pack_container_scaling = tk.Frame(container_scaling, background=color_left)
label_scalingx = ttk.Label(pack_container_scaling, text="X:", background=color_left)
label_scalingx.pack(side="left", padx=10)
entry_scalingx = tk.Entry(pack_container_scaling)
entry_scalingx.pack(side="left")
entry_scalingx.bind("<Return>", scale)
label_scalingx2 = ttk.Label(pack_container_scaling, text="%", background=color_left)
label_scalingx2.pack(side="left")
label_scalingy = ttk.Label(pack_container_scaling, text="Y:", background=color_left)
label_scalingy.pack(side="left", padx=10)
entry_scalingy = tk.Entry(pack_container_scaling)
entry_scalingy.pack(side="left")
entry_scalingy.bind("<Return>", scale)
label_scalingy2 = ttk.Label(pack_container_scaling, text="%", background=color_left)
label_scalingy2.pack(side="left")
pack_container_scaling.pack(side="top")
addToLeftGrid(container_scaling, colspan=4)

label_rotate = ttk.Label(frame_grid_left, text="Rotate: 0°", anchor="center", background=color_left)
addToLeftGrid(label_rotate, colspan=4, pady=10)
slider_rotate = ttk.Scale(frame_grid_left, from_=-180, to=180, orient="horizontal", command=lambda value: label_rotate.config(text=f"Rotate: {float(value):.2f}°"))
addToLeftGrid(slider_rotate, colspan=4)
slider_rotate.bind("<ButtonRelease-1>", lambda event: onRotateChange(slider_rotate.get()))

label_crop = ttk.Label(frame_grid_left, text="Toggle crop: ", background=color_left, anchor="center")
addToLeftGrid(label_crop, colspan=2, pady=10)
button_crop = ttk.Button(frame_grid_left, text="Crop", command=change_crop_state)
addToLeftGrid(button_crop, colspan=2, pady=10)

button_blend = ttk.Button(frame_grid_left, text="Image Blending", command=lambda : binaryOperation("BLEND"))
addToLeftGrid(button_blend, colspan=2, pady = (0,10))
label_blend = ttk.Label(frame_grid_left, text="Image Blend Alpha: 0.5", background=color_left, anchor="center")
button_overlay = ttk.Button(frame_grid_left, text="Overlay", command=lambda : binaryOperation("OVERLAY"))
addToLeftGrid(button_overlay, colspan=2, pady=(0,10))
label_blend_pos = addToLeftGrid(label_blend, colspan=4)
label_blend.grid_forget()
alpha_slider = ttk.Scale(frame_grid_left, 
                        from_=0, to=1,
                        orient=tk.HORIZONTAL, command=lambda e: label_blend.config(text = f"Image Blend Alpha: {float(e):.2f}"))
alpha_slide_pos = addToLeftGrid(alpha_slider, colspan=4)
alpha_slider.bind("<ButtonRelease-1>", alphaChange)
alpha_slider.set(0.5)
alpha_slider.grid_forget()

overlay_parameter_container = tk.Frame(frame_grid_left, background=color_left)
label_overlay = ttk.Label(overlay_parameter_container, text="Image overlay Alpha: 1", background=color_left)
overlay_slider_alpha = ttk.Scale(overlay_parameter_container, 
                        from_=0, to=1,
                        orient=tk.HORIZONTAL, command=lambda e: label_overlay.config(text = f"Image overlay Alpha: {float(e):.2f}"))
                        
overlay_slider_alpha.bind("<ButtonRelease-1>", overlayChange)
overlay_slider_alpha.set(1)
label_overlay_x = ttk.Label(overlay_parameter_container, text="Overlay X: 0", anchor="center", background=color_left)
overlay_slider_x = ttk.Scale(overlay_parameter_container,
                        from_=0, to=1,
                        orient=tk.HORIZONTAL, command=lambda e: label_overlay_x.config(text = f"Overlay X: {float(e):.2f}"))
overlay_slider_x.bind("<ButtonRelease-1>", overlayChange)
overlay_slider_x.set(0)

label_overlay_y = ttk.Label(overlay_parameter_container, text="Overlay Y: 0", anchor="center", background=color_left)
overlay_slider_y = ttk.Scale(overlay_parameter_container,
                        from_=0, to=1,
                        orient=tk.HORIZONTAL, command=lambda e: label_overlay_y.config(text = f"Overlay Y: {float(e):.2f}"))
overlay_slider_y.bind("<ButtonRelease-1>", overlayChange)
overlay_slider_y.set(0)
addToLeftGrid(overlay_parameter_container, colspan=4)
label_color_filter = ttk.Label(frame_grid_left, text="Color Filter", background=color_left, anchor="center")
addToLeftGrid(label_color_filter, colspan=4, pady=10)
frame_filter = tk.Frame(frame_grid_left, background=color_left)
checkbox_sepia = ttk.Checkbutton(frame_filter, text="Sepia", onvalue=1, offvalue=0, variable=sepia)
checkbox_sepia.grid(row=0, column=0)
checkbox_cyanotype = ttk.Checkbutton(frame_filter, text="Cyanotype", onvalue=1, offvalue=0, variable=cyanotype)
checkbox_cyanotype.grid(row=0, column=1)
checkbox_vignette = ttk.Checkbutton(frame_filter, text="Vignette", onvalue=1, offvalue=0, variable=vignette)
checkbox_vignette.grid(row=0, column=2)
for i in range(3): frame_filter.grid_columnconfigure(i, weight=1)
for i, checkbox in enumerate([checkbox_sepia, checkbox_cyanotype, checkbox_vignette]):
    checkbox.configure(command=lambda: color_filter())
addToLeftGrid(frame_filter, colspan=4)
label_brightness = ttk.Label(frame_grid_left, text="Brightness: 1.0", anchor="center", background=color_left)
addToLeftGrid(label_brightness, colspan=4, pady = (10,0))
brightness_slider = ttk.Scale(frame_grid_left, 
                            from_=0.1, to=3.0, 
                            orient=tk.HORIZONTAL, command=lambda value: label_brightness.config(text=f"Brightness: {float(value):.1f}"))
brightness_slider.bind("<ButtonRelease-1>", brightnessChange)
brightness_slider.set(1.0)
addToLeftGrid(brightness_slider, colspan=4)

label_contrast = ttk.Label(frame_grid_left, text="Contrast:", anchor="center", background=color_left)
contrast_slider = ttk.Scale(frame_grid_left, 
                            from_=0.1, to=3.0, 
                            orient=tk.HORIZONTAL, command=lambda value: label_contrast.config(text=f"Contrast: {float(value):.1f}"))
contrast_slider.set(1.0)
contrast_slider.bind("<ButtonRelease-1>", contrastChange)
addToLeftGrid(label_contrast, colspan=4, pady=(10,0))
addToLeftGrid(contrast_slider, colspan=4)





label_basic_operations2 = ttk.Label(frame_grid_right, text="Basic Operations", anchor="center",font=("Arial", 15, "bold"), background=color_right)
addToRightGrid(label_basic_operations2, colspan=4)

label_border = ttk.Label(frame_grid_right, text="Border and padding adjustments", anchor="center", background=color_right)
addToRightGrid(label_border, colspan=4, pady = 10)
border_dropdown = ttk.Combobox(frame_grid_right,
                            values=list(border_types.keys()),
                            state="readonly")
border_dropdown.set("None")
addToRightGrid(border_dropdown, colspan=2, sticky="nswe")
border_dropdown.bind('<<ComboboxSelected>>', borderChange)
container_padding = tk.Frame(frame_grid_right, background=color_right)
label_padding = ttk.Label(container_padding, text="Padding: 0", background=color_right)
label_padding.pack()
padding_slider = ttk.Scale(container_padding, 
                        from_=0, to=100, 
                        orient=tk.HORIZONTAL, command= lambda e: label_padding.config(text = f"Padding: {int(float(e))}"))
padding_slider.bind("<ButtonRelease-1>", paddingChange)
padding_slider.pack(expand=True, fill="x")
addToRightGrid(container_padding, colspan=2)

label_mathematical_operations = ttk.Label(frame_grid_right, text="Mathematical Operations", anchor="center",font=("Arial", 15, "bold"), background=color_right)
addToRightGrid(label_mathematical_operations, colspan=4, pady = 10)
button_and = ttk.Button(frame_grid_right, text="AND", command=lambda : binaryOperation("AND"))
button_or = ttk.Button(frame_grid_right, text="OR", command=lambda : binaryOperation("OR"))
button_xor = ttk.Button(frame_grid_right, text="XOR", command=lambda : binaryOperation("XOR"))
button_not = ttk.Button(frame_grid_right, text="NOT", command=lambda : binaryOperation("NOT"))
addToRightGrid(button_and)
addToRightGrid(button_or)
addToRightGrid(button_xor)
addToRightGrid(button_not)
def pixelOperation(operation):
    global image
    img = image
    value = 50
    scalar_matrix = np.full(img.shape, (value, value, value), dtype=np.uint8)

    # Perform the selected operation
    if operation == "ADD":
        img = cv2.add(img, scalar_matrix)  # Add value to all pixels
    elif operation == "SUB":
        img = cv2.subtract(img, scalar_matrix)  # Subtract value from all pixels
    elif operation == "MUL":
        img = cv2.multiply(img, 1.5)  # Multiply each pixel by a scalar
    elif operation == "DIV":
        img = cv2.divide(img, 2)  # Divide each pixel by a scalar
    else:
        return    
    image = img
    canvas.show_image(image)

button_add = ttk.Button(frame_grid_right, text="ADD", command=lambda : pixelOperation("ADD"))
button_sub = ttk.Button(frame_grid_right, text="SUB", command=lambda : pixelOperation("SUB"))
button_mul = ttk.Button(frame_grid_right, text="MUL", command=lambda : pixelOperation("MUL"))
button_div = ttk.Button(frame_grid_right, text="DIV", command=lambda : pixelOperation("DIV"))
addToRightGrid(button_add)
addToRightGrid(button_sub)
addToRightGrid(button_mul)
addToRightGrid(button_div)



label_transforms = ttk.Label(frame_grid_right, text="Transforms and Filtering", anchor="center",font=("Arial", 15, "bold"), background=color_right)
addToRightGrid(label_transforms, colspan=4, pady = 10)
select_box_fourier = ttk.Combobox(frame_grid_right,
                            values=["FFT", "DFT"],
                            state="readonly")
select_box_fourier.set("FFT")
label_fourier = ttk.Label(frame_grid_right, text="Fourier transformation", anchor="center", background=color_right)
button_fourier = ttk.Button(frame_grid_right, text="Apply", command=fourier_transform)
addToRightGrid(label_fourier, colspan=4, pady = (0,10))
addToRightGrid(select_box_fourier, pady = (0,10), colspan=2, sticky="nsew")
addToRightGrid(button_fourier, pady = (0,10), colspan=2, padx = 10)

label_spatial = ttk.Label(frame_grid_right, text="Spatial Filters", anchor="center", background=color_right)
addToRightGrid(label_spatial, colspan=4, pady = (0,10))

select_box_blur = ttk.Combobox(frame_grid_right,
                            values=["Mean Filter", "Gaussian Filter", "Median Filter","None"],
                            state="readonly")
select_box_blur.set("Mean Filter")
button_blur = ttk.Button(frame_grid_right, text="Apply", command=apply_blur_filter)
addToRightGrid(select_box_blur, pady = (0,10), colspan=2, sticky="nsew")
addToRightGrid(button_blur, pady = (0,10), colspan=2, padx = 10)

label_edge = ttk.Label(frame_grid_right, text="Edge Detection", anchor="center", background=color_right)
select_box_edge = ttk.Combobox(frame_grid_right,
                            values=["Sobel", "Laplacian", "Canny", "None"],
                            state="readonly")
select_box_edge.set("Sobel")
select_box_edge.grid(column=0, row=0)
button_edge = ttk.Button(frame_grid_right, text="Apply", command= applyEdgeDetection)
addToRightGrid(label_edge, colspan=4, pady = (0,10))
addToRightGrid(select_box_edge, pady = (0,10), colspan=2, sticky="nsew")
addToRightGrid(button_edge, pady = (0,10), colspan=2, padx = 10)

label_enhancement = ttk.Label(frame_grid_right, text="Transforms and Filtering", anchor="center",font=("Arial", 15, "bold"), background=color_right)
addToRightGrid(label_enhancement, colspan=4, pady = 10)
button_histogram = ttk.Button(frame_grid_right, text="Histogram Equalization", command=apply_histogram_equalization)
button_contrast_stretch = ttk.Button(frame_grid_right, text="Contrast stretching", command=contrast_stretching)
addToRightGrid(button_histogram, pady = (0,10), colspan=2)
addToRightGrid(button_contrast_stretch, pady = (0,10), colspan=2)



label_gamma = ttk.Label(frame_grid_right, text="Gamma: 1.0", anchor="center", background=color_right)
gamma_slider = ttk.Scale(frame_grid_right,
                        from_=0.1, to=3.0,
                        orient=tk.HORIZONTAL, command=lambda value: label_gamma.config(text=f"Gamma: {float(value):.1f}"))
gamma_slider.bind("<ButtonRelease-1>", gammaChange)
gamma_slider.set(1.0)
addToRightGrid(label_gamma, colspan=4)
addToRightGrid(gamma_slider, colspan=4)

label_segmentation = ttk.Label(frame_grid_right, text="Image Segmentation", anchor="center",font=("Arial", 15, "bold"), background=color_right)
addToRightGrid(label_segmentation, colspan=4, pady = 10)

select_box_segmentation = ttk.Combobox(frame_grid_right,
                            values=["K-means", "Thresholding"],
                            state="readonly")
select_box_segmentation.set("K-means")
button_segmentation = ttk.Button(frame_grid_right, text="Apply", command=applySegmentation)
addToRightGrid(select_box_segmentation, colspan=2, sticky="nsew", pady=(0,10))
addToRightGrid(button_segmentation, colspan=2, padx = 10, pady=(0,10))

label_binary = ttk.Label(frame_grid_right, text="Binary Image Processing", anchor="center",font=("Arial", 15, "bold"), background=color_right)
addToRightGrid(label_binary, colspan=4, pady = 10)
button_dilation = ttk.Button(frame_grid_right, text="Dilation", command=lambda : imageMorph("Dilation"))
button_erosion = ttk.Button(frame_grid_right, text="Erosion", command=lambda : imageMorph("Erosion"))
button_opening = ttk.Button(frame_grid_right, text="Opening", command=lambda : imageMorph("Opening"))
button_closing = ttk.Button(frame_grid_right, text="Closing", command=lambda : imageMorph("Closing"))
button_boundary = ttk.Button(frame_grid_right, text="Extract Boundary", command=lambda : imageMorph("Boundary Extraction"))
button_skeletonize = ttk.Button(frame_grid_right, text="Skeletonize", command=lambda : imageMorph("Skeletonize"))
addToRightGrid(button_dilation)
addToRightGrid(button_erosion)
addToRightGrid(button_opening)
addToRightGrid(button_closing)
addToRightGrid(button_boundary, colspan=2)
addToRightGrid(button_skeletonize, colspan=2)

label_restoration = ttk.Label(frame_grid_right, text="Image Restoration", anchor="center",font=("Arial", 15, "bold"), background=color_right)
addToRightGrid(label_binary, colspan=4, pady = 10)
container_restoration = tk.Frame(frame_grid_right, background=color_right)
button_wiener = ttk.Button(container_restoration, text="Wiener", command=lambda : applyRestore("Wiener"))
button_wiener.grid(row=0,column=0, sticky="ew")
button_gaussian = ttk.Button(container_restoration, text="Gaussian", command=lambda : applyRestore("Gaussian"))
button_gaussian.grid(row=0,column=1, sticky="ew")
button_median = ttk.Button(container_restoration, text="Inpainting", command=lambda : applyRestore("Inpainting"))
button_median.grid(row=0, column=2, sticky="ew")
for i in range(3):container_restoration.columnconfigure(i, weight=1)
addToRightGrid(container_restoration, colspan=4)

label_matching = ttk.Label(frame_grid_right, text="Image Matching", anchor="center",font=("Arial", 15, "bold"), background=color_right)
addToRightGrid(label_matching, colspan=4, pady = 10)
container_matching = tk.Frame(frame_grid_right, background=color_right)

button_SIFT = ttk.Button(container_matching, text="Match Image SIFT", command=lambda:binaryOperation("MATCH"))
button_SIFT.grid(row=0, column=0, sticky="ew")
button_ORB = ttk.Button(container_matching, text="Match Image ORB", command=lambda:binaryOperation("MATCH_ORB"))
button_ORB.grid(row=0, column=1, sticky="ew")
button_template = ttk.Button(container_matching, text="Match Template", command=lambda:binaryOperation("TEMPLATE"))
button_template.grid(row=0, column=2, sticky="ew")
for i in range(3):container_matching.columnconfigure(i, weight=1)
addToRightGrid(container_matching, colspan=4)

label_compress = ttk.Label(frame2, text="Compression type", background=color_middle)
label_compress.pack()
frame_compress = tk.Frame(frame2, background=color_middle)
frame_compress.columnconfigure(0, weight=1, pad=10)
frame_compress.columnconfigure(1, weight=1, pad=10)
frame_compress.columnconfigure(2, weight=1, pad=10)
radio_compressNone = ttk.Radiobutton(frame_compress, text="None", variable=compress_type, value="None")
radio_compressNone.grid(column=0, row=0)
radio_compressDCT = ttk.Radiobutton(frame_compress, text="DCT", variable=compress_type, value="DCT")
radio_compressDCT.grid(column=1, row=0)
radio_compressRLE = ttk.Radiobutton(frame_compress, text="RLE", variable=compress_type, value="RLE")
radio_compressRLE.grid(column=2, row=0)
frame_compress.pack(pady=10)
compress_type.set("None")
button_save = ttk.Button(frame2, text="Save", command=save)
button_save.pack(pady=(0,10))
label_canvas2 = ttk.Label(frame2, text="Image 2", background="white")
canvas2 = ImageCanvas(frame2)


root.mainloop()