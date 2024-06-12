import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI rendering
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw
import time
import smbus2
import evdev
from evdev import InputDevice, ecodes
import threading
from collections import deque
import csv
import sys  # To use sys.exit() for a clean exit
from datetime import datetime  # Import datetime module

# Configuration dictionary with I2C addresses for each sensor and temperature points
config = {
    "FL": {"Inner": 0x50, "Center": 0x50, "Outer": 0x50},  # Front Left sensors
    "FR": {"Inner": 0x51, "Center": 0x51, "Outer": 0x51},  # Front Right sensors
    "RL": {"Inner": 0x52, "Center": 0x52, "Outer": 0x52},  # Rear Left sensors
    "RR": {"Inner": 0x53, "Center": 0x53, "Outer": 0x53},  # Rear Right sensors
    "TemperaturePoints": {
        "Blue": 20,          # Blue temperature point
        "GreenLow": 80,      # Lower bound of green temperature range
        "GreenHigh": 100,    # Upper bound of green temperature range
        "Red": 120           # Red temperature point
    },
    "SmoothingAlpha": 0.1,   # Alpha value for exponential smoothing
    "UpdateInterval": 0.1    # Update interval in seconds
}

# Initialize last known temperatures with None
last_known_temps = {tire: {"Inner": None, "Center": None, "Outer": None} for tire in config.keys() if tire not in ["TemperaturePoints", "SmoothingAlpha", "UpdateInterval"]}

# Initialize smoothed temperatures with None
smoothed_temps = {tire: {"Inner": None, "Center": None, "Outer": None} for tire in config.keys() if tire not in ["TemperaturePoints", "SmoothingAlpha", "UpdateInterval"]}

# Function to read temperature from an MLX90614 sensor
def read_temperature(sensor_address):
    bus = smbus2.SMBus(1)  # 1 indicates /dev/i2c-1
    try:
        # Read object temperature
        object_temp = bus.read_word_data(sensor_address, 0x07)
        
        # Convert the temperature data to Celsius
        object_temp = (object_temp * 0.02) - 273.15
        
        return object_temp
    except Exception as e:
        # In case of an error, return None
        return None
    finally:
        bus.close()

# Function to apply exponential smoothing
def smooth_temperature(last, current, alpha):
    if last is None:
        return current
    return alpha * current + (1 - alpha) * last

# Function to obtain tire temperature data from MLX90614 sensors and apply smoothing
def get_tire_data():
    alpha = config["SmoothingAlpha"]
    for position, sensors in config.items():
        if position not in ["TemperaturePoints", "SmoothingAlpha", "UpdateInterval"]:
            for sensor, address in sensors.items():
                temp = read_temperature(address)
                if temp is not None:
                    last_known_temps[position][sensor] = temp
                    if smoothed_temps[position][sensor] is None:
                        smoothed_temps[position][sensor] = temp
                    else:
                        smoothed_temps[position][sensor] = smooth_temperature(smoothed_temps[position][sensor], temp, alpha)

# Create a single figure and axes layout with reduced height for each subplot
fig = plt.figure(figsize=(4, 3.5), dpi=100)  # Adjust figure size and DPI
fig.patch.set_facecolor('black')  # Set figure background color to black
gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.4)  # Create a 2x2 grid for subplots
axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]  # Create subplots
for ax in axs:
    ax.set_facecolor('black')  # Set subplot background color to black
    ax.tick_params(colors='white')  # Set tick parameters to white

# Initialize image objects for each subplot
img_objs = [None] * 4
labels_visible = True
record_button_pressed = False
logging_active = False  # Track logging state
log_file = None  # Log file handle

# Function to create the heatmap plot as an image
def create_heatmap_image():
    global labels_visible, record_button_pressed
    tire_positions = ["FL", "FR", "RL", "RR"]
    
    # Extract temperature points from config
    blue_temp = config["TemperaturePoints"]["Blue"]
    green_low_temp = config["TemperaturePoints"]["GreenLow"]
    green_high_temp = config["TemperaturePoints"]["GreenHigh"]
    red_temp = config["TemperaturePoints"]["Red"]
    
    # Create a custom colormap
    colors = [
        (0.0, "darkblue"),    # Below blue_temp, darker blue
        ((blue_temp - blue_temp) / (red_temp - blue_temp), "blue"),  # Blue temperature point
        ((green_low_temp - blue_temp) / (red_temp - blue_temp), "#00FF00"),  # Lower bound of green temperature range
        ((green_high_temp - blue_temp) / (red_temp - blue_temp), "#00FF00"),  # Upper bound of green temperature range
        ((red_temp - blue_temp) / (red_temp - blue_temp), "red"),  # Red temperature point
        (1.0, "darkred")      # Above red_temp, darker red
    ]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    norm = plt.Normalize(vmin=blue_temp, vmax=red_temp)

    for i, (tire, ax) in enumerate(zip(tire_positions, axs)):
        # Retrieve smoothed temperatures, defaulting to 50 if None
        sensor_temps = [
            smoothed_temps[tire]["Inner"] if smoothed_temps[tire]["Inner"] is not None else 50,
            smoothed_temps[tire]["Center"] if smoothed_temps[tire]["Center"] is not None else 50,
            smoothed_temps[tire]["Outer"] if smoothed_temps[tire]["Outer"] is not None else 50
        ]
        
        if tire in ["FL", "RL"]:
            sensor_temps = sensor_temps[::-1]  # Reverse the data for left-hand tires
            labels = ['O', 'C', 'I']  # Labels for reversed data
        else:
            labels = ['I', 'C', 'O']  # Labels for normal data
        
        # Reshape and expand the temperature data for better visualization
        sensor_temps = np.array(sensor_temps).reshape(1, 3)
        sensor_temps = np.repeat(sensor_temps, 10, axis=0)

        if img_objs[i] is None:
            img_objs[i] = ax.imshow(sensor_temps, cmap=cmap, norm=norm, interpolation='bilinear', aspect='auto')
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(labels, color='white', fontsize=10 if labels_visible else 0)
            ax.xaxis.set_ticks_position('none')  # Remove the tick marks
            ax.set_yticks([])  # Remove y-ticks
            ax.set_title(tire, color='white', pad=7.5, fontsize=10 if labels_visible else 0)  # Reduced padding by 50%
        else:
            img_objs[i].set_data(sensor_temps)
            img_objs[i].set_clim(vmin=blue_temp, vmax=red_temp)
            ax.set_xticklabels(labels, color='white', fontsize=10 if labels_visible else 0)
            ax.set_title(tire, color='white', pad=7.5, fontsize=10 if labels_visible else 0)

        # Map label indexes to the correct keys in last_known_temps
        key_map = {'I': 'Inner', 'C': 'Center', 'O': 'Outer'}
        
        # Color the labels red if the sensor is missing
        xtick_labels = ax.get_xticklabels()
        for j, label in enumerate(labels):
            if last_known_temps[tire][key_map[label]] is None:
                xtick_labels[j].set_color('red')
    
    # Update the colorbar visibility based on labels_visible
    if not hasattr(create_heatmap_image, 'colorbar'):
        create_heatmap_image.colorbar = fig.colorbar(img_objs[0], ax=axs, location='right', fraction=0.05, pad=0.04)
        create_heatmap_image.colorbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(create_heatmap_image.colorbar.ax.yaxis.get_ticklabels(), color='white')
        create_heatmap_image.colorbar.set_label('Temperature (°C)', color='white')
    create_heatmap_image.colorbar.ax.set_visible(labels_visible)

    # Save the plot to a file and display it using PIL
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = Image.fromarray(img)
    img = img.convert("RGB")  # Convert to RGB format

    # Draw the record button on the image
    draw = ImageDraw.Draw(img)
    button_fill = (255, 140, 0) if record_button_pressed else (169, 169, 169)
    draw.ellipse([10, 10, 40, 40], fill=button_fill)  # Circle for the record button

    return img

# Function to convert an image to RGB565 format
def convert_to_rgb565(img):
    img = img.convert("RGB")
    r, g, b = img.split()
    
    r = np.array(r).astype(np.uint16)
    g = np.array(g).astype(np.uint16)
    b = np.array(b).astype(np.uint16)
    
    rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    return rgb565.tobytes()

# Function to write the image to the framebuffer
def write_to_framebuffer(img):
    # Resize and convert the image to match the framebuffer format
    img = img.resize((480, 320))
    img_data = convert_to_rgb565(img)

    with open("/dev/fb1", "wb") as fb:
        fb.write(img_data)

# Function to clear the framebuffer
def clear_framebuffer():
    img = Image.new("RGB", (480, 320), "black")
    img_data = convert_to_rgb565(img)

    with open("/dev/fb1", "wb") as fb:
        fb.write(img_data)

# Find the touchscreen input device
def find_touchscreen():
    devices = [InputDevice(path) for path in evdev.list_devices()]
    for device in devices:
        if 'touchscreen' in device.name.lower():
            return device
    raise RuntimeError('Touchscreen device not found')

touchscreen = find_touchscreen()

# Function to handle touchscreen events
def handle_touchscreen_events():
    global labels_visible, record_button_pressed, logging_active, log_file
    touchscreen.grab()  # Grab exclusive access to the touchscreen
    try:
        x, y = None, None  # Initialize x and y to None
        for event in touchscreen.read_loop():
            if event.type == ecodes.EV_ABS:
                if event.code == 0:  # ABS_X
                    x = event.value
                elif event.code == 1:  # ABS_Y
                    y = event.value
                
                if x is not None and y is not None:
                    # Convert touchscreen coordinates to image coordinates
                    img_x = int(x / 4096 * 320)
                    img_y = int(y / 4096 * 480)
                    print(f"Touch coordinates: ({img_x}, {img_y})")  # Debug print

                    # Check if the tap is within the record button area
                    record_button_x_start = 0
                    record_button_x_end = 50
                    record_button_y_start = 420
                    record_button_y_end = 480

                    if record_button_x_start <= img_x <= record_button_x_end and record_button_y_start <= img_y <= record_button_y_end:
                        record_button_pressed = not record_button_pressed  # Toggle record button state
                        logging_active = record_button_pressed  # Start/stop logging
                        if logging_active:
                            # Generate filename with start datetime
                            start_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
                            filename = f'tire_temps_{start_datetime}.csv'
                            log_file = open(filename, 'w', newline='')
                            log_writer = csv.writer(log_file)
                            # Write MoTeC-style header and channels
                            log_writer.writerow(['[Header]'])
                            log_writer.writerow(['File created on', start_datetime])
                            log_writer.writerow(['Logged by', 'Custom Tire Temperature Logger'])
                            log_writer.writerow([])
                            log_writer.writerow(['[Channels]'])
                            log_writer.writerow([
                                'Time', 'Front Left Inner', 'Front Left Center', 'Front Left Outer', 
                                'Front Right Inner', 'Front Right Center', 'Front Right Outer', 
                                'Rear Left Inner', 'Rear Left Center', 'Rear Left Outer', 
                                'Rear Right Inner', 'Rear Right Center', 'Rear Right Outer'
                            ])
                            log_writer.writerow(['s', '°C', '°C', '°C', '°C', '°C', '°C', '°C', '°C', '°C', '°C', '°C', '°C'])
                        else:
                            if log_file:
                                log_file.close()
                    else:
                        # Check if the tap is within the color bar area
                        colorbar_x_start = 0
                        colorbar_x_end = 320
                        colorbar_y_start = 0
                        colorbar_y_end = 100

                        if colorbar_x_start <= img_x <= colorbar_x_end and colorbar_y_start <= img_y <= colorbar_y_end:
                            labels_visible = not labels_visible  # Toggle label visibility

                    # Reset x and y to None after processing
                    x, y = None, None
    except KeyboardInterrupt:
        pass
    finally:
        touchscreen.ungrab()  # Release exclusive access to the touchscreen

# Start the touchscreen event handler in a separate thread
touchscreen_thread = threading.Thread(target=handle_touchscreen_events)
touchscreen_thread.start()

# Main loop
running = True
frame_times = deque(maxlen=100)  # Store the last 100 frame times for smoothing
start_time = time.time()

try:
    while running:
        frame_start_time = time.time()
        
        # Update the frame times deque
        frame_times.append(frame_start_time)
        if len(frame_times) > 1:
            ups = len(frame_times) / (frame_times[-1] - frame_times[0])
        else:
            ups = 0.0

        get_tire_data()  # Update the tire temperature data

        # Log data if logging is active
        if logging_active and log_file:
            log_writer = csv.writer(log_file)
            timestamp = time.time() - start_time
            log_writer.writerow([
                f"{timestamp:.3f}", 
                smoothed_temps['FL']['Inner'], smoothed_temps['FL']['Center'], smoothed_temps['FL']['Outer'],
                smoothed_temps['FR']['Inner'], smoothed_temps['FR']['Center'], smoothed_temps['FR']['Outer'],
                smoothed_temps['RL']['Inner'], smoothed_temps['RL']['Center'], smoothed_temps['RL']['Outer'],
                smoothed_temps['RR']['Inner'], smoothed_temps['RR']['Center'], smoothed_temps['RR']['Outer']
            ])

        img = create_heatmap_image()
        write_to_framebuffer(img)
        
        elapsed_time = time.time() - frame_start_time
        time.sleep(max(config["UpdateInterval"] - elapsed_time, 0))  # Update at specified interval

except KeyboardInterrupt:
    running = False

finally:
    clear_framebuffer()  # Clear the framebuffer on exit
    if log_file:
        log_file.close()  # Close the log file on exit
    sys.exit()  # Ensure the program exits

# Wait for the touchscreen thread to finish
touchscreen_thread.join()
