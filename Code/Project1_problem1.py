import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Creating empty lists to store a and y coordinates
x_centers = []
y_centers = []

# Read the video file using OpenCV
cap = cv.VideoCapture("/home/ab/enpm673/project1/ball.mov")

# loop through each frame
while (True):
  
    # Read the frame
    ret, frame = cap.read()

    if ret == True: 
        
        # Convert the BGR colorspace to HSV colorspace
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # Hue is colour, saturation is intensity, value is brightness

        # Define the lower and upper range for red pixels in hsv
        lower_range = np.array([0, 50, 50 ])
        upper_range = np.array([2, 255, 255])

        # mask sets pixel falling in the range as 255 i.e. white and remaining pixels as 0 i.e. black
        mask = cv.inRange(hsv, lower_range, upper_range)

        # Define a kernel
        kernel = np.ones((5, 5), np.uint8)

        # Opening operation
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)  # erosion - dilation

        # Closing operation
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel) # dilation - erosion

        # Create array of all the white pixels i.e. value 1
        non_zero = np.array(np.nonzero(mask))  

        # Find min and max of x and y coordinates of red pixels
        if np.any(non_zero):

            min_x = np.min(non_zero[1])
            min_y = np.min(non_zero[0])
            max_x = np.max(non_zero[1])
            max_y = np.max(non_zero[0])

            # Take mean to calculate center pixels
            center_x = int((max_x + min_x) / 2)
            x_centers.append(center_x)

            center_y = int((max_y + min_y )/ 2)
            y_centers.append(center_y)

            # Draw center at the center of the ball in each frame
            center = (center_x, center_y)

            cv.circle(frame, center, 5, (0, 255, 0), -1) 

        cv.imshow("Red Ball Center", frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break 
 
# Plot the graph of trajectory of the ball
plt.scatter(x_centers, y_centers)  
plt.gca().invert_yaxis()  

cv.destroyAllWindows()

plt.show()

# Standard equation of parabola is y = ax*2 + bx + c

x_squares = []
for x in x_centers:
    x_squares.append(x**2) 

x_centers = np.array(x_centers)
y_centers = np.array(y_centers)
x_squares = np.array(x_squares)

length = len(x_centers)

# Using least square method
X = np.column_stack((x_squares, x_centers, np.ones(length)))

inverse = np.linalg.inv(X.T @ X)

B = inverse @ X.T @ y_centers

a, b, c = B

plt.scatter(x_centers, y_centers, c='r')
plt.plot(x_centers, a*x_squares + b*x_centers +c)
plt.gca().invert_yaxis()  
plt.show()

x = None
y = None

equation= f'y = {a}*x**2 + {b}*x + {c}'
print("Equation of the curve is: ")
print(equation)

# using the quadtric equation/ roots of ax*2 + bx + c = 0 is given by x = (-b - sqrt(b*2 - 4ac)/2*a)
landing_y = y_centers[0] + 300
delta = np.sqrt(b**2 - 4*a*(c - landing_y))
landing_x = (-b + delta) / (2*a)

print("The x coordinate of landing spot is: " , landing_x)