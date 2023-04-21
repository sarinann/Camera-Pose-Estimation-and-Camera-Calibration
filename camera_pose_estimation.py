import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

#declaration the matrices
theta_increment = np.pi/180
roll_values = []
pitch_values = []
yaw_values = []
x_translation = []
y_translation = []
z_translation = []

number_of_frames = []

#Reading the video
capture = cv.VideoCapture('camera_pose.avi')
while True:

    bool_value, frame = capture.read()
    if not bool_value:
        break
    resize_frame = cv.resize(frame,(0,0), fx=0.5, fy=0.5)
    #Gray filter
    gray_frame = cv.cvtColor(resize_frame, cv.COLOR_BGR2GRAY)
    #Gaussian Blur
    blurred_frame = cv.GaussianBlur(gray_frame, (5,5), 0)    
    # Create a kernel for morphology operations
    kernel = np.ones((5, 5), np.uint8)
    # Perform morphology first time
    closing_frame1 = cv.morphologyEx(blurred_frame, cv.MORPH_CLOSE, kernel)
    mask = cv.inRange(closing_frame1, 200, 255)
    # Perform morphology second time
    closing_frame = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)   
    edges_detected = cv.Canny(closing_frame, 50, 150)

    h, w = edges_detected.shape
    maximum_distance = int(np.sqrt(h**2 + w**2))
    accumulator_mat = np.zeros((2*maximum_distance, 180))

    for Y in range(h):
        for X in range(w):
            if edges_detected[Y][X] != 0:
                for theta_value in range(0, 180):
                    rho = int(X*np.cos(theta_value*theta_increment) + Y*np.sin(theta_value*theta_increment))
                    accumulator_mat[rho + maximum_distance][theta_value] += 1

    detect_lines = []
    #This helps in getting the peak values
    for i in range(4):
        maximum_freq = -1
        for i in range(4):
            maximum_value = np.max(accumulator_mat)
            maximum_value_index = np.argwhere(accumulator_mat == maximum_value)
            for j in maximum_value_index:
                freq = accumulator_mat[j[0], j[1]]
                if freq> maximum_freq:
                    maximum_freq = freq
                    rho = j[0] - maximum_distance
                    theta_value = j[1]
        detect_lines.append((rho, theta_value))

        peaks = np.unravel_index(np.argmax(accumulator_mat), accumulator_mat.shape)
        accumulator_mat[peaks[0], peaks[1]] = 0

        for j in range(-10, 10):
            for k in range(-10, 10):
                if rho+j+maximum_distance >=0 and rho+j+maximum_distance < 2*maximum_distance and theta_value+k < 180:
                    accumulator_mat[rho+j+maximum_distance][theta_value+k] = 0

    for lines in detect_lines:
        rho, theta_value = lines
        a_value = np.cos(theta_value*theta_increment)
        inter_c = np.sin(theta_value*theta_increment)
        x_0 = a_value*rho
        y_0 = inter_c*rho
        x = int(x_0 + 800*(-inter_c))
        y = int(y_0 + 800*a_value)
        x_2 = int(x_0 - 800*(-inter_c))
        y_2 = int(y_0 - 800*a_value)
        cv.line(resize_frame, (x, y), (x_2, y_2), (0, 0, 255), 2)

    #displaying the lines
    line_slope = []
    line_intercpt = []

    for rho, theta_value in detect_lines:
        slope_m = -np.cos(np.radians(theta_value)) / np.sin(np.radians(theta_value))
        inter_c = rho / np.sin(np.radians(theta_value))
        line_slope.append(slope_m)
        line_intercpt.append(inter_c)
    line_intersections = []

    for i in range(len(detect_lines)):
        for j in range(i+1, len(detect_lines)):
            m1, c1 = line_slope[i], line_intercpt[i]
            m2, c2 = line_slope[j], line_intercpt[j]
            X = (c2 -c1) / (m1 - m2)
            Y = m1*X + c1
            line_intersections.append((X, Y))

    line_intersections.sort(key = lambda point: point[1], reverse=True)
    line_intersections = line_intersections[:4]

    for pt in line_intersections:
        X, Y = pt
        if not (float('-inf') < X < float('inf') and float('-inf') < Y < float('inf')):
            continue
        cv.circle(frame, (int(X), int(Y)), 5, (0,255,0), -1)

    #creating the numpy array
    line_intersections = np.array(line_intersections)
    real_paper_coordinates = np.array([[0,0], [21.6, 0], [0, 27.9], [21.6, 27.9]])

    A = []
    for i in range(4):
        x, y = real_paper_coordinates[i]
        u, v = line_intersections[i]
        A.append([x, y, 1, 0, 0, 0, -x*u, -y*v, -u])
        A.append([0, 0, 0, x, y, 1, -x*v, -y*v, -v])  

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)    
    H = Vt[-1].reshape((3, 3))
    best_H = H / H[2, 2]
    #Modifying intrinsic matrix
    intrinsic_mat = np.array([[1.38e+03/2, 0, 9.46e+02/2],
                  [0, 1.38e+03/2, 5.27e+02/2],
                  [0, 0, 1]])
    
    # Rotation
    intrinsic_mat_inverse = np.linalg.inv(intrinsic_mat)
    ext_matrix = intrinsic_mat_inverse @ best_H
    H_1 = ext_matrix[:, 0]
    H_2 = ext_matrix[:, 1]
    H_3 = ext_matrix[:, 2]
    lamb = 1 / np.linalg.norm(H_1)
    R_1 = lamb * H_1
    R_2 = lamb * H_2
    R_3 = np.cross(R_1, R_2)

    rot = np.column_stack((R_1, R_2, R_3))

    trans = lamb * H_3
    x_translation.append(trans[0])
    y_translation.append(trans[1])
    z_translation.append(trans[2])

    c_matrix = np.column_stack((np.dot(intrinsic_mat, rot), trans))
    print(c_matrix)

    c_pos = -np.dot(np.linalg.inv(c_matrix[:, :-1]), c_matrix[:, -1])

    R = c_matrix[:3,:3]

    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    
    roll_values.append(roll)
    pitch_values.append(pitch)
    yaw_values.append(yaw)

    cv.imshow('frame', edges_detected)  
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

add_color = ['r'] + ['b'] * (len(roll_values) - 1)
for i in range(len(roll_values)):
    number_of_frames.append(i)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.set_title('Plot')
ax.scatter(number_of_frames, x_translation, y_translation, z_translation, c = add_color)
plt.show()

plt.plot(number_of_frames, roll_values)
plt.title('Roll value variation over each frame')
plt.xlabel('frames')
plt.ylabel('roll')
plt.show()

plt.plot(number_of_frames, pitch_values)
plt.title('Pitch value variation over each frame')
plt.xlabel('frames')
plt.ylabel('pitch')
plt.show()

plt.plot(number_of_frames, yaw_values)
plt.title('Yaw value variation over each frame')
plt.xlabel('frames')
plt.ylabel('yaw')
plt.show()

capture.release()
cv.destroyAllWindows()