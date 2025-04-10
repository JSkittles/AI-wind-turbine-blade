import numpy as np
import matplotlib.pyplot as plt


# X and Y coordinates of the airfoil
x_coords = np.array([0.0000, 0.0012, 0.0050, 0.0112, 0.0198, 0.0308, 0.0443, 0.0600, 0.0779, 0.0980, 0.1201, 0.1443, 0.1703, 0.1981, 0.2275, 0.2584, 0.2907, 0.3243, 0.3589, 0.3945, 0.4309, 0.4680, 0.5055, 0.5434, 0.5814, 0.6194, 0.6573, 0.6948, 0.7319, 0.7683, 0.8039, 0.8385, 0.8721, 0.9044, 0.9353, 0.9647, 0.9925, 1.0000])

# Split x and y into upper and lower surfaces based on y coordinates
upper_surface_y = np.array([-0.047, -0.002, 0.017, 0.025, 0.034, 0.044, 0.055, 0.066, 0.078, 0.090, 0.103, 0.116, 0.128, 0.140, 0.152, 0.164, 0.175, 0.186, 0.194, 0.200, 0.201, 0.201, 0.196, 0.190, 0.180, 0.170, 0.157, 0.143, 0.127, 0.110, 0.092, 0.075, 0.056, 0.039, 0.021, 0.006, -0.012, -0.027])


lower_surface_y = np.array([-0.052, -0.070, -0.079, -0.091, -0.101, -0.111, -0.119, -0.128, -0.136, -0.145, -0.152, -0.160, -0.167, -0.174, -0.180, -0.186, -0.190, -0.194, -0.197, -0.200, -0.201, -0.201, -0.200, -0.197, -0.193, -0.188, -0.181, -0.172, -0.160, -0.149, -0.137, -0.127, -0.118, -0.110, -0.102, -0.096, -0.091, -0.062])
 # Lower surface is the negative of the y coordinates



plt.figure(figsize=(10, 5))
plt.plot(x_coords, upper_surface_y, label="Upper Surface", color="blue")
plt.plot(x_coords, lower_surface_y, label="Lower Surface", color="red")
#plt.scatter(1.0, 0.0055, color='red', s=100, label="Highlight") 
#plt.scatter(1.0, -0.0055, color='red', s=100, label="Highlight")  

plt.title("Optimized Airfoil Shape")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
plt.legend()
plt.grid()
plt.show()