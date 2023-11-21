import magpylib as magpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import magpylib.current as current
from mpl_toolkits.mplot3d import Axes3D


currentZ, diffZ = 11.69, 0
currentX, diffX = -10.4, 0
currentY, diffY = 10.5, 0
currentC = 1000
geoNorth = 20140 * 1e-6
geoEast = 103 * 1e-6
geoDown = 43325 * 1e-6
BEarth = np.array([0.707107 * (geoNorth - geoEast), -0.707107 * (geoNorth + geoEast), -geoDown])
scaling = 20


# mind that when the person is standing in front of the experiment and facing free space experiment, x is to
# the right, y to the front, z to the top; all currents are assumed to be righthand going with value specified above


class fieldOneCom:
    def __init__(self, src, grid):
        self.src = src
        self.grid = grid

    def computeOne(self):
        return self.src.getB(self.grid)


class displayB:
    def __init__(self, totalB, grid):
        self.totalB = totalB
        self.grid = grid

    def show3dRegular(self):
        strength = np.sum(self.totalB ** 2, axis=0)
        strength = np.sqrt(strength)
        norm = Normalize()
        norm.autoscale(strength)
        colormap = cm.viridis_r
        totalB = self.totalB
        baseG = self.grid
        fig = plt.figure(figsize=(10, 8))  # Adjust the figsize parameter
        ax = fig.add_subplot(111, projection='3d')
        # print('debug')
        for i in range(np.shape(baseG)[0]):
            for j in range(np.shape(baseG)[1]):
                for k in range(np.shape(baseG)[2]):
                    quiver = ax.quiver(baseG[i][j][k][0], baseG[i][j][k][1], baseG[i][i][k][2], totalB[i][j][k][0],
                                       totalB[i][j][k][1], totalB[i][j][k][2],
                                       normalize=True,
                                       color=colormap(
                                           norm(np.sqrt(
                                               totalB[i][j][k][0] ** 2 + totalB[i][j][k][1] ** 2 + totalB[i][j][
                                                   k][2] ** 2))))
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        # print('here')
        cbar = fig.colorbar(sm, ax=ax, label='Strength')

        # Set plot limits
        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])
        ax.set_zlim([-100, 100])

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # Improve layout
        plt.tight_layout()

        # Show the plot
        plt.show()

    def show2dStreamZ(self,z):
        strength = np.sum(self.totalB ** 2, axis=0)
        strength = np.sqrt(strength)
        norm = Normalize()
        norm.autoscale(strength)
        colormap = "spring_r"
        totalB = self.totalB
        fig = plt.figure(figsize=(10, 8))  # Adjust the figsize parameter
        ax = fig.add_subplot(111)

        ax.streamplot(self.grid[:, :, z, 0].transpose(1,0), self.grid[:, :, z, 1].transpose(1,0), totalB[:, :, z, 0].transpose(1,0), totalB[:, :, z, 1].transpose(1,0),
                      density=1.5, color=np.log(np.linalg.norm(totalB[:, :, z], axis=2)), linewidth=1, cmap=colormap)

        #plt.tight_layout()
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, label='Strength')
        plt.show()

    def show2dRegularZ(self,z):
        strength = np.sum(self.totalB[:,:,z,:] ** 2, axis=0)
        strength = np.sqrt(strength)
        norm = Normalize()
        norm.autoscale(strength)
        colormap = cm.viridis_r
        totalB = self.totalB[:,:,z,:]
        baseG = self.grid[:,:,z,:]
        fig = plt.figure(figsize=(10, 8))  # Adjust the figsize parameter
        ax = fig.add_subplot(111)
        for i in range(np.shape(baseG)[0]):
            for j in range(np.shape(baseG)[1]):
                quiver = ax.quiver(baseG[i][j][0], baseG[i][j][1], totalB[i][j][0],
                                   totalB[i][j][1],
                                   color=colormap(norm(np.sqrt(
                                    totalB[i][j][0] ** 2 + totalB[i][j][1] ** 2))))
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label='Strength')

        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Improve layout
        plt.tight_layout()

        # Show the plot
        plt.show()

    def show2dStreamX(self,x):
        strength = np.sum(self.totalB ** 2, axis=0)
        strength = np.sqrt(strength)
        norm = Normalize()
        norm.autoscale(strength)
        colormap = "spring_r"
        totalB = self.totalB
        fig = plt.figure(figsize=(10, 8))  # Adjust the figsize parameter
        ax = fig.add_subplot(111)
        ax.streamplot(self.grid[x, :, :, 1].transpose(1,0), self.grid[x, :, :, 2].transpose(1,0), totalB[x, :, :, 1].transpose(1,0), totalB[x, :, :, 2].transpose(1,0),
                      density=1.5, color=np.log(np.linalg.norm(totalB[x, :, :], axis=2)), linewidth=1, cmap=colormap)

        #plt.tight_layout()
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, label='Strength')
        plt.show()

    def show2dRegularX(self,x):
        strength = np.sum(self.totalB[x,:,:,:] ** 2, axis=0)
        strength = np.sqrt(strength)
        norm = Normalize()
        norm.autoscale(strength)
        colormap = cm.viridis_r
        totalB = self.totalB[x,:,:,:]
        baseG = self.grid[x,:,:,:]
        fig = plt.figure(figsize=(10, 8))  # Adjust the figsize parameter
        ax = fig.add_subplot(111)
        for i in range(np.shape(baseG)[0]):
            for j in range(np.shape(baseG)[1]):
                quiver = ax.quiver(baseG[i][j][1], baseG[i][j][2], totalB[i][j][1],
                                   totalB[i][j][2],
                                   color=colormap(norm(np.sqrt(
                                    totalB[i][j][2] ** 2 + totalB[i][j][1] ** 2))))
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label='Strength')

        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])

        ax.set_xlabel('Y-axis')
        ax.set_ylabel('Z-axis')

        # Improve layout
        plt.tight_layout()

        # Show the plot
        plt.show()

    def show2dStreamY(self,y):
        strength = np.sum(self.totalB ** 2, axis=0)
        strength = np.sqrt(strength)
        norm = Normalize()
        norm.autoscale(strength)
        colormap = "spring_r"
        totalB = self.totalB
        fig = plt.figure(figsize=(10, 8))  # Adjust the figsize parameter
        ax = fig.add_subplot(111)
        ax.streamplot(self.grid[:, y, :, 0].transpose(1,0), self.grid[:, y, :, 2].transpose(1,0), totalB[:, y, :, 0].transpose(1,0), totalB[:, y, :, 2].transpose(1,0),
                      density=1.5, color=np.log(np.linalg.norm(totalB[:, y, :], axis=2)), linewidth=1, cmap=colormap)

        #plt.tight_layout()
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, label='Strength')
        plt.show()

    def show2dRegularY(self,y):
        strength = np.sum(self.totalB[:,y,:,:] ** 2, axis=0)
        strength = np.sqrt(strength)
        norm = Normalize()
        norm.autoscale(strength)
        colormap = cm.viridis_r
        totalB = self.totalB[:,y,:,:]
        baseG = self.grid[:,y,:,:]
        fig = plt.figure(figsize=(10, 8))  # Adjust the figsize parameter
        ax = fig.add_subplot(111)
        for i in range(np.shape(baseG)[0]):
            for j in range(np.shape(baseG)[1]):
                quiver = ax.quiver(baseG[i][j][0], baseG[i][j][2], totalB[i][j][0],
                                   totalB[i][j][2],
                                   color=colormap(norm(np.sqrt(
                                    totalB[i][j][0] ** 2 + totalB[i][j][2] ** 2))))
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label='Strength')

        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Z-axis')

        # Improve layout
        plt.tight_layout()

        # Show the plot
        plt.show()


srcZ1 = magpy.current.Line(
    current=currentZ + diffZ / 2,
    vertices=((-240, -240, 110), (240, -240, 110), (240, 240, 110), (-240, 240, 110), (-240, -240, 110)),
)

srcZ2 = magpy.current.Line(
    current=currentZ - diffZ / 2,
    vertices=((-240, -240, -110), (240, -240, -110), (240, 240, -110), (-240, 240, -110), (-240, -240, -110)),
)

srcX1 = magpy.current.Line(
    current=currentX + diffX / 2,
    vertices=((-240, -240, 110), (-240, -240, -110), (-240, 240, -110), (-240, 240, 110), (-240, -240, 110)),
)

srcX2 = magpy.current.Line(
    current=currentX - diffX / 2,
    vertices=((240, -240, 110), (240, -240, -110), (240, 240, -110), (240, 240, 110), (240, -240, 110)),
)

srcY1 = magpy.current.Line(
    current=currentY + diffY / 2,
    vertices=((-240, -240, 110), (240, -240, 110), (240, -240, -110), (-240, -240, -110), (-240, -240, 110)),
)

srcY2 = magpy.current.Line(
    current=currentY - diffY / 2,
    vertices=((-240, 240, 110), (240, 240, 110), (240, 240, -110), (-240, 240, -110), (-240, 240, 110)),
)

srcC1 = magpy.current.Line(
    current=currentC,
    vertices=((-75, -80, 50), (75, -80, 50), (75, -80, -50), (-75, -80, -50), (-75, -80, 50)),
)

srcC2 = magpy.current.Line(
    current=-currentC,
    vertices=((-75, 80, 50), (75, 80, 50), (75, 80, -50), (-75, 80, -50), (-75, 80, 50)),
)

grid = np.mgrid[-100:100:21j, -100:100:21j, -100:100:21j].transpose((1, 2, 3, 0))

Z1 = fieldOneCom(srcZ1, grid)
Z2 = fieldOneCom(srcZ2, grid)
Y1 = fieldOneCom(srcY1, grid)
Y2 = fieldOneCom(srcY2, grid)
X1 = fieldOneCom(srcX1, grid)
X2 = fieldOneCom(srcX2, grid)
C1 = fieldOneCom(srcC1, grid)
C2 = fieldOneCom(srcC2, grid)

fieldZ1 = Z1.computeOne()
fieldZ2 = Z2.computeOne()
fieldY1 = Y1.computeOne()
fieldY2 = Y2.computeOne()
fieldX1 = X1.computeOne()
fieldX2 = X2.computeOne()
fieldC1 = C1.computeOne()
fieldC2 = C2.computeOne()

B = fieldZ1 + fieldZ2 + fieldY1 + fieldY2 + fieldX1 + fieldX2 + fieldC1 + fieldC2
B = B + BEarth[np.newaxis, np.newaxis, np.newaxis, :]
B = B * scaling

expVisual = displayB(B, grid)
expVisual.show2dStreamZ(10)
