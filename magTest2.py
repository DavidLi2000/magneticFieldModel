import magpylib as magpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

currentZ = 11.7
currentX = -6.8
currentY = 6.8
currentC = 1000
geoNorth = 20140*1e-6
geoEast = 103*1e-6
geoDown = 43325*1e-6
BEarth = np.array([0.707107*(geoNorth-geoEast),-0.707107*(geoNorth+geoEast),-geoDown])
scaling = 20

#mind that when the person is standing in front of the experiment and facing free space experiment, x is to
#the right, y to the front, z to the top; all currents are assumed to be righthand going with value specified above


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
        strength = np.sum(self.totalB**2, axis=0)
        strength = np.sqrt(strength)
        norm = Normalize()
        norm.autoscale(strength)
        colormap = cm.spring_r
        totalB = np.transpose(self.totalB, axes=(3, 0, 1, 2))
        baseG = np.transpose(self.grid, axes=(3, 0, 1, 2))
        fig = plt.figure(figsize=(10, 8))  # Adjust the figsize parameter
        ax = fig.add_subplot(111, projection='3d')
        #print('debug')
        for i in range(np.shape(baseG[0])[0]):
            for j in range(np.shape(baseG[0])[1]):
                for k in range(np.shape(baseG[0])[2]):
                    quiver = ax.quiver(baseG[0][i][j][k], baseG[1][i][j][k], baseG[2][i][j][k], totalB[0][i][j][k],
                                       totalB[1][i][j][k], totalB[2][i][j][k],
                                       normalize=False,
                                       color=colormap(
                                           norm(np.sqrt(totalB[0][i][j][k] ** 2 + totalB[1][i][j][k] ** 2 + totalB[2][i][j][k] ** 2))))
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        #print('here')
        cbar = fig.colorbar(sm, ax=ax, label='Strength')

        # Set plot limits
        ax.set_xlim([-500, 500])
        ax.set_ylim([-500, 500])
        ax.set_zlim([-500, 500])

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # Improve layout
        plt.tight_layout()

        # Show the plot
        plt.show()

srcZ1 = magpy.current.Line(
    current=currentZ,
    vertices=((-240, -240, 110), (240, -240, 110), (240, 240, 110), (-240, 240, 110), (-240, -240, 110)),
)

srcZ2 = magpy.current.Line(
    current=currentZ,
    vertices=((-240, -240, -110), (240, -240, -110), (240, 240, -110), (-240, 240, -110), (-240, -240, -110)),
)

srcX1 = magpy.current.Line(
    current=currentX,
    vertices=((-240, -240, 110), (-240, -240, -110), (-240, 240, -110), (-240, 240, 110), (-240, -240, 110)),
)

srcX2 = magpy.current.Line(
    current=currentX,
    vertices=((240, -240, 110), (240, -240, -110), (240, 240, -110), (240, 240, 110), (240, -240, 110)),
)

srcY1 = magpy.current.Line(
    current=currentY,
    vertices=((-240, -240, 110), (240, -240, 110), (240, -240, -110), (-240, -240, -110), (-240, -240, 110)),
)

srcY2 = magpy.current.Line(
    current=currentY,
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



grid = np.mgrid[-500:500:11j, -500:500:11j, -500:500:11j].transpose((1, 2, 3, 0))

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
print(B[:][5][5][5])
B = B*scaling

expVisual = displayB(B,grid)
expVisual.show3dRegular()
