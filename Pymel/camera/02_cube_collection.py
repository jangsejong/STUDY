import maya.cmds as cmds
import random
    
         
for n in range(25):
    cube, cubeShape = cmds.polyCube()
    x = random.randrange(-50, 50)
    y = random.randrange(-50, 50)
    z = random.randrange(-50, 50)
    cmds.xform(cube, t = (x,y,z))
