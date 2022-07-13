import maya.cmds as cmds          
cameras = cmds.ls(type ='camera')  
for each_camera in cameras:
    parent = cmds.listRelatives(each_camera, parent=True)
    position = cmds.xform(parent, q=True, translation=True)
    print each_camera, "is at", position
