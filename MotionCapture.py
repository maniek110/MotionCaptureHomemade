from PoseModule import poseDetector
import cv2
import sys
import fbx


def main():
    scene = fbx.FbxScene
    animation_stack = fbx.FbxAnimStack(scene, "AnimationStack")
    animation_layer = fbx.FbxAnimLayer(scene, "Base Layer")

    movie = 0
    if len(sys.argv) >= 2:
        movie = cv2.VideoCapture(sys.argv[1])
    else:
        movie = cv2.VideoCapture(0)
    detector=poseDetector(False,False,False,0.2,0.2)
    while movie.isOpened():
        success, image = movie.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
        image,results=detector.findPose(image,False)
        motion_data = {}
        for i, pose_landmarks in enumerate(results.pose_landmarks):
            bone_name = f"Bone{i}"
            motion_data[bone_name] = {
                "position_x": pose_landmarks.x,
                "position_y": pose_landmarks.y,
                "position_z": pose_landmarks.z
            }
        motion_node = fbx.FbxNode.create(scene, "Motion")

        for bone_name, bone_data in motion_data.items():
            bone_node = fbx.FbxNode.create(scene, bone_name)
            motion_node.add_child(bone_node)

            for channel_name, channel_data in bone_data.items():
                curve_node = fbx.FbxAnimCurveNode.create(scene, channel_name)
                bone_node.add_anim_curve_node(curve_node)

                for i in range(len(channel_data)):
                    curve_node.add_keyframe(i, channel_data[i])

            # Add the motion node to the scene
        scene.add_child(motion_node)



        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    pyfbx.FbxExporter.save(scene, "motion_capture.fbx")


if __name__ =="__main__":
    main()