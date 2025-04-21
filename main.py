import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Load the image
image_path = 'input_face.jpg'
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to detect facial landmarks
results = face_mesh.process(rgb_image)

if results.multi_face_landmarks:
    landmarks = results.multi_face_landmarks[0].landmark
    print(f"Detected {len(landmarks)} landmarks.")
    # Extract XYZ coordinates of each landmark
    coords = [(lm.x, lm.y, lm.z) for lm in landmarks]
else:
    print("No face detected.")

def save_to_obj(coords, output_path='face_mesh.obj'):
    with open(output_path, 'w') as obj_file:
        obj_file.write("# OBJ file generated from MediaPipe FaceMesh\n")
        for x, y, z in coords:
            obj_file.write(f"v {x} {y} {z}\n")
        # Define triangular mesh connections (optional)
        # Add your own logic for creating faces (f lines in OBJ)
    print(f"OBJ file saved at {output_path}")

save_to_obj(coords)
