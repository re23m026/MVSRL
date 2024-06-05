import cv2
import numpy as np
import matplotlib.pyplot as plt 

# Bei Aruco: rvec, tvec --> Absolute Koordinatenangaben von Kamera zu Marker
# Bei Visual Odometrie: R, t --> Relative Koordinatenangaben von Frame zu Frame

## ------------------- VISUAL ODOMETRY PIPELINE ------------------------------------

def visual_pipeline():
    
    # Öffnen des Videos
    video = cv2.VideoCapture('video.mp4')

    # Initialisierung des SIFT-Detektors und eines Brute-Force-Matchers (oder ORB-Detektor)
    sift = cv2.SIFT.create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Variablen zur Speicherung der vorherigen Keypoints und Deskriptoren
    previous_frame = None
    previous_keypoints = None
    previous_descriptors = None

    # Startposition der Kamera (4x4 Identitätsmatrix)
    pose = np.eye(4)
    all_poses = [pose[:3, 3].copy()]  # Liste zur Speicherung der Trajektorie
    
    # Fenster für Visual Trajektorie erstellen
    cv2.namedWindow("SIFT Keypoints", cv2.WINDOW_NORMAL)
    cv2.namedWindow("SIFT Matches", cv2.WINDOW_NORMAL)
    
    while video.isOpened():
        
        # Lesen des nächsten Frames
        ret, frame = video.read()
        if not ret:
            break
        
        # Umwandeln des Bildes in Graustufen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extrahieren der Keypoints und Deskriptoren im aktuellen Frame
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        # Zeichnen der Keypoints im aktuellen Frame
        keypoint_image = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #keypoint_image = cv2.resize(keypoint_image, (1000, 1000))
        cv2.imshow('SIFT Keypoints', keypoint_image)
        #cv2.resizeWindow("SIFT Keypoints", 400, 400)
        
        if previous_frame is not None:
            # Vergleichen der aktuellen Keypoints mit denen aus dem vorherigen Frame
            matches = bf.match(previous_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)  # Sortieren nach Übereinstimmungsqualität

            # Wählen der besten N Übereinstimmungen
            number_features = 10
            matches = matches[:number_features]
            
            # Zeichnen der Matches zwischen aktuellen und vorherigem Frame
            match_image = cv2.drawMatches(previous_frame, previous_keypoints, frame, keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #match_image = cv2.resize(match_image, (1000, 1000))
            cv2.imshow('SIFT Matches', match_image)
            
            # Extrahieren der Koordinaten für die gefundenen Übereinstimmungen
            prev_pts = np.float32([previous_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            curr_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)

            # Berechnen der essentiellen Matrix mit RANSAC zur Unterdrückung von Ausreißern
            E, mask = cv2.findEssentialMat(curr_pts, prev_pts, method=cv2.RANSAC, prob=0.98, threshold=0.7)

            # Wiederherstellen der Rotation und Translation (Pose) zwischen den beiden Frames
            _, R, t, _ = cv2.recoverPose(E, curr_pts, prev_pts)

            # Aktualisieren der Pose durch Multiplizieren mit der aktuellen Transformation
            new_pose = np.eye(4)
            #new_pose[:3, :3] = R  # Rotation
            new_pose[:3, 3] = t.squeeze()  # Translation

            pose = pose @ new_pose  # Update der Gesamtposition
            all_poses.append(pose[:3, 3].copy())
            
        
        # Aktualisieren der vorherigen Keypoints und Deskriptoren für das nächste Frame
        previous_frame = frame
        previous_keypoints = keypoints
        previous_descriptors = descriptors
        
        # Visualisieren der Visual Pipeline  
        print("New Pose: ")
        print(pose)
                
        # Auf Tastatureingabe warten
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
     
    
    np.savetxt('all_traj_visual.txt', all_poses) 

    # Aufräumen
    video.release()
    cv2.destroyAllWindows()
    return np.array(all_poses)


## ------------------------ ARUCO PIPELINE ----------------------------------------

def aruco_pipeline():
    
    video = cv2.VideoCapture('video.mp4')
    cv2.namedWindow("Aruco", cv2.WINDOW_NORMAL)
    
    all_tvecs = []

    
    while video.isOpened():
        
        #Lesen den nächsten Frames
        ret, frame=video.read()
        if not ret:
            break

        # Kopie des aktuellen Frames
        image = frame.copy()

        # Konvertiere das Bild in Graustufen
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Lade das ARuco-Dictionary und den ARuco-Parameter-Detektor
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()

        # Erkenne die Marker im Bild
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Wenn Marker gefunden wurden, zeichne sie
        if ids is not None:
            
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
        
            #------Pose Estimation------
        
            # Definiere Kameramatrix und Verzerrungskoeffizienten
            cameraMatrix = np.array([[1184.494873221266, 0, 578.893666990779],
                                [0, 1189.11908403797, 828.682884187951],
                                [0, 0, 1]])
            distCoeffs = np.array([0.1749986657708608, -0.652490031555235, 0.00507051440408092, 0.000505163201103598,  1.3196125905465841])

            # Markerlänge festlegen (angenommen)
            markerLength = 0.1  # z. B. 10 cm

            # Pose der Marker schätzen
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs)

            # Aktuelle Pose der Trajektorie hinzufügen        
            for tvec in tvecs:
                all_tvecs.append(tvec.squeeze())

        # rvecs und tvecs enthalten die Rotations- und Translationsvektoren für jeden Marker
        print("Rotationsvektoren für jeden Marker:")
        print(rvecs)
        print("Translationsvektoren für jeden Marker:")
        print(tvecs)

        #image = cv2.resize(image, (1000, 1000))
        cv2.imshow("Aruco", image)

        # Auf Tastatureingabe warten
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        
    # Kamera-Objekt freigeben und Fenster schließen
    video.release()
    cv2.destroyAllWindows()
    
    # Subtrahiere den ersten Translationsvektor von allen folgenden
    if len(all_tvecs) > 0:
        first_tvec = all_tvecs[0]       
        all_tvecs = [(tvec - first_tvec) * 100 for tvec in all_tvecs]            
        all_tvecs = [-x for x in all_tvecs]
    
    # Speichern von all_tvecs in einer Textdatei
    np.savetxt('all_tvecs_aruco.txt', all_tvecs)
    
    return np.array(all_tvecs)


## ------------------- VISUALIZATION ---------------------

def plot_trajectories(visual_odom_traj, aruco_traj):
    
    plt.figure()
    plt.plot(visual_odom_traj[:, 0], visual_odom_traj[:, 1], label='Visual Odometry')
    plt.plot(aruco_traj[:, 0], aruco_traj[:, 1], label='Aruco')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.show()   

## ------------------- PROGRAMMABLAUF --------------------

# 1. Aruco Pipeline ausführen (Ground Truth)
# 2. Visual Odometry Pipeline ausführen

aruco_trajectory = aruco_pipeline()
visual_odom_trajectory = visual_pipeline()

# 3. Visualisierung beider Trajektorien (Vergleich)

plot_trajectories(visual_odom_trajectory, aruco_trajectory)