## VERGLEICH ZWEIER TRAJEKTORIEN - BERECHNUNG MITTELS ARUCO VS. FEATURE-MATCHING

Im vorliegenden Projekt wird die Trajektorie einer Drohnenbewegung durch zwei verschiedene Pipelines berechnet. Diese werden im Anschluss grafisch dargestellt und miteinander verglichen.
Das Programm 'main.py' liefert hierfür drei Methoden:
- def aruco_pipeline()
- def visual_pipeline()
- def plot_trajectories(visual_odom_traj, aruco_traj)

Mit den erstgenannten werden die Trajektorien berechnet, mit der Letzeren werden Sie dargestellt.

## VORGEHENSWEISE

Verwendete OpenCV Version: 4.9.0

- Python Paket numpy installieren (pip install numpy)
- Python Paket matplotlib installieren (pip install matplotlib)
- Python Paket cv2 installieren (pip install numpy)

Das zu analysierende Video muss 'video.mp4' heißen und im gleichen Verzeichnis, wie die Main-Datei abliegen.
