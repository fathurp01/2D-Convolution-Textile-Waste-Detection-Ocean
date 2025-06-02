import cv2
import argparse
import os

def test_on_image(cascade_path, image_path, scale_factor=1.1, min_neighbors=3):
    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"Tidak dapat membaca gambar: {image_path}")
        return
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Muat cascade classifier
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print(f"Error: Tidak dapat memuat cascade classifier dari {cascade_path}")
        return
    
    # Deteksi objek
    objects = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    
    print(f"Ditemukan {len(objects)} objek")
    
    # Gambar bounding box
    for (x, y, w, h) in objects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Tampilkan gambar
    cv2.imshow("Deteksi", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Test Haar Cascade Classifier')
    parser.add_argument('--cascade', default='./cascade/cascade.xml', 
                        help='Path ke file cascade.xml')
    parser.add_argument('--image', required=True, 
                        help='Path ke gambar untuk diuji')
    parser.add_argument('--scale', type=float, default=1.1, 
                        help='Scale factor untuk detectMultiScale')
    parser.add_argument('--neighbors', type=int, default=3, 
                        help='minNeighbors untuk detectMultiScale')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.cascade):
        print(f"Error: File cascade {args.cascade} tidak ditemukan.")
        return
        
    if not os.path.exists(args.image):
        print(f"Error: File gambar {args.image} tidak ditemukan.")
        return
        
    test_on_image(args.cascade, args.image, args.scale, args.neighbors)

if __name__ == "__main__":
    main()