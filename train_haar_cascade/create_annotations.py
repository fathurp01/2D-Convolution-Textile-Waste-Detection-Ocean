import cv2
import os
import argparse

# Variabel global untuk menyimpan koordinat bounding box
rectangles = []
current_rectangle = []
drawing = False

def click_and_crop(event, x, y, flags, param):
    global rectangles, current_rectangle, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_rectangle = [(x, y)]
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            clone = image.copy()
            cv2.rectangle(clone, current_rectangle[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", clone)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_rectangle.append((x, y))
        
        # Hitung koordinat dan dimensi bounding box
        x1, y1 = current_rectangle[0]
        x2, y2 = current_rectangle[1]
        
        # Pastikan x1,y1 adalah pojok kiri atas dan x2,y2 adalah pojok kanan bawah
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
            
        width = x2 - x1
        height = y2 - y1
        
        # Simpan koordinat bounding box (x, y, width, height)
        rectangles.append((x1, y1, width, height))
        
        # Gambar bounding box pada gambar
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Image", image)

def annotate_images(input_dir, output_file):
    global image, rectangles
    
    # Buat daftar semua file gambar di direktori input
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Buka file output untuk menulis anotasi
    with open(output_file, 'w') as f:
        for img_file in image_files:
            img_path = os.path.join(input_dir, img_file)
            
            # Baca gambar
            image = cv2.imread(img_path)
            if image is None:
                print(f"Tidak dapat membaca gambar: {img_path}")
                continue
                
            # Reset daftar bounding box
            rectangles = []
            
            # Tampilkan gambar dan atur callback mouse
            cv2.namedWindow("Image")
            cv2.setMouseCallback("Image", click_and_crop)
            cv2.imshow("Image", image)
            
            print(f"\nAnotasi untuk {img_file}:")
            print("Klik dan seret untuk membuat bounding box.")
            print("Tekan 's' untuk menyimpan anotasi dan lanjut ke gambar berikutnya.")
            print("Tekan 'r' untuk mereset anotasi pada gambar ini.")
            print("Tekan 'q' untuk keluar.")
            
            while True:
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('s'):  # Simpan anotasi dan lanjut ke gambar berikutnya
                    if rectangles:
                        # Format: path/to/img.jpg num_objects x y width height [x y width height ...]
                        rel_path = os.path.join('positives', img_file)
                        line = f"{rel_path} {len(rectangles)}"
                        
                        for rect in rectangles:
                            line += f" {rect[0]} {rect[1]} {rect[2]} {rect[3]}"
                            
                        f.write(line + '\n')
                        print(f"Anotasi disimpan: {line}")
                    else:
                        print("Tidak ada anotasi yang dibuat, melewati gambar.")
                    break
                    
                elif key == ord('r'):  # Reset anotasi
                    rectangles = []
                    image = cv2.imread(img_path)
                    cv2.imshow("Image", image)
                    print("Anotasi direset.")
                    
                elif key == ord('q'):  # Keluar
                    return
    
    cv2.destroyAllWindows()
    print(f"\nSelesai! Anotasi disimpan ke {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Tool untuk membuat anotasi bounding box')
    parser.add_argument('--input_dir', default='../dataset/positives', 
                        help='Direktori berisi gambar positif')
    parser.add_argument('--output_file', default='../dataset/info.lst', 
                        help='File output untuk menyimpan anotasi')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Direktori {args.input_dir} tidak ditemukan.")
        return
        
    annotate_images(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()