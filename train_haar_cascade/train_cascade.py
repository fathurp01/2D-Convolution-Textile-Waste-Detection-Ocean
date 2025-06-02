import os
import subprocess
import argparse

def create_samples(info_path, bg_path, vec_path, width, height, num_samples):
    """Membuat file vektor sampel positif"""
    cmd = [
        'opencv_createsamples',
        '-info', info_path,
        '-bg', bg_path,
        '-vec', vec_path,
        '-w', str(width),
        '-h', str(height),
        '-num', str(num_samples)
    ]
    
    print("Menjalankan: " + " ".join(cmd))
    subprocess.call(cmd)

def train_cascade(data_dir, vec_path, bg_path, num_pos, num_neg, num_stages, 
                 width, height, feature_type='HAAR'):
    """Melatih cascade classifier"""
    cmd = [
        'opencv_traincascade',
        '-data', data_dir,
        '-vec', vec_path,
        '-bg', bg_path,
        '-numPos', str(num_pos),
        '-numNeg', str(num_neg),
        '-numStages', str(num_stages),
        '-w', str(width),
        '-h', str(height),
        '-featureType', feature_type,
        '-precalcValBufSize', '1024',
        '-precalcIdxBufSize', '1024'
    ]
    
    print("Menjalankan: " + " ".join(cmd))
    subprocess.call(cmd)

def main():
    parser = argparse.ArgumentParser(description='Train Haar Cascade for Textile Waste Detection')
    parser.add_argument('--info', default='../dataset/info.lst', help='Path to positive samples info file')
    parser.add_argument('--bg', default='../dataset/bg.txt', help='Path to background samples file')
    parser.add_argument('--vec', default='samples.vec', help='Output vector file')
    parser.add_argument('--width', type=int, default=24, help='Width of training samples')
    parser.add_argument('--height', type=int, default=24, help='Height of training samples')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of positive samples')
    parser.add_argument('--num_pos', type=int, default=900, help='Number of positive samples for training')
    parser.add_argument('--num_neg', type=int, default=450, help='Number of negative samples for training')
    parser.add_argument('--num_stages', type=int, default=10, help='Number of stages in cascade')
    parser.add_argument('--feature_type', default='HAAR', choices=['HAAR', 'LBP'], help='Feature type')
    parser.add_argument('--data_dir', default='./cascade', help='Directory to store cascade files')
    
    args = parser.parse_args()
    
    # Buat direktori output jika belum ada
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Langkah 1: Buat file vektor sampel positif
    create_samples(
        args.info, 
        args.bg, 
        args.vec, 
        args.width, 
        args.height, 
        args.num_samples
    )
    
    # Langkah 2: Latih cascade classifier
    train_cascade(
        args.data_dir,
        args.vec,
        args.bg,
        args.num_pos,
        args.num_neg,
        args.num_stages,
        args.width,
        args.height,
        args.feature_type
    )
    
    print("\nPelatihan selesai! File cascade.xml tersedia di: " + os.path.join(args.data_dir, 'cascade.xml'))
    print("Salin file cascade.xml ke folder haarcascade_sampah/ untuk digunakan dalam aplikasi utama.")

if __name__ == "__main__":
    main()