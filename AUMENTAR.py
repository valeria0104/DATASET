import os
import cv2
import albumentations as A

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (416, 416))  # Redimensionar a 416x416
    return image

def apply_augmentations(image):
    transform = A.Compose([
        A.Rotate(limit=10),
        A.Affine(shear=(10, 10)),
        A.ToGray(p=0.1),
        A.Hue((-25, 25), p=0.5),
        A.Saturation((-25, 25), p=0.5),
        A.Brightness((-25, 25), p=0.5),
        A.Exposure((-15, 15), p=0.5),
        A.Blur(blur_limit=3, p=0.5)
    ], p=1.0)  # Aplica todas las transformaciones

    augmented = transform(image=image)
    return augmented['image']

def main(images_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(images_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(images_folder, filename)
            image = load_image(image_path)
            augmented_image = apply_augmentations(image)
            # Guardar la imagen aumentada
            cv2.imwrite(os.path.join(output_folder, f'aug_{filename}'), augmented_image)

if __name__ == '__main__':
    images_folder = r'C:\Users\VALERIA\Desktop\DATASET\Letras\A\images'
    output_folder = r'C:\Users\VALERIA\Desktop\DATASET\FormatoYolo\train'
    main(images_folder, output_folder)
