from deepface import DeepFace

path = './assets/cara1.png'
path2 = './assets/cara2.jpeg'

verification_result = DeepFace.verify(img1_path=path, img2_path=path2)

# print(verification_result)

for k, v in verification_result.items():
    print(f'{k}: {v}')
