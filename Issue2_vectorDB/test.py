import numpy as np

# 각 모델의 혼동 행렬을 여기에 추가합니다.
confusion_matrices = {
    'OmniVec (ViT)': np.array([
        [9, 0, 1, 0, 1, 0, 3, 0, 0, 0, 2, 0],
        [0, 14, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 8, 0, 1, 0, 0, 2, 0, 0, 1, 2],
        [1, 2, 0, 8, 1, 1, 1, 0, 0, 0, 2, 0],
        [1, 0, 0, 0, 11, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 9, 0, 0, 0, 0, 0, 2],
        [2, 0, 0, 1, 1, 0, 9, 0, 0, 0, 3, 0],
        [1, 0, 2, 0, 1, 1, 0, 8, 0, 0, 1, 2],
        [0, 0, 1, 0, 1, 0, 1, 0, 11, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 1],
        [1, 0, 2, 2, 0, 1, 3, 1, 0, 0, 5, 1],
        [1, 0, 4, 0, 1, 1, 2, 0, 0, 0, 1, 6]
    ]),
    'CoCa': np.array([
        [8, 0, 1, 0, 0, 1, 2, 0, 0, 0, 3, 1],
        [0, 14, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [3, 1, 5, 0, 0, 1, 0, 2, 0, 0, 0, 4],
        [0, 2, 1, 9, 1, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 11, 4, 0, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 3, 8, 0, 1, 0, 0, 2, 0],
        [6, 0, 0, 1, 0, 0, 8, 0, 0, 0, 1, 0],
        [0, 0, 3, 0, 0, 2, 0, 10, 0, 0, 1, 0],
        [1, 0, 2, 1, 1, 1, 0, 0, 8, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 1],
        [2, 3, 0, 0, 1, 3, 1, 2, 0, 0, 4, 0],
        [4, 0, 3, 0, 1, 0, 0, 0, 0, 1, 1, 6]
    ]),
    'Swin Transformer V2': np.array([
        [10, 0, 1, 0, 1, 0, 2, 0, 0, 0, 2, 0],
        [0, 15, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 1, 6, 0, 0, 1, 0, 3, 0, 0, 0, 3],
        [0, 1, 0, 8, 0, 1, 1, 0, 1, 0, 4, 0],
        [0, 0, 0, 0, 9, 7, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 4, 10, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 3, 10, 0, 0, 0, 1, 0],
        [0, 0, 2, 1, 0, 0, 0, 11, 0, 0, 0, 2],
        [0, 0, 0, 0, 1, 0, 2, 0, 11, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 1],
        [0, 0, 0, 2, 3, 1, 2, 1, 1, 0, 6, 0],
        [2, 0, 5, 0, 0, 0, 2, 1, 1, 0, 0, 5]
    ]),
    'ConvNeXt': np.array([
        [11, 0, 0, 0, 0, 2, 2, 0, 0, 0, 1, 0],
        [0, 15, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 7, 0, 1, 0, 0, 2, 0, 0, 0, 5],
        [1, 2, 0, 8, 0, 1, 1, 0, 1, 0, 2, 0],
        [1, 0, 1, 0, 11, 3, 0, 0, 0, 0, 0, 0],
        [2, 0, 1, 0, 4, 9, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 1, 9, 0, 0, 0, 3, 1],
        [0, 0, 3, 0, 0, 0, 0, 11, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 12, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 1],
        [1, 0, 0, 3, 1, 2, 1, 1, 1, 0, 5, 1],
        [1, 0, 3, 0, 1, 0, 1, 0, 0, 1, 0, 9]
    ]),
    'EfficientNetV2': np.array([
        [7, 0, 1, 0, 0, 4, 4, 0, 0, 0, 0, 0],
        [0, 14, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 8, 0, 1, 0, 1, 2, 0, 0, 0, 2],
        [0, 1, 0, 11, 0, 1, 2, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 13, 2, 0, 0, 0, 0, 0, 0],
        [3, 0, 1, 0, 1, 8, 0, 0, 0, 0, 3, 0],
        [6, 1, 0, 0, 0, 1, 8, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 11, 0, 0, 1, 2],
        [1, 0, 1, 0, 0, 0, 0, 0, 13, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0],
        [1, 0, 0, 3, 1, 2, 2, 1, 0, 0, 5, 1],
        [2, 0, 3, 0, 1, 0, 1, 1, 0, 0, 0, 8]
    ]),
    'RegNet': np.array([
        [9, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 2],
        [0, 14, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [2, 1, 7, 0, 1, 0, 0, 0, 0, 0, 0, 5],
        [1, 1, 1, 7, 1, 1, 1, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 11, 4, 1, 0, 0, 0, 0, 0],
        [2, 0, 1, 0, 4, 8, 0, 0, 0, 0, 1, 0],
        [2, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0],
        [0, 0, 4, 0, 0, 0, 0, 10, 0, 0, 0, 2],
        [1, 0, 0, 0, 0, 0, 2, 0, 12, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0],
        [0, 0, 1, 2, 2, 2, 3, 0, 0, 0, 6, 0],
        [1, 0, 4, 0, 1, 0, 1, 0, 0, 0, 0, 9]
    ]),
    'DeiT': np.array([
        [9, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 1],
        [0, 14, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [3, 1, 6, 0, 0, 1, 0, 4, 0, 0, 0, 1],
        [0, 1, 0, 8, 4, 0, 1, 0, 0, 0, 2, 0],
        [1, 0, 0, 0, 8, 6, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 5, 10, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 8, 0, 0, 0, 3, 0],
        [0, 0, 3, 0, 0, 0, 0, 10, 0, 0, 1, 2],
        [1, 0, 0, 1, 2, 1, 0, 0, 11, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 0],
        [1, 0, 0, 0, 4, 1, 3, 1, 0, 0, 6, 0],
        [2, 0, 7, 0, 1, 0, 2, 1, 1, 0, 0, 2]
    ]),
    'NFNet': np.array([
        [6, 0, 1, 0, 0, 1, 6, 0, 0, 0, 1, 1],
        [0, 13, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 8, 0, 0, 0, 0, 1, 0, 0, 1, 3],
        [0, 3, 0, 7, 3, 0, 0, 0, 1, 0, 2, 0],
        [0, 0, 0, 2, 9, 5, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 4, 8, 0, 1, 0, 0, 1, 0],
        [4, 0, 0, 1, 0, 0, 7, 1, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 2, 2, 9, 0, 0, 0, 2],
        [1, 0, 0, 3, 2, 0, 0, 0, 8, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 0, 0],
        [2, 0, 1, 3, 2, 2, 0, 0, 1, 0, 5, 0],
        [1, 0, 7, 0, 0, 2, 1, 2, 0, 0, 0, 3]
    ]),
    'ResNet18': np.array([
        [11, 0, 0, 0, 0, 1, 2, 0, 0, 0, 2, 0],
        [0, 14, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 10, 0, 1, 0, 0, 0, 0, 0, 1, 3],
        [0, 3, 0, 6, 1, 0, 1, 0, 1, 0, 4, 0],
        [1, 0, 0, 0, 11, 3, 0, 0, 0, 0, 1, 0],
        [3, 0, 2, 0, 2, 7, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 9, 0, 0, 1, 3, 1],
        [1, 0, 2, 0, 2, 0, 0, 9, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0, 11, 0, 2, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 5, 0, 0],
        [1, 0, 1, 2, 1, 3, 1, 0, 1, 0, 6, 0],
        [1, 0, 2, 0, 1, 0, 1, 0, 0, 1, 0, 10]
    ]),
    'ResNet50': np.array([
        [9, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 1],
        [0, 15, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [2, 1, 5, 0, 1, 0, 0, 2, 0, 0, 0, 5],
        [0, 2, 0, 9, 2, 0, 2, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 10, 3, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 4, 9, 0, 0, 0, 0, 1, 1],
        [2, 0, 0, 0, 1, 0, 12, 0, 0, 0, 1, 0],
        [0, 0, 3, 0, 1, 0, 1, 8, 0, 0, 1, 2],
        [1, 0, 0, 0, 0, 1, 3, 0, 11, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 0, 0],
        [1, 0, 0, 0, 2, 1, 2, 1, 0, 0, 8, 1],
        [1, 0, 2, 0, 0, 1, 2, 1, 0, 0, 0, 9]
    ]),
    'EfficientNet B0': np.array([
        [6, 0, 1, 0, 0, 1, 5, 0, 0, 0, 2, 1],
        [0, 15, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 4, 0, 0, 1, 0, 3, 0, 0, 0, 4],
        [0, 1, 0, 9, 1, 1, 1, 0, 1, 0, 2, 0],
        [0, 0, 0, 0, 10, 3, 0, 1, 0, 0, 0, 2],
        [1, 0, 1, 0, 2, 10, 0, 0, 0, 0, 1, 1],
        [3, 0, 0, 1, 1, 0, 10, 0, 0, 0, 1, 0],
        [0, 0, 5, 0, 1, 0, 0, 8, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 1, 12, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0],
        [0, 0, 0, 1, 1, 3, 2, 1, 0, 0, 8, 0],
        [0, 0, 5, 0, 0, 2, 1, 0, 2, 0, 0, 6]
    ]),
    'EfficientNet B7': np.array([
        [8, 0, 1, 0, 0, 0, 5, 0, 0, 0, 1, 1],
        [0, 15, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 7, 0, 0, 2, 0, 3, 0, 0, 0, 3],
        [0, 0, 0, 10, 1, 1, 0, 0, 1, 0, 3, 0],
        [0, 1, 0, 0, 10, 5, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 4, 7, 0, 0, 0, 0, 2, 1],
        [5, 0, 0, 0, 0, 0, 9, 0, 0, 0, 2, 0],
        [0, 0, 6, 0, 0, 1, 1, 6, 0, 0, 0, 2],
        [1, 0, 0, 0, 1, 2, 0, 0, 10, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0],
        [1, 0, 1, 2, 2, 1, 3, 0, 0, 0, 6, 0],
        [2, 0, 5, 0, 0, 1, 2, 0, 1, 1, 0, 4]
    ])

}

# 각 모델의 평균 정확도 계산 함수
def calculate_average_accuracy(confusion_matrix):
    accuracy_per_class = []
    for i in range(len(confusion_matrix)):
        true_positives = confusion_matrix[i, i]
        total_samples = np.sum(confusion_matrix[i, :])
        accuracy = true_positives / total_samples if total_samples > 0 else 0
        accuracy_per_class.append(accuracy)
    
    average_accuracy = np.mean(accuracy_per_class)
    return average_accuracy

# 각 모델의 평균 정확도 계산
average_accuracies = {model: calculate_average_accuracy(cm) for model, cm in confusion_matrices.items()}

# 결과 출력
for model, accuracy in average_accuracies.items():
    print(f"{model}: {accuracy:.2%}")

# 가장 높은 평균 정확도를 가진 모델 선택
best_model = max(average_accuracies, key=average_accuracies.get)
best_accuracy = average_accuracies[best_model]

print(f"\n최고의 모델: {best_model}")
print(f"평균 정확도: {best_accuracy:.2%}")
