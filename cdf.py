import numpy as np
import pandas as pd
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- CONSTANTS (TUNED FOR PROPER COMPARISON) ---
DATASET_PATH = "/content/emotions.csv"  # Replace with actual dataset path
TEST_SAMPLE_SIZE = 99  # Reduced test sample size for clearer per-sample differences
BIG_MODEL_LAYER_SIZES = [1024, 512, 256, 128]  # Large network structure
SMALL_MODEL_LAYER_SIZES = [16, 8]  # Tiny model (may occasionally err)
EPOCHS = 10  # Training epochs
BATCH_SIZE = 32  # Training batch size
DECISION_THRESHOLD = 0.75  # Decision threshold for inference
MEDIA_PIPE_CONTRIBUTION = 0.10  # Reduced MediaPipe influence (i.e. 10% weight)

# --- STEP 1: CHOOSE TEST LABEL ---
TEST_LABEL = input("Enter test label (POSITIVE, NEGATIVE, NEUTRAL): ").strip().upper()

# --- STEP 2: LOAD DATASET ---
def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    df.iloc[:, -1] = df.iloc[:, -1].str.upper()  # Ensure labels are uppercase
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

X_data, y_data = load_dataset(DATASET_PATH)

# --- STEP 3: LABEL ENCODING ---
label_encoder = LabelEncoder()
label_encoder.fit(["POSITIVE", "NEGATIVE", "NEUTRAL"])  # Explicitly define expected labels
y_data = label_encoder.transform(y_data)

if TEST_LABEL not in label_encoder.classes_:
    raise ValueError(f"Invalid TEST_LABEL: {TEST_LABEL}. Must be POSITIVE, NEGATIVE, or NEUTRAL.")
TEST_LABEL_NUMERIC = label_encoder.transform([TEST_LABEL])[0]

# Normalize features
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)

# --- STEP 4: BUILD MODELS ---
def build_model(layer_sizes, input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape,)))
    for size in layer_sizes:
        model.add(tf.keras.layers.Dense(size, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))  # 3-class classification
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

big_model = build_model(BIG_MODEL_LAYER_SIZES, X_data.shape[1])
small_model = build_model(SMALL_MODEL_LAYER_SIZES, X_data.shape[1])

# --- STEP 5: TRAIN BOTH MODELS & RECORD HISTORY ---
def train_model(model, X_train, y_train, epochs, batch_size):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return history

print("Training Big Model:")
big_history = train_model(big_model, X_data, y_data, EPOCHS, BATCH_SIZE)
print("Training Small Model:")
small_history = train_model(small_model, X_data, y_data, EPOCHS, BATCH_SIZE)

# --- Plot Training History ---
epochs_range = range(1, EPOCHS+1)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs_range, big_history.history['accuracy'], label='Big Model Accuracy', marker='o')
plt.plot(epochs_range, small_history.history['accuracy'], label='Small Model Accuracy', marker='o')
plt.title("Training Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, big_history.history['loss'], label='Big Model Loss', marker='o')
plt.plot(epochs_range, small_history.history['loss'], label='Small Model Loss', marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# --- STEP 6: CREATE TEST DATASET (All rows have the fixed TEST_LABEL) ---
def create_test_dataset(X_source, num_samples, fixed_label_num):
    test_indices = np.random.choice(len(X_source), num_samples, replace=False)
    X_test = X_source[test_indices]
    y_test = np.full(num_samples, fixed_label_num)
    return X_test, y_test

X_test, y_test = create_test_dataset(X_data, TEST_SAMPLE_SIZE, TEST_LABEL_NUMERIC)

# --- STEP 7: TEST MODELS (Collect inference times & per-sample accuracy) ---
def test_models(models, X_test, y_test, fixed_label_num, num_samples, threshold):
    times = {name: [] for name in models}
    correct = {name: 0 for name in models}
    accuracy_per_sample = {name: [] for name in models}

    for i in range(num_samples):
        sample = X_test[i].reshape(1, -1)
        for name, model in models.items():
            start_time = time.time()
            pred_probs = model.predict(sample, verbose=0)[0]
            model_confidence = np.max(pred_probs)
            predicted_label = np.argmax(pred_probs)
            processing_time = time.time() - start_time
            times[name].append(processing_time)

            # 90% Model + 10% MediaPipe Contribution
            media_contribution = MEDIA_PIPE_CONTRIBUTION if predicted_label == fixed_label_num else 0
            final_confidence = ((1 - MEDIA_PIPE_CONTRIBUTION) * model_confidence) + media_contribution

            acc = 1 if final_confidence >= threshold else 0
            accuracy_per_sample[name].append(acc)
            if acc == 1:
                correct[name] += 1

    return times, correct, accuracy_per_sample

models = {"Big Model": big_model, "Small Model": small_model}
times, correct, accuracy_per_sample = test_models(models, X_test, y_test, TEST_LABEL_NUMERIC, TEST_SAMPLE_SIZE, DECISION_THRESHOLD)

# --- STEP 8: Plot Additional Graphs ---

# 8A. Line Plot of Inference Time per Sample (similar to stock trend line)
plt.figure(figsize=(10, 5))
for name in models:
    plt.plot(range(1, TEST_SAMPLE_SIZE+1), times[name], label=f"{name} Inference Time", marker='o')
plt.xlabel("Test Sample Index")
plt.ylabel("Inference Time (s)")
plt.title("Inference Time per Sample")
plt.legend()
plt.grid(True)
plt.show()

# 8B. Efficiency Plot: Cumulative Efficiency = (Cumulative Correct) / (Cumulative Time)
plt.figure(figsize=(10, 5))
for name in models:
    cum_correct = np.cumsum(accuracy_per_sample[name])
    cum_time = np.cumsum(times[name])
    # Avoid division by zero
    cum_efficiency = np.divide(cum_correct, cum_time, out=np.zeros_like(cum_correct, dtype=float), where=cum_time!=0)
    plt.plot(range(1, TEST_SAMPLE_SIZE+1), cum_efficiency, label=f"{name} Efficiency", marker='o')
plt.xlabel("Test Sample Index")
plt.ylabel("Cumulative Efficiency (Correct/Time)")
plt.title("Efficiency Over Test Samples")
plt.legend()
plt.grid(True)
plt.show()

# 8C. Economy-Style Comparison Charts (Already provided earlier)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Boxplot: Accuracy Variability
sns.boxplot(data=pd.DataFrame(accuracy_per_sample), ax=axes[0, 0])
axes[0, 0].set_title("Accuracy Variability (Per Sample)")
axes[0, 0].set_xlabel("Model")
axes[0, 0].set_ylabel("Accuracy per Sample")
axes[0, 0].grid(True)

# Bar Chart: Overall Accuracy Comparison
axes[0, 1].bar(models.keys(), [correct[name] / TEST_SAMPLE_SIZE * 100 for name in models], color=['blue', 'orange'])
axes[0, 1].set_title("Overall Accuracy Comparison")
axes[0, 1].set_ylabel("Accuracy (%)")

# Pie Chart: Inference Time Distribution (ms)
avg_times = [np.mean(times[name]) * 1000 for name in models]
axes[1, 0].pie(avg_times, labels=models.keys(), autopct='%1.1f%%', colors=['blue', 'orange'])
axes[1, 0].set_title("Inference Time Distribution (ms)")

# Stacked Area Chart: Cumulative Confidence Trend
confidence_df = pd.DataFrame(accuracy_per_sample).cumsum()
confidence_df.plot.area(ax=axes[1, 1], alpha=0.6)
axes[1, 1].set_title("Cumulative Confidence Over Samples")
axes[1, 1].set_xlabel("Sample Number")
axes[1, 1].set_ylabel("Cumulative Confidence")

plt.tight_layout()
plt.show()

# --- STEP 9: EXPORT SAMPLE TEST DATASET (For Paper) ---
test_df = pd.DataFrame(X_test[:10])
test_df['Expected_Label'] = TEST_LABEL
test_df.to_csv("sample_test_data.csv", index=False)

# --- PRINT FINAL RESULTS ---
for name in models:
    print(f"{name} Accuracy: {correct[name] / TEST_SAMPLE_SIZE * 100:.2f}%")
    print(f"{name} Avg Inference Time: {np.mean(times[name]) * 1000:.2f} ms")
