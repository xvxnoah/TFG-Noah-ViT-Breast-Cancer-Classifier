from common import *


### RESET AND SHUFFLE DATA ###
def reset_and_shuffle_data():
    base_path = '/'
    images_path = os.path.join(base_path, 'breast_images')
    train_path = os.path.join(base_path, 'train')
    val_path = os.path.join(base_path, 'val')
    test_path = os.path.join(base_path, 'test')

    # Move files back to the original images path
    def move_files_back(set_path):
        for client_folder in os.listdir(set_path):
            source_folder = os.path.join(set_path, client_folder)
            if os.path.isdir(source_folder):  # Check if it is a directory
                destination_folder = os.path.join(images_path, client_folder)
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                for file_name in os.listdir(source_folder):
                    shutil.move(os.path.join(source_folder, file_name), destination_folder)
                os.rmdir(source_folder)

    move_files_back(train_path)
    move_files_back(val_path)
    move_files_back(test_path)


### DATA SPLITTING ###
def split_data():
    reset_and_shuffle_data()

    # Load the CSV file containing client images and their statuses
    csv_file = 'optimam_assets/client_images_screening.csv'
    df = pd.read_csv(csv_file)

    # Filter out clients whose status is Normal or Interval Cancer
    df = df[~df['status'].isin(["Normal", "Interval Cancer"])]

    # Group by client_id
    client_labels = df.groupby('client_id')['status'].agg(lambda x: x.value_counts().index[0])

    # Split clients into training, validation, and test sets
    train_clients, temp_clients = train_test_split(client_labels.index, test_size=0.3, stratify=client_labels)
    val_clients, test_clients = train_test_split(temp_clients, test_size=(2/3), stratify=client_labels[temp_clients])

    # Print class distribution for each set
    print("Train set class distribution:")
    print(client_labels[train_clients].value_counts())
    print("Validation set class distribution:")
    print(client_labels[val_clients].value_counts())
    print("Test set class distribution:")
    print(client_labels[test_clients].value_counts())

    base_path = '/'
    dest_path = '/'
    images_path = os.path.join(base_path, 'breast_images')
    train_path = os.path.join(dest_path, 'train')
    val_path = os.path.join(dest_path, 'val')
    test_path = os.path.join(dest_path, 'test')

    # Ensure the directories for train, val, and test sets exist
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Function to copy images of specified clients to a designated set path
    def copy_client_images(client_ids, set_path):
        for client_id in client_ids:
            client_folder = os.path.join(images_path, client_id)
            destination_folder = os.path.join(set_path, client_id)
            os.makedirs(destination_folder, exist_ok=True)
            if os.path.exists(client_folder):
                for file_name in os.listdir(client_folder):
                    file_path = os.path.join(client_folder, file_name)
                    shutil.move(file_path, destination_folder)

    # Function to save client data to CSV files for convenient access
    def save_csv():
        train_df = pd.DataFrame(train_clients, columns=['client_id'])
        train_df['status'] = train_df['client_id'].map(client_labels)

        val_df = pd.DataFrame(val_clients, columns=['client_id'])
        val_df['status'] = val_df['client_id'].map(client_labels)

        test_df = pd.DataFrame(test_clients, columns=['client_id'])
        test_df['status'] = test_df['client_id'].map(client_labels)

        train_df.to_csv('train_clients.csv', index=False)
        val_df.to_csv('val_clients.csv', index=False)
        test_df.to_csv('test_clients.csv', index=False)

    # Copy client images to their respective directories
    copy_client_images(train_clients, train_path)
    copy_client_images(val_clients, val_path)
    copy_client_images(test_clients, test_path)

    # Save the client data to CSV files
    save_csv()

    print("\nData splitting, shuffle, and copying completed.\n")


### DATA LOADING AND PREPROCESSING ###
def load_data(base_path, set_csv_file, data_csv, img_size=(224, 224)):
    # Load the CSV files containing the dataset information
    df = pd.read_csv(set_csv_file)
    df_data = pd.read_csv(data_csv)
    data = []

    # Iterate over unique client IDs
    for client_id in tqdm(df['client_id'].unique()):
        client_path = os.path.join(base_path, client_id)
        for root, dirs, files in os.walk(client_path):
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)
                    image_id = os.path.basename(img_path).rsplit('.', 1)[0]
                    bbox_data = df_data[(df_data['client_id'] == client_id) & (df_data['image_id'] == image_id)]

                    # Check if bounding box data is available and valid
                    if not bbox_data.empty and not bbox_data[['x1', 'x2', 'y1', 'y2']].isnull().values.any():
                        with Image.open(img_path) as img:
                            for _, row in bbox_data.iterrows():
                                status = row['status']
                                xmin, xmax, ymin, ymax = row[['xmin_cropped', 'xmax_cropped', 'ymin_cropped', 'ymax_cropped']].astype(int)
                                x1, y1, x2, y2 = row[['x1', 'y1', 'x2', 'y2']].astype(int)

                                # Adjust bounding box coordinates based on the crop
                                x1_adj = x1 - xmin
                                y1_adj = y1 - ymin
                                x2_adj = x2 - xmin
                                y2_adj = y2 - ymin

                                # Ensure the bounding box is valid and within the cropped area
                                if x1_adj < x2_adj and y1_adj < y2_adj:
                                    img_cropped = img.crop((xmin, ymin, xmax, ymax)).convert('L')
                                    
                                    # Convert PIL image to numpy array for processing
                                    img_array = np.array(img_cropped, dtype=np.uint8)

                                    # Apply median filter for noise reduction
                                    img_filtered = cv2.medianBlur(img_array, 5)

                                    # Apply CLAHE
                                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                                    img_clahe = clahe.apply(img_filtered)

                                    # Resize image
                                    img_resized = cv2.resize(img_clahe, img_size)

                                    # Normalize pixel values
                                    image_np = img_resized / 255.0

                                    # Check if the cropped image is completely black
                                    if np.max(image_np) > 0:
                                        data.append((image_np, status))  # Append the processed image and label

    return data


def prepare_data_for_training():
    split_data()

    train_data = load_data('train/', 'train_clients.csv', 'optimam_assets/client_images_screening.csv')
    val_data = load_data('val/', 'val_clients.csv', 'optimam_assets/client_images_screening.csv')
    test_data = load_data('test/', 'test_clients.csv', 'optimam_assets/client_images_screening.csv')

    return train_data, val_data, test_data


### DATA SAVING AND LOADING ###
def save_data(train_data, val_data, test_data, train_file='train_data.pkl', val_file='val_data.pkl', test_file='test_data.pkl'):
    # Save training data to a pickle file
    with open(train_file, 'wb') as file:
        pickle.dump(train_data, file)

    # Save validation data to a pickle file
    with open(val_file, 'wb') as file:
        pickle.dump(val_data, file)

    # Save testing data to a pickle file
    with open(test_file, 'wb') as file:
        pickle.dump(test_data, file)


def load_set_data(train_file='train_data.pkl', val_file='val_data.pkl', test_file='test_data.pkl'):
    # Load training data from a pickle file
    with open(train_file, 'rb') as file:
        train_data = pickle.load(file)

    # Load validation data from a pickle file
    with open(val_file, 'rb') as file:
        val_data = pickle.load(file)

    # Load testing data from a pickle file
    with open(test_file, 'rb') as file:
        test_data = pickle.load(file)

    return train_data, val_data, test_data


def plot_sample_images_and_histogram(train_data, num_samples=5):
    # Plot a few sample images
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        image, label = random.choice(train_data)
        
        # Plot the image
        axes[0, i].imshow(image, cmap='gray')
        axes[0, i].set_title(f"Label: {label}")
        axes[0, i].axis('off')
        
        # Plot the histogram
        axes[1, i].hist(image.ravel(), bins=256, range=(0.0, 1.0), fc='black', ec='black')
        axes[1, i].set_title("Pixel Value Histogram")

    plt.tight_layout()
    plt.show()


### COUNT LABELS IN SET ###
def count_labels_set(data):
    # Initialize a dictionary to store the counts
    counts = {'Benign': 0, 'Malignant': 0}

    # Loop through the data
    for item in data:
        # Increment the count for the corresponding label
        if item[1] in counts:
            counts[item[1]] += 1

    return len(data), counts


### EVALUATION METRICS, PLOTS, AND LOGS ###
label_names = {0: 'Benign', 1: 'Malignant'}

def log_experiment(net_final, epochs_list, train_accs, val_accs, train_losses, val_losses, test_loader, device, eval_batchsize, model_type='resnet'):

    def plot_training_curves(epochs_list, train_losses, val_losses):
        # Log data to wandb
        wandb.log({
            "epochs": epochs_list,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "train_losses": train_losses,
            "val_losses": val_losses
        })

        # Plot for accuracy and save to wandb
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_list, train_accs, 'b-', label='Training Accuracy')
        plt.plot(epochs_list, val_accs, 'r-', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0.5, 1)
        plt.title('Prediction Accuracy Over Time')
        plt.legend()
        plt.savefig("accuracy_plot.png")
        plt.close()
        wandb.log({"Accuracy Over Time": wandb.Image("accuracy_plot.png")})
        
        # Delete the image file
        os.remove("accuracy_plot.png")

        # Plot for validation loss and save to wandb
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_list, train_losses, 'b--', label='Training Loss')
        plt.plot(epochs_list, val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, 1)
        plt.title('Validation Loss Over Time')
        plt.legend()
        plt.savefig("loss_plot.png")
        plt.close()
        wandb.log({"Loss Over Time": wandb.Image("loss_plot.png")})

        # Delete the image file
        os.remove("loss_plot.png")

    def plot_confusion_matrix(all_targets, all_predictions, label_names):
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        # Setting tick labels
        ax.set_xticklabels(label_names)
        ax.set_yticklabels(label_names)
        plt.xticks(rotation=45)  # Optional: rotate for better display if needed
        plt.yticks(rotation=45)
        plt.savefig("confusion_matrix.png")
        plt.close()
        return "confusion_matrix.png"

    # ROC curve plotter
    def plot_roc_curve(targets, probabilities):
        fpr, tpr, thresholds = roc_curve(targets, probabilities, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig("roc_curve.png")
        plt.close()
        return roc_auc, "roc_curve.png"

    def log_example_images(inputs, targets, predicted_class, label_names, num_viz):
        fig, axs = plt.subplots(nrows=num_viz, ncols=1, figsize=(10, num_viz * 2))
        for i in range(num_viz):
            img = inputs[i].cpu().float()
            img = img.permute(1, 2, 0)  # Convert from CxHxW to HxWxC
            axs[i].imshow(img, cmap='gray')
            axs[i].set_title(f"Label: {label_names[targets[i].item()]} - Prediction: {label_names[predicted_class[i].item()]}")
            axs[i].axis('off')
        plt.tight_layout()
        plt.savefig("example_images.png")
        plt.close()

        # Log the combined image to wandb
        wandb.log({"Example Images with Predictions": wandb.Image("example_images.png")})

        # Delete the image file
        os.remove("example_images.png")

    def plot_sensitivity_specificity_curve(targets, probabilities):
        fpr, tpr, thresholds = roc_curve(targets, probabilities, pos_label=1)
        specificity = 1 - fpr  # Specificity is 1 - False Positive Rate
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, tpr, label='Sensitivity (Recall)', color='blue')
        plt.plot(thresholds, specificity, label='Specificity', color='green')
        plt.title('Sensitivity and Specificity vs Thresholds')
        plt.xlabel('Threshold')
        plt.ylabel('Percentage')
        plt.legend(loc='best')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.savefig("sensitivity_specificity.png")
        plt.close()

        # Log the combined image to wandb
        wandb.log({"Plots of the Sensitivity-Specificity curves": wandb.Image("sensitivity_specificity.png")})

        # Delete the image file
        os.remove("sensitivity_specificity.png")


    plot_training_curves(epochs_list, train_losses, val_losses)

    # Initialize and predict settings
    num_viz = 10
    viz_index = random.randint(0, len(test_loader) // eval_batchsize)
    all_targets, all_predictions, all_probabilities = [], [], []
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0

    with torch.no_grad():
        # Iterate through the test data batches
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass through the network
            if model_type == 'resnet':
                outputs = net_final(inputs)
            else:
                outputs = net_final(inputs).logits

            # Calculate class probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get the predicted class with the highest score
            _, predicted_class = outputs.max(1)

            # Store the probabilities, targets, and predictions for later analysis
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted_class.cpu().numpy())

            # Update count of true positives, true negatives, false positives, and false negatives
            true_positives += ((predicted_class == 1) & (targets == 1)).sum().item()
            true_negatives += ((predicted_class == 0) & (targets == 0)).sum().item()
            false_positives += ((predicted_class == 1) & (targets == 0)).sum().item()
            false_negatives += ((predicted_class == 0) & (targets == 1)).sum().item()

            # If current batch index matches the visualization index, log example images
            if batch_index == viz_index:
                log_example_images(inputs[:num_viz], targets[:num_viz], predicted_class[:num_viz], label_names, num_viz)


    plot_sensitivity_specificity_curve(all_targets, all_probabilities)


    # Log metrics
    metrics = {
        "Accuracy": accuracy_score(all_targets, all_predictions) * 100,
        "Recall": recall_score(all_targets, all_predictions, pos_label=1) * 100,
        "Precision": precision_score(all_targets, all_predictions, pos_label=1) * 100,
        "F1 Score": f1_score(all_targets, all_predictions, pos_label=1)
    }
    wandb.log(metrics)

    # Detailed report
    detailed_report = classification_report(all_targets, all_predictions, target_names=['Benign', 'Malignant'], output_dict=True)
    wandb.log({"Classification Report": detailed_report})

    # Plot and log ROC curve
    auc_score, roc_image_path = plot_roc_curve(all_targets, all_probabilities)
    wandb.log({"ROC Curve": wandb.Image(roc_image_path), "AUC Score": auc_score})

    # Delete the image file
    os.remove("roc_curve.png")

    cm_image_path = plot_confusion_matrix(all_targets, all_predictions, label_names)
    wandb.log({"Confusion Matrix": wandb.Image(cm_image_path)})

    # Delete the image file
    os.remove("confusion_matrix.png")

    # Save and log model
    model_name = f"model_{wandb.run.id}_ROC_{auc_score:.2f}.pth"
    torch.save(net_final.state_dict(), model_name)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_name)
    wandb.log_artifact(artifact)

    # Finish the run
    wandb.finish()


### FUNCTION TO TEST TRAINED MODELS ###
def plot_imgbatch(imgs):
    imgs = imgs.cpu()  # Transfer images back to CPU for visualization
    imgs = imgs.float()
    plt.figure(figsize=(15, 3 * (imgs.shape[0] // 5)))
    grid_img = make_grid(imgs, nrow=5)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()


def plot_roc_curve(targets, probabilities):
    fpr, tpr, thresholds = roc_curve(targets, probabilities, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Baseline (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc


def plot_sensitivity_specificity_curve(targets, probabilities):
    fpr, tpr, thresholds = roc_curve(targets, probabilities, pos_label=1)
    specificity = 1 - fpr  # Specificity is 1 - False Positive Rate
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, tpr, label='Sensitivity (Recall)', color='blue')
    plt.plot(thresholds, specificity, label='Specificity', color='green')
    plt.title('Sensitivity and Specificity vs Thresholds')
    plt.xlabel('Threshold')
    plt.ylabel('Percentage')
    plt.legend(loc='best')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.show()


def plot_precision_recall_curve(targets, probabilities):
    precision, recall, _ = precision_recall_curve(targets, probabilities, pos_label=1)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()
    return pr_auc


def calculate_balanced_accuracy(targets, predictions):
    return balanced_accuracy_score(targets, predictions)
    

def evaluate_model(test_loader, net_final, device, num_viz=10, eval_batchsize=32, model_type='resnet'):
    all_targets = []
    all_predictions = []
    all_probabilities = []

    true_positives = 0  # Malignant correctly classified
    true_negatives = 0  # Benign correctly classified
    false_positives = 0  # Benign incorrectly classified as Malignant
    false_negatives = 0  # Malignant incorrectly classified as Benign

    # Randomly select a batch index for visualization
    viz_index = random.randint(0, len(test_loader) // eval_batchsize)

    # Set the model to evaluation mode
    net_final.eval()

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass through the model
            if model_type == 'resnet':
                outputs = net_final(inputs)
            else:
                outputs = net_final(inputs).logits

            # Ensure model outputs are valid
            if outputs is None or not isinstance(outputs, torch.Tensor):
                raise ValueError(f"Unexpected model output: {outputs}")

            # Compute probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted_class = outputs.max(1)

            # Store results for later analysis
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted_class.cpu().numpy())

            # Count classification results
            true_positives += ((predicted_class == 1) & (targets == 1)).sum().item()
            true_negatives += ((predicted_class == 0) & (targets == 0)).sum().item()
            false_positives += ((predicted_class == 1) & (targets == 0)).sum().item()
            false_negatives += ((predicted_class == 0) & (targets == 1)).sum().item()

            if batch_index == viz_index:
                print('Example Images:')
                plot_imgbatch(inputs[:num_viz])
                print('Target labels (Malignant/Benign):')
                print([label_names[label.item()] for label in targets[:num_viz]])
                print('\nClassifier predictions (Malignant/Benign):')
                print([label_names[label.item()] for label in predicted_class[:num_viz]])

    # Calculating metrics and classification report
    accuracy = accuracy_score(all_targets, all_predictions) * 100
    recall = recall_score(all_targets, all_predictions, pos_label=1) * 100
    precision = precision_score(all_targets, all_predictions, pos_label=1) * 100
    f1 = f1_score(all_targets, all_predictions, pos_label=1)
    print("\n-----Before Threshold Adjustment-----")

    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    class_names = ['Benign', 'Malignant']  # Class names as per your setup

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Overall Recall (Sensitivity): {recall:.2f}%")
    print(f"Overall Precision: {precision:.2f}%")
    print(f"Overall F1 Score: {f1:.2f}")
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_predictions, target_names=['Benign', 'Malignant']))

    # Print counts of all classification results
    print(f"True Positives (Malignant correctly classified): {true_positives}")
    print(f"True Negatives (Benign correctly classified): {true_negatives}")
    print(f"False Positives (Benign incorrectly classified as Malignant): {false_positives}")
    print(f"False Negatives (Malignant incorrectly classified as Benign): {false_negatives}")

    # Plot ROC curve and print AUC
    auc_score = plot_roc_curve(all_targets, all_probabilities)
    print(f"AUC Score: {auc_score:.2f}")

    # Plot Precision-Recall curve and print AUC
    pr_auc_score = plot_precision_recall_curve(all_targets, all_probabilities)
    print(f"Precision-Recall AUC Score: {pr_auc_score:.2f}")

    plot_sensitivity_specificity_curve(all_targets, all_probabilities)

    # Evaluate different thresholds
    thresholds = np.linspace(0, 1, num=100)
    best_threshold = 0
    best_metric = 0

    for threshold in thresholds:
        predictions = (all_probabilities > threshold).astype(int)
        # Choose the metric to maximize
        metric_value = calculate_balanced_accuracy(all_targets, predictions)
        
        if metric_value > best_metric:
            best_metric = metric_value
            best_threshold = threshold

    print(f"Best Threshold: {best_threshold}")
    print(f"Best Metric Value (Balanced Accuracy): {best_metric}")

    # Convert probabilities to predictions based on the best threshold
    predictions = (all_probabilities > best_threshold).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, predictions)
    class_names = ['Benign', 'Malignant']  # Class names as per your setup

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Printing classification metrics
    print("Detailed Classification Report:")
    print("-----After Threshold Adjustment-----")
    print(classification_report(all_targets, predictions, target_names=class_names))

    # Calculate and print specific statistics
    accuracy = accuracy_score(all_targets, predictions) * 100
    precision = precision_score(all_targets, predictions, pos_label=1) * 100
    recall = recall_score(all_targets, predictions, pos_label=1) * 100
    f1 = f1_score(all_targets, predictions, pos_label=1)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Overall Recall (Sensitivity): {recall:.2f}%")
    print(f"Overall Precision: {precision:.2f}%")
    print(f"Overall F1 Score: {f1:.2f}")

    # Extracting elements from confusion matrix for detailed counts
    tn, fp, fn, tp = cm.ravel()
    print(f"True Positives (Malignant correctly classified): {tp}")
    print(f"True Negatives (Benign correctly classified): {tn}")
    print(f"False Positives (Benign incorrectly classified as Malignant): {fp}")
    print(f"False Negatives (Malignant incorrectly classified as Benign): {fn}")


### PLOT GRID OF SAMPLE IMAGES ###
def sample_grid(data, num_samples=8):
    # Select 8 benign and 8 malignant samples
    benign_samples = [item for item in data if item[1] == 'Benign'][:num_samples]
    malignant_samples = [item for item in data if item[1] == 'Malignant'][:num_samples]

    # Combine the samples into a single list
    samples_to_plot = benign_samples + malignant_samples

    # Create a figure with 4x4 subplots
    _, axes = plt.subplots(4, 4, figsize=(10, 10))

    for i, (image_np, status) in enumerate(samples_to_plot):
        ax = axes[i // 4, i % 4]
        ax.imshow(image_np, cmap='gray')
        ax.axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_samples_status(data, num_samples=5):
    # Select the first num_samples from the data
    samples_to_plot = data[:num_samples]

    for i, (image_np, status) in enumerate(samples_to_plot):
        # Display each image with its status
        plt.imshow(image_np, cmap='gray')
        plt.title('Status: ' + status)
        plt.axis('off')
        plt.show()


### SHOW IMAGES FROM DATALOADER AND COUNT LABELS ###
def show_images(dataloader, num_images):
    # Get an iterator from the dataloader
    dataiter = iter(dataloader)
    # Get the first batch of images and labels
    images, labels = next(dataiter)

    # Set the figure size for the plot
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        # Create a subplot for each image
        _ = plt.subplot(1, num_images, i + 1)
        # Change the image format from (C, H, W) to (H, W, C) and convert to numpy array
        img = images[i].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.show()


def count_labels(dataloader):
    label_counts = np.zeros(2)  # 2 classes
    for _, labels in dataloader:
        # Count unique labels in the current batch
        unique, counts = np.unique(labels.numpy(), return_counts=True)
        label_counts[unique] += counts

    print(f"Number of benign (label 0) images: {label_counts[0]}")
    print(f"Number of malignant (label 1) images: {label_counts[1]}")


def show_images_grid(dataloader, num_images):
    # Get an iterator from the dataloader
    dataiter = iter(dataloader)
    # Get the first batch of images
    images, _ = next(dataiter)
    
    plt.figure(figsize=(12, 12))

    # Randomly select num_images indices from the images
    indices = random.sample(range(len(images)), num_images)

    for i, idx in enumerate(indices):
        # Create a subplot for each image
        plt.subplot(3, 3, i + 1)
        # Change the image format from (C, H, W) to (H, W, C) and convert to numpy array
        img = images[idx].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_images_and_histograms(dataloader, num_images):
    # Get an iterator from the dataloader
    dataiter = iter(dataloader)
    # Get the first batch of images and labels
    images, labels = next(dataiter)

    plt.figure(figsize=(12, 8))

    # Mapping from label to its name
    label_map = {0: 'Benign', 1: 'Malignant'}

    for i in range(num_images):
        # Change the image format from (C, H, W) to (H, W, C) and convert to numpy array
        img = images[i].permute(1, 2, 0).numpy()

        # Plot the image
        ax_img = plt.subplot(2, num_images, i + 1)
        plt.imshow(img)
        plt.title(f'Label: {label_map[labels[i].item()]}')
        plt.axis('off')

        # Plot the histogram of the image's pixel values
        ax_hist = plt.subplot(2, num_images, num_images + i + 1)
        plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='black', ec='black')
        plt.title("Pixel Value Histogram")

    plt.tight_layout()
    plt.show()