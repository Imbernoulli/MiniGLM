import os
import matplotlib.pyplot as plt

def visualize_loss(train_loss_list, train_interval, val_loss_list, val_interval, dataset, out_dir):
    train_epochs = [i*train_interval for i in range(len(train_loss_list))]
    val_epochs = [i*val_interval for i in range(len(val_loss_list))]

    plt.figure(figsize=(10,6))
    
    plt.plot(train_epochs, train_loss_list, label="Training Loss", color="blue", marker='o')
    plt.plot(val_epochs, val_loss_list, label="Validation Loss", color="red", marker='x')
    
    plt.title(f"Training and Validation Loss for {dataset}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, 'loss.png'))

    plt.show()
