import matplotlib.pyplot as plt
import json

def plot_metrics_json(input, output):
    # Read the JSON file and handle multiple separate JSON objects
    metrics_list = []
    with open(input, 'r') as f:
        for line in f:
            metrics_list.append(json.loads(line))

    # Combine individual dictionaries into a single dictionary
    combined_metrics = {}
    for entry in metrics_list:
        for key, value in entry.items():
            combined_metrics.setdefault(key, []).append(value)

    losses = ['loss_ce', 'loss_dice', 'loss_mask', 'total_loss']
    val_losses = ['val_loss_ce', 'val_loss_dice', 'val_loss_mask', 'total_val_loss']
    
    val_losses_filtered = [val_loss if val_loss in combined_metrics.keys() else None for val_loss in val_losses]
    metric_pairs = [(loss, val_loss) for loss, val_loss in zip(losses, val_losses_filtered)]
    metric_pairs.append(('lr', None))
    print(metric_pairs)

    plt.figure(figsize=(20, 20))

    for idx, (metric, val_metric) in enumerate(metric_pairs, 1):
        plt.subplot(len(metric_pairs), 1, idx)
        
        # Plot metric and annotate the last point with its rounded value
        iterations = combined_metrics['iteration']
        metric_values = combined_metrics[metric]
        plt.plot(iterations, metric_values, '-o', label=metric)
        plt.annotate(f"{round(metric_values[-1], 3)}", (iterations[-1], metric_values[-1]))

        # Plot validation metric and annotate the last point with its rounded value, if it exists
        if val_metric:
            val_metric_values = combined_metrics[val_metric]
            plt.plot(iterations, val_metric_values, '-o', label=val_metric, alpha=0.7)
            plt.annotate(f"{round(val_metric_values[-1], 3)}", (iterations[-1], val_metric_values[-1]), alpha=0.7)
        
        plt.xlabel('Iteration')
        plt.ylabel(metric)
        plt.title(metric)
        plt.legend()


    plt.tight_layout()
    plt.savefig(output)

    
if __name__=="__main__":
    # Call the function with your filename
    #plot_metrics_json(input="/workspace/shared/output_lr0.001/metrics.json", 
    #                       output="/workspace/shared/output_lr0.001/plot.png")

    #plot_metrics_json(input="/workspace/shared/output_lr0.0001/metrics.json", 
    #                       output="/workspace/shared/output_lr0.0001/plot.png")
    
    folder_with_val = "r50/C1_lr0.0001" 
    folder_no_val = "r50_no_val/C1"
    plot_metrics_json(input=f"/root/autofish_training/output/{folder_no_val}/metrics.json", 
                      output=f"/root/autofish_training/output/{folder_no_val}/plot.png")