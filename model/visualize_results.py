import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

def qlike_loss(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1e-8)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

def main():
    # 1. Load Data
    cnn_path = Path("data/cnn_har_ks_multitask_predictions_best_params.csv")
    benchmark_path = Path("data/benchmark_predictions.csv")
    
    if not cnn_path.exists() or not benchmark_path.exists():
        print("Error: Prediction files not found.")
        return

    df_cnn = pd.read_csv(cnn_path, parse_dates=['date'])
    df_bench = pd.read_csv(benchmark_path, parse_dates=['date'])

    # Merge on date
    df = pd.merge(df_cnn, df_bench[['date', 'garch_pred', 'ewma_pred']], on='date', how='inner')
    df.set_index('date', inplace=True)
    
    # 2. Metrics
    models = ['cnn_pred', 'garch_pred', 'ewma_pred']
    labels = ['CNN-HAR-KS', 'GARCH(1,1)', 'EWMA']
    colors = ['#d62728', '#1f77b4', '#2ca02c'] # Red, Blue, Green
    
    metrics = []
    for model, label in zip(models, labels):
        mse = mean_squared_error(df['true_rv'], df[model])
        mae = mean_absolute_error(df['true_rv'], df[model])
        qlike = qlike_loss(df['true_rv'], df[model])
        metrics.append({'Model': label, 'MSE': mse, 'MAE': mae, 'QLIKE': qlike})

    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)
    metrics_df.to_csv("data/model_performance_metrics.csv", index=False)

    # 3. Plotting
    plt.style.use('bmh')
    plt.rcParams['figure.figsize'] = (12, 6)

    # A. Time Series
    plt.figure()
    plt.plot(df.index, df['true_rv'], label='True RV', color='black', alpha=0.9, linewidth=1)
    for model, label, color in zip(models, labels, colors):
        plt.plot(df.index, df[model], label=label, color=color, alpha=0.7, linewidth=1, linestyle='--' if label!='CNN-HAR-KS' else '-')
    
    plt.title('Volatility Forecast Comparison')
    plt.ylabel('Realized Volatility')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/volatility_forecast_comparison.png', dpi=300)
    print("Saved time series plot.")

    # B. Cumulative MSE Improvement
    plt.figure()
    cum_mse_cnn = np.cumsum((df['true_rv'] - df['cnn_pred'])**2)
    cum_mse_ewma = np.cumsum((df['true_rv'] - df['ewma_pred'])**2)
    
    plt.plot(df.index, cum_mse_ewma - cum_mse_cnn, label='Cumulative MSE Gain (EWMA - CNN)', color='green')
    plt.axhline(0, color='black', linestyle=':')
    plt.title('Cumulative Outperformance of CNN over EWMA')
    plt.ylabel('Cumulative Squared Error Difference')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/cumulative_mse_improvement.png', dpi=300)
    print("Saved cumulative MSE plot.")

    # C. Scatter Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    limit_max = max(df['true_rv'].max(), df['cnn_pred'].max()) * 1.05
    
    for i, (model, label, color) in enumerate(zip(models, labels, colors)):
        ax = axes[i]
        corr = df[['true_rv', model]].corr().iloc[0, 1]
        ax.scatter(df['true_rv'], df[model], alpha=0.5, s=15, color=color)
        ax.plot([0, limit_max], [0, limit_max], 'k--', alpha=0.5)
        ax.set_title(f'{label} (Corr: {corr:.3f})')
        ax.set_xlabel('True RV')
        if i == 0: ax.set_ylabel('Predicted RV')
        ax.set_xlim(0, limit_max)
        ax.set_ylim(0, limit_max)

    plt.tight_layout()
    plt.savefig('data/prediction_scatter_plots.png', dpi=300)
    print("Saved scatter plots.")

    # D. Rolling Accuracy
    if 'true_cls' in df.columns and 'cnn_cls_pred' in df.columns:
        plt.figure()
        rolling_acc = (df['true_cls'] == df['cnn_cls_pred']).rolling(window=30).mean()
        overall = accuracy_score(df['true_cls'], df['cnn_cls_pred'])
        
        plt.plot(df.index, rolling_acc, label='30-Day Rolling Accuracy', color='purple')
        plt.axhline(overall, color='gray', linestyle='--', label=f'Mean Acc: {overall:.1%}') # pyright: ignore[reportArgumentType]
        plt.title('Classification Accuracy Trend')
        plt.ylim(0.3, 0.8)
        plt.legend()
        plt.tight_layout()
        plt.savefig('data/classification_accuracy_rolling.png', dpi=300)
        print("Saved accuracy plot.")

if __name__ == "__main__":
    main()
