import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import os

def load_data(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_execution_time(metadata):
    """Extract execution time from metadata"""
    if not metadata or not isinstance(metadata, list):
        return None
    try:
        metadata_str = metadata[0]
        if isinstance(metadata_str, str):
            metadata_dict = json.loads(metadata_str)
            return metadata_dict.get("execution time")
    except:
        pass
    return None

def analyze_results():
    """Main analysis function"""
    
    # Load data
    print("Loading data...")
    before_data = load_data('data/gpt5_no_memory.json')
    after_data = load_data('data/gpt5_with_memory.json')
    
    # Create DataFrame for analysis
    results = []
    
    for before_item, after_item in zip(before_data, after_data):
        # Ensure we're comparing the same question
        if before_item['question_id'] != after_item['question_id']:
            print(f"Warning: Question ID mismatch: {before_item['question_id']} vs {after_item['question_id']}")
            continue
            
        before_correct = before_item['pass@1'] == 1.0
        after_correct = after_item['pass@1'] == 1.0
        
        before_time = extract_execution_time(before_item.get('metadata'))
        after_time = extract_execution_time(after_item.get('metadata'))
        
        results.append({
            'question_id': before_item['question_id'],
            'question_title': before_item['question_title'],
            'difficulty': before_item['difficulty'],
            'before_correct': before_correct,
            'after_correct': after_correct,
            'before_time': before_time,
            'after_time': after_time,
            'improved': not before_correct and after_correct,
            'regressed': before_correct and not after_correct
        })
    
    df = pd.DataFrame(results)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Set improved style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Set font sizes and style
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # 1. Questions that improved (wrong -> right)
    print("Generating improvement analysis...")
    create_improvement_charts(df)
    
    # 2. Questions that regressed (right -> wrong)
    print("Generating regression analysis...")
    create_regression_charts(df)
    
    # 3. Execution time comparison
    print("Generating execution time analysis...")
    create_execution_time_charts(df)
    
    # 4. Correct answers comparison
    print("Generating correct answers comparison...")
    create_correct_answers_charts(df)
    
    print("Analysis complete! Charts saved in plots/ directory.")

def create_improvement_charts(df):
    """Create charts for questions that improved"""
    
    # Overall improvement
    total_improved = df['improved'].sum()
    total_questions = len(df)
    improvement_rate = (total_improved / total_questions) * 100
    
    # By difficulty
    difficulty_improvements = df.groupby('difficulty').agg({
        'improved': ['sum', 'count']
    }).round(2)
    difficulty_improvements.columns = ['improved_count', 'total_count']
    difficulty_improvements['improvement_rate'] = (difficulty_improvements['improved_count'] / difficulty_improvements['total_count'] * 100).round(2)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Overall improvement pie chart
    labels = ['Improved\n(wrong → right)', 'No Improvement']
    sizes = [total_improved, total_questions - total_improved]
    colors = ['#2ecc71', '#ecf0f1']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                       startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    # Make the percentage text more visible with better contrast
    for i, autotext in enumerate(autotexts):
        if i == 0:  # Improved section (green background)
            autotext.set_color('#2c3e50')
        else:  # No improvement section (light gray background)
            autotext.set_color('#2c3e50')  # Dark blue-gray for better contrast
        autotext.set_fontweight('bold')
    
    ax1.set_title(f'Overall Improvement Rate\n{total_improved}/{total_questions} questions ({improvement_rate:.1f}%)', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Improvement by difficulty
    difficulties = difficulty_improvements.index
    improvement_rates = difficulty_improvements['improvement_rate']
    improved_counts = difficulty_improvements['improved_count']
    total_counts = difficulty_improvements['total_count']
    
    bars = ax2.bar(difficulties, improvement_rates, color='#3498db', alpha=0.8, edgecolor='#2980b9', linewidth=1.5)
    ax2.set_title('Improvement Rate by Difficulty\n(wrong → right)', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Improvement Rate (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Difficulty', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, rate, count, total) in enumerate(zip(bars, improvement_rates, improved_counts, total_counts)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}/{total}\n({rate:.1f}%)', ha='center', va='bottom', 
                fontweight='bold', fontsize=11, color='#2c3e50')
    
    # Set y-axis limits for better visualization
    ax2.set_ylim(0, max(improvement_rates) * 1.2)
    
    plt.tight_layout()
    plt.savefig('plots/gpt5_improvement_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save detailed data
    with open('plots/improvement_summary.txt', 'w') as f:
        f.write("IMPROVEMENT ANALYSIS (Wrong → Right)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall: {total_improved}/{total_questions} questions ({improvement_rate:.1f}%)\n\n")
        f.write("By Difficulty:\n")
        for diff in difficulties:
            rate = difficulty_improvements.loc[diff, 'improvement_rate']
            count = difficulty_improvements.loc[diff, 'improved_count']
            total = difficulty_improvements.loc[diff, 'total_count']
            f.write(f"  {diff}: {count}/{total} ({rate:.1f}%)\n")

def create_regression_charts(df):
    """Create charts for questions that regressed"""
    
    # Overall regression
    total_regressed = df['regressed'].sum()
    total_questions = len(df)
    regression_rate = (total_regressed / total_questions) * 100
    
    # By difficulty
    difficulty_regressions = df.groupby('difficulty').agg({
        'regressed': ['sum', 'count']
    }).round(2)
    difficulty_regressions.columns = ['regressed_count', 'total_count']
    difficulty_regressions['regression_rate'] = (difficulty_regressions['regressed_count'] / difficulty_regressions['total_count'] * 100).round(2)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Overall regression pie chart
    labels = ['Regressed\n(right → wrong)', 'No Regression']
    sizes = [total_regressed, total_questions - total_regressed]
    colors = ['#e74c3c', '#ecf0f1']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                       startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    # Make the percentage text more visible with better contrast
    for i, autotext in enumerate(autotexts):
        if i == 0:  # Regressed section (red background)
            autotext.set_color('#2c3e50')
            autotext.set_fontsize(11)
            # Move percentage text slightly upward for better positioning
            x, y = autotext.get_position()
            autotext.set_position((x - 0.02, y + 0.12))
            # Move percentage text slightly upward for better positioning
            x, y = autotext.get_position()
            autotext.set_position((x, y + 0.08))
        else:  # No regression section (light gray background)
            autotext.set_color('#2c3e50')  # Dark blue-gray for better contrast
                  # Set font size for no regression percentage
        autotext.set_fontweight('bold')
    
    ax1.set_title(f'Overall Regression Rate\n{total_regressed}/{total_questions} questions ({regression_rate:.1f}%)', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Regression by difficulty
    difficulties = difficulty_regressions.index
    regression_rates = difficulty_regressions['regression_rate']
    regressed_counts = difficulty_regressions['regressed_count']
    total_counts = difficulty_regressions['total_count']
    
    bars = ax2.bar(difficulties, regression_rates, color='#e74c3c', alpha=0.8, edgecolor='#c0392b', linewidth=1.5)
    ax2.set_title('Regression Rate by Difficulty\n(right → wrong)', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Regression Rate (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Difficulty', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, rate, count, total) in enumerate(zip(bars, regression_rates, regressed_counts, total_counts)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}/{total}\n({rate:.1f}%)', ha='center', va='bottom', 
                fontweight='bold', fontsize=11, color='#2c3e50')
    
    # Set y-axis limits for better visualization
    ax2.set_ylim(0, max(regression_rates) * 1.2)
    
    plt.tight_layout()
    plt.savefig('plots/gpt5_regression_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save detailed data
    with open('plots/gpt5_regression_summary.txt', 'w') as f:
        f.write("REGRESSION ANALYSIS (Right → Wrong)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall: {total_regressed}/{total_questions} questions ({regression_rate:.1f}%)\n\n")
        f.write("By Difficulty:\n")
        for diff in difficulties:
            rate = difficulty_regressions.loc[diff, 'regression_rate']
            count = difficulty_regressions.loc[diff, 'regressed_count']
            total = difficulty_regressions.loc[diff, 'total_count']
            f.write(f"  {diff}: {count}/{total} ({rate:.1f}%)\n")

def create_execution_time_charts(df):
    """Create simplified execution time comparison charts"""
    
    # Filter out None values
    time_df = df.dropna(subset=['before_time', 'after_time'])
    
    # Overall execution time comparison
    before_mean = time_df['before_time'].mean()
    after_mean = time_df['after_time'].mean()
    time_change = ((after_mean - before_mean) / before_mean) * 100
    
    # By difficulty
    difficulty_times = time_df.groupby('difficulty').agg({
        'before_time': 'mean',
        'after_time': 'mean'
    }).round(4)
    difficulty_times['time_change_pct'] = ((difficulty_times['after_time'] - difficulty_times['before_time']) / 
                                         difficulty_times['before_time'] * 100).round(2)
    
    # Create a single, clean figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Overall execution time comparison (simplified)
    labels = ['Without Cipher', 'With Cipher']
    times = [before_mean, after_mean]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax1.bar(labels, times, color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=1.5)
    ax1.set_title(f'Overall Average Execution Time\nChange: {time_change:+.1f}%', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{time:.3f}s', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 2. Execution time by difficulty (simplified)
    difficulties = difficulty_times.index
    before_times = difficulty_times['before_time']
    after_times = difficulty_times['after_time']
    
    x = np.arange(len(difficulties))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, before_times, width, label='Without Cipher', color='#3498db', alpha=0.8, edgecolor='#2980b9', linewidth=1)
    bars2 = ax2.bar(x + width/2, after_times, width, label='With Cipher', color='#e74c3c', alpha=0.8, edgecolor='#c0392b', linewidth=1)
    
    ax2.set_title('Average Execution Time by Difficulty', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Difficulty', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(difficulties)
    ax2.legend(fontsize=12, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars (simplified)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.1:  # Only show labels for significant values
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                        f'{height:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/gpt5_execution_time_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save detailed data
    with open('plots/gpt5_execution_time_summary.txt', 'w') as f:
        f.write("EXECUTION TIME ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Average:\n")
        f.write(f"  Before: {before_mean:.4f} seconds\n")
        f.write(f"  After:  {after_mean:.4f} seconds\n")
        f.write(f"  Change: {time_change:+.1f}%\n\n")
        f.write("By Difficulty:\n")
        for diff in difficulties:
            before = difficulty_times.loc[diff, 'before_time']
            after = difficulty_times.loc[diff, 'after_time']
            change = difficulty_times.loc[diff, 'time_change_pct']
            f.write(f"  {diff}:\n")
            f.write(f"    Before: {before:.4f}s\n")
            f.write(f"    After:  {after:.4f}s\n")
            f.write(f"    Change: {change:+.1f}%\n\n")

def create_correct_answers_charts(df):
    """Create charts for correct answers comparison"""
    
    # Overall correct answers comparison
    before_correct = df['before_correct'].sum()
    after_correct = df['after_correct'].sum()
    total_questions = len(df)
    before_accuracy = (before_correct / total_questions) * 100
    after_accuracy = (after_correct / total_questions) * 100
    accuracy_change = after_accuracy - before_accuracy
    
    # By difficulty
    difficulty_accuracy = df.groupby('difficulty').agg({
        'before_correct': ['sum', 'count'],
        'after_correct': 'sum'
    }).round(2)
    difficulty_accuracy.columns = ['before_correct_count', 'total_count', 'after_correct_count']
    difficulty_accuracy['before_accuracy'] = (difficulty_accuracy['before_correct_count'] / difficulty_accuracy['total_count'] * 100).round(2)
    difficulty_accuracy['after_accuracy'] = (difficulty_accuracy['after_correct_count'] / difficulty_accuracy['total_count'] * 100).round(2)
    difficulty_accuracy['accuracy_change'] = difficulty_accuracy['after_accuracy'] - difficulty_accuracy['before_accuracy']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Overall accuracy comparison
    labels = ['Without Cipher', 'With Cipher']
    accuracies = [before_accuracy, after_accuracy]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax1.bar(labels, accuracies, color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=1.5)
    ax1.set_title(f'Overall Accuracy Comparison\nChange: {accuracy_change:+.1f}%', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 105)  # Give extra space for text labels
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, accuracy, correct_count in zip(bars, accuracies, [before_correct, after_correct]):
        height = bar.get_height()
        # Position text inside the bar if it's tall enough, otherwise above
        if height > 20:
            text_y = height * 0.6
            text_color = 'black'
        else:
            text_y = height + 2
            text_color = '#2c3e50'
        
        ax1.text(bar.get_x() + bar.get_width()/2., text_y,
                f'{correct_count}/{total_questions}\n({accuracy:.1f}%)', ha='center', va='center', 
                fontweight='bold', fontsize=11, color=text_color)
    
    # Accuracy by difficulty
    difficulties = difficulty_accuracy.index
    before_accuracies = difficulty_accuracy['before_accuracy']
    after_accuracies = difficulty_accuracy['after_accuracy']
    
    x = np.arange(len(difficulties))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, before_accuracies, width, label='Without Cipher', color='#3498db', alpha=0.8, edgecolor='#2980b9', linewidth=1)
    bars2 = ax2.bar(x + width/2, after_accuracies, width, label='With Cipher', color='#e74c3c', alpha=0.8, edgecolor='#c0392b', linewidth=1)
    
    ax2.set_title('Accuracy by Difficulty', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Difficulty', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(difficulties)
    ax2.set_ylim(0, 105)  # Give extra space for text labels
    ax2.legend(fontsize=12, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars, accuracies, correct_counts in [(bars1, before_accuracies, difficulty_accuracy['before_correct_count']), 
                                            (bars2, after_accuracies, difficulty_accuracy['after_correct_count'])]:
        for bar, accuracy, correct_count, total_count in zip(bars, accuracies, correct_counts, difficulty_accuracy['total_count']):
            height = bar.get_height()
            # Position text inside the bar if it's tall enough, otherwise above
            if height > 15:
                text_y = height * 0.6
                text_color = 'black'
            else:
                text_y = height + 2
                text_color = '#2c3e50'
            
            ax2.text(bar.get_x() + bar.get_width()/2., text_y,
                    f'{int(correct_count)}/{int(total_count)}\n({accuracy:.1f}%)', ha='center', va='center', 
                    fontsize=9, fontweight='bold', color=text_color)
    
    plt.tight_layout()
    plt.savefig('plots/gpt5_correct_answers_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save detailed data
    with open('plots/gpt5_correct_answers_summary.txt', 'w') as f:
        f.write("CORRECT ANSWERS ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy:\n")
        f.write(f"  Without Cipher: {before_correct}/{total_questions} ({before_accuracy:.1f}%)\n")
        f.write(f"  With Cipher:    {after_correct}/{total_questions} ({after_accuracy:.1f}%)\n")
        f.write(f"  Change:         {accuracy_change:+.1f}%\n\n")
        f.write("By Difficulty:\n")
        for diff in difficulties:
            before_acc = difficulty_accuracy.loc[diff, 'before_accuracy']
            after_acc = difficulty_accuracy.loc[diff, 'after_accuracy']
            change = difficulty_accuracy.loc[diff, 'accuracy_change']
            before_count = int(difficulty_accuracy.loc[diff, 'before_correct_count'])
            after_count = int(difficulty_accuracy.loc[diff, 'after_correct_count'])
            total_count = int(difficulty_accuracy.loc[diff, 'total_count'])
            f.write(f"  {diff}:\n")
            f.write(f"    Without Cipher: {before_count}/{total_count} ({before_acc:.1f}%)\n")
            f.write(f"    With Cipher:    {after_count}/{total_count} ({after_acc:.1f}%)\n")
            f.write(f"    Change:         {change:+.1f}%\n\n")

if __name__ == "__main__":
    analyze_results()
